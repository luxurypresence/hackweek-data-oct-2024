import uuid

import pandas as pd
from numpy import ndarray

from s3_service import S3Service
from chatgpt import CGPClient
from property_feature_flag_semantic_cache import FeatureFlagCache, FeatureFlag
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

#staging
PROPERTY_DB = 'lp_data_model_stage'
S3_OUTPUT = 's3://lp-datalakehouse-stage/hackweek/'

#production
# PROPERTY_DB = 'lp_data_model_production'
# S3_OUTPUT = 's3://qa-extract-s3-bucket-production/hackweek/'

#set environment variables
import os
#os.environ['AWS_PROFILE'] = 'newetlstaging'
#os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

class DataScan:

    def __init__(self, feed_name, s3_service=S3Service()):
        self.sql_process_batch_limit = 10000
        self.sql_process_current_offset = 0
        self.s3_service = s3_service
        self.feed_name = feed_name
        self.select_params = []
        self.table_types = self.s3_service.get_table_columns(
                PROPERTY_DB, 'property')
        for k, v in self.table_types.items():
            # Temp. code to skip unexpected field
            if (k in ['lp_price_sqft',
                      'event_type',
                      'media',
                      'property_subtype',
              'lp_photos',
              'community_features',
              'association_amenities',
              'close_date',
              'frontage_type',
              'green_energy_generation',
              'land_lease_expiration_date',
              'levels',
              'off_market_date',
              'property_condition',
              'roof',
              'sewer',
              'utilities',
              'water_source',
                'waterfront_features',
                'withdrawn_date',
                'listing_terms',
                'green_building_verification_type',
                'lock_box_type',
                'showing_requirements',
                'current_financing',
                'disclosures',
                      'lot_size_square_acres',
                      'lp_property_local_timezone_offset',
                      'lp_property_local_timezone_abbreviation']):
                continue
            if v == 'timestamp':
                # Special handling for timestamp fields
                self.select_params.append(f"CAST(a.{k} as timestamp) AS {k}")
            else:
                # All other fields go as is
                self.select_params.append(f"a.{k}")

        self.select_params_string = ", ".join(self.select_params)

    def prepare_sql(self):
        additional_filter_for_testing = 'and (cardinality(a.security_features ) > 0 or cardinality(a.building_features) > 0)'
        sql = \
            f" WITH b as (" \
            f" 	SELECT p.lp_provider_id," \
            f" 		p.lp_listing_id," \
            f" 		CAST(p.event_modification_timestamp as timestamp) event_modification_timestamp" \
            f" 	FROM property p" \
            f" 		INNER JOIN (" \
            f" 			SELECT lp_provider_id," \
            f" 				lp_listing_id," \
            f" 				MAX(event_modification_timestamp) as max_event_ts" \
            f" 			FROM property" \
            f" 			GROUP BY lp_provider_id," \
            f" 				lp_listing_id" \
            f" 		) pp ON pp.lp_provider_id = p.lp_provider_id" \
            f" 		AND pp.lp_listing_id = p.lp_listing_id" \
            f" 		AND pp.max_event_ts = p.event_modification_timestamp" \
            f" 	WHERE pp.lp_provider_id = '{self.feed_name}'" \
            f" 		AND pp.max_event_ts > CAST('2024-04-10 00:00:00' as timestamp)" \
            f" 		AND p.lp_listing_status = 'ACTIVE'" \
            f" )" \
            f" SELECT {self.select_params_string}" \
            f" FROM property a" \
            f" 	JOIN b ON a.lp_provider_id = b.lp_provider_id" \
            f" 	and a.lp_listing_id = b.lp_listing_id" \
            f" 	and a.event_modification_timestamp = b.event_modification_timestamp" \
            f"  {additional_filter_for_testing}" \
            f" ORDER BY a.lp_listing_id" \
            f" OFFSET {self.sql_process_current_offset}" \
            f" LIMIT {self.sql_process_batch_limit} "
        return sql
    
    def prepare_sql_custom_listings(self, custom_listings):
        custom_listings_str = ', '.join([f"'{listing}'" for listing in custom_listings])
        sql = \
            f" SELECT a.lp_provider_id, " \
            f"           a.listing_id," \
            f"           a.lp_listing_id," \
            f"           a.lp_listing_status," \
            f"           CAST(a.event_modification_timestamp as timestamp) event_modification_timestamp," \
            f"           CAST(a.lp_processed_timestamp as timestamp) lp_processed_timestamp," \
            f"           a.interior_features," \
            f"           a.exterior_features," \
            f"           a.lot_features," \
            f"           a.community_features,"\
            f"           a.pool_features," \
            f"           a.security_features," \
            f"           a.building_features" \
            f" FROM property a" \
            f" 	WHERE a.lp_provider_id = '{self.feed_name}'" \
            f" 		AND a.listing_id in ({custom_listings_str})"
            
        return sql

    def load_property_data(self):
        sql = self.prepare_sql()
        # sql = self.prepare_sql_custom_listings(['RPLU-33423020830', 'RPLU-33423096980'])
        df = self.s3_service.read_athena(sql, PROPERTY_DB, S3_OUTPUT)
        df = df.sort_values(['event_modification_timestamp', 'lp_processed_timestamp']).drop_duplicates(
            subset=['lp_provider_id', 'listing_id'],
            keep='last')
        self.sql_process_current_offset += self.sql_process_batch_limit
        return df

class DataLoadService:
    def __init__(self, s3_bucket, table_name, target_athena_database, s3_service=S3Service()):
        self.s3_bucket = s3_bucket
        self.table_name = table_name
        self.target_athena_database = target_athena_database
        self.s3_service = s3_service
    
    def insert_into_iceberg_table(self, df: pd.DataFrame, table_name: str, partition_cols: list = None) -> None:
        try:
            if df.empty:
                print(f"No records to insert into {table_name} table")
                return
            print(f"Inserting {df.shape[0]} records into {table_name} table")
            #check if table exists
            # if not self.s3_service.check_db_table_exists(
            #         database=self.target_athena_database,
            #         table_name=table_name
            # ):
            #     raise Exception(f"Table {table_name} does not exist")
            temp_path = f's3://{self.s3_bucket}/warehouse/{self.table_name}/temp/{str(uuid.uuid4())}'

            #set mode to overwrite
            self.s3_service.wr_client.athena.to_iceberg(
                df=df,
                partition_cols=partition_cols,
                table_location=f's3://{self.s3_bucket}/tagged_properties',
                database=self.target_athena_database,
                table=table_name,
                temp_path=temp_path,
                workgroup='primary',
                mode='overwrite'
            )
            print(f"Successfully inserted {df.shape[0]} records into {table_name} table")
        except Exception as ex:
            print(f"Error while inserting into {table_name} table: {ex}")
            raise ex


def main():
    feed='trestle-rebny'
    data_scan = DataScan(feed)
    qdrant_client = None
        # QdrantClient(
        # url="https://3d4cf461-fb47-40ed-81be-2630ab5ac214.us-east4-0.gcp.cloud.qdrant.io:6333",
        # api_key='add your api key here',
        # )
    
    data_load_service = DataLoadService(
        s3_bucket='lp-datalakehouse-stage',
        table_name='property_feature_flags',
        target_athena_database='lp_data_model_stage'
    )

    ff_cache = FeatureFlagCache(llm_client=CGPClient(), qdrant_client=qdrant_client)
    raw_df = data_scan.load_property_data()
    raw_list = raw_df.to_dict('records')
    run_once = True
    while len(raw_df) > 0:
        listings_processed = {}
        raw_df['lp_custom_filter_tags'] = None

        #create dataframe to store the data so that we can upload it to property_feature_flags iceberg table later
        #property_df = pd.DataFrame(columns=['lp_provider_id', 'lp_listing_id', 'lp_custom_filter_tags'])

        for index, row in raw_df.iterrows():
            listings_processed[row['lp_listing_id']] = {}
            lp_custom_filter_tags = []
            for feature in FeatureFlag:
                feat_value_tuple = ff_cache.get_feature_value(row, feature)
                print(f"Feature {feature} is {feat_value_tuple[0]} for listing {row['lp_listing_id']}."
                      f" Cache hit is {feat_value_tuple[1]}, accuracy {feat_value_tuple[2]}")
                listings_processed[row['lp_listing_id']][feature] = feat_value_tuple[0]
                if feat_value_tuple[0]:
                    lp_custom_filter_tags.append(feature.value)

            #add the listing to the dataframe using concat
            if len(lp_custom_filter_tags) > 0:
                raw_df.loc[index, 'lp_custom_filter_tags'] = lp_custom_filter_tags
                # property_df = pd.concat([property_df, pd.DataFrame({
                #     'lp_provider_id': [row['lp_provider_id']],
                #     'lp_listing_id': [row['lp_listing_id']],
                #     'lp_custom_filter_tags': [lp_custom_filter_tags]
                # })])
                
            print(f"Listing processed {listings_processed[row['lp_listing_id']]}")

        #upload the data to iceberg table
        data_load_service.insert_into_iceberg_table(raw_df, 'property_with_feat_flags', ['lp_provider_id'])

        print(listings_processed)

        if run_once:
            break
        # next data row
        raw_df = data_scan.load_property_data()

    #print cache stats
    print(ff_cache.get_stats())

def sample():
    #df = load_property_data()

    qd_client = QdrantClient(url="http://localhost:6333")

    collection_name = "listings"
    vector_size = 384  # Dimension of your vectors
    distance = Distance.COSINE  # Similarity measure (COSINE, EUCLID, etc.)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    #qd_client.delete_collection(collection_name=collection_name)
    if collection_name not in list(map(lambda l: l.name, qd_client.get_collections().collections)):

        qd_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )

    points = []
    features = {
        'building_features': 'Building features',
        'lot_features': 'Lot features',
        'community_features': 'Community features',
        'exterior_features': 'Exterior features',
        'interior_features': 'Interior features',
        'pool_features': 'Pool features'
    }
    for index, row in df.iterrows():
        text_data = []
        for key, feature in features.items():
            value = row[key] if row[key] is not None else None
            str_value = None
            is_list = isinstance(value, ndarray)
            if (is_list or pd.notnull(value)):
                if is_list :
                    value = value.tolist()
                str_value = ", ".join(value)
            final_value = f"{feature}: {str_value}"
            text_data.append(final_value)
        text = ";".join(text_data)
        embeddings = model.encode(text, batch_size=32, show_progress_bar=True)
        points.append(PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, row['lp_listing_id'])), vector=embeddings.tolist(), payload={"text": text}))

    print('==== start inserting points ======')
    path_pages = [points[i:i + 500]
                  for i in range(0, len(points), 500)]
    for page in path_pages:
        qd_client.upsert(collection_name=collection_name, points=page)

    query_text = "Listings with in-ground pool"
    query_embedding = model.encode(query_text).tolist()
    search_results = qd_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5
    )
    print(search_results)

main()