import uuid

import pandas as pd
from numpy import ndarray

from s3_service import S3Service
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer



def load_property_data(feed='cws-bright'):
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
        f" 	WHERE pp.lp_provider_id = '{feed}'" \
        f" 		AND pp.max_event_ts > CAST('2024-04-10 00:00:00' as timestamp)" \
        f" 		AND p.lp_listing_status = 'ACTIVE'" \
        f"      AND list_price is not null AND list_price >= 350000" \
        f"      AND property_type = 'Residential'" \
        f" )" \
        f" SELECT a.lp_provider_id, " \
        f"           a.listing_id," \
        f"           a.lp_listing_id," \
        f"           a.lp_listing_status," \
        f"           CAST(a.event_modification_timestamp as timestamp) event_modification_timestamp," \
        f"           a.interior_features," \
        f"           a.exterior_features," \
        f"           a.lot_features," \
        f"           a.community_features,"\
        f"           a.pool_features," \
        f"           a.building_features" \
        f" FROM property a" \
        f" 	JOIN b ON a.lp_provider_id = b.lp_provider_id" \
        f" 	and a.lp_listing_id = b.lp_listing_id" \
        f" 	and a.event_modification_timestamp = b.event_modification_timestamp" \
        f" LIMIT 15000"

    s3_service = S3Service()
    df = s3_service.read_athena(sql, 'lp_data_model_production', 's3://qa-extract-s3-bucket-production/hackweek/')
    return df

def main():
    df = load_property_data()

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