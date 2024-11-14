import asyncio
import logging
import time

import aioboto3
import awswrangler as wr
import boto3
import botocore
import sqlglot
from botocore.exceptions import BotoCoreError, ClientError

from log import logger


class S3Service:
    """A class for writing Pandas DataFrame to S3 and updating Glue Catalog using
    AWS Data Wrangler Refer: https://aws-sdk-pandas.readthedocs.io/en/stable/.
    """

    def __init__(self, wr_client=wr, debug=False):
        """Initializes a new instance of the S3Service class.

        Args:
            wr_client: (optional) The AWS Data Wrangler client to use.
                Defaults to the `awswrangler` module.
        """
        self.wr_client = wr_client
        self.async_boto_session = aioboto3.Session()
        if debug:
            logging.getLogger("awswrangler").setLevel(logging.DEBUG)
            # We are setting only 1 handler for general logger, reuse it here to avoid re-creation of its stuff like
            # format, converters, etc.
            logging.getLogger("awswrangler").addHandler(logger.handlers[0])
            logging.getLogger("botocore.credentials").setLevel(logging.CRITICAL)

    def write_dataframe_s3_glue(self,
                                df,
                                database,
                                table,
                                path,
                                partition_cols=None,
                                mode="append",
                                schema_evolution=False,
                                max_rows_by_file=100000,
                                compression="snappy",
                                dtype: dict[str, str] | None = None,
                                use_threads=False,
                                **kwargs):
        """Writes the specified DataFrame to the specified S3 location as Parquet files
        and creates an AWS Glue table. The Database needs to exist in the Glue Catalog
        before calling this method - we can add support for creating db later if needed.

        Args:
            df: The DataFrame to write.
            database: (str) The name of the AWS Glue database.
            table: (str) The name of the AWS Glue table.
            path: (str) The S3 path where the Parquet files should be written.
            partition_cols: (List[str], optional) A list of partition columns.
            mode: (str, optional) The write mode.
                    Defaults to "append".
            schema_evolution: (bool, optional) Whether to enable schema evolution.
                    Defaults to False.
            max_rows_by_file: (int, optional) Max no of rows to store in each file
                    Defaults to 100000.
            compression: (str, optional) Compression algorithm to use.
                    Defaults to "snappy".
            dtype : (Dict[str, str], optional),
                Dictionary of columns names and Athena/Glue types to be casted.
                Useful when you have columns with undetermined or mixed data types.
                (e.g. {'col name': 'bigint', 'col2 name': 'int'})
            use_threads : (bool, optional) True to enable concurrent requests,
                False to disable multiple threads.
                If enabled os.cpu_count() will be used as the max number of threads.
                Currnently set to False
            kwargs: (optional) Additional keyword arguments to pass to the
                    AWS Data Wrangler `to_parquet()` method
        Raises:
            Exception: If writing dataframe to S3 / Glue Catalog fails
        """
        logger.info(f"Writing DataFrame to {path}...")
        try:
            self.wr_client.s3.to_parquet(
                df=df,
                path=path,
                partition_cols=partition_cols,
                mode=mode,
                dataset=True,
                database=database,
                table=table,
                schema_evolution=schema_evolution,
                max_rows_by_file=max_rows_by_file,
                compression=compression,
                dtype=dtype,
                use_threads=use_threads,
                **kwargs
            )
            logger.info(f"write_dataframe completed: {path}")
        except Exception as e:
            logger.error(f"Failed to write DataFrame, error: {e}")
            raise e

    def read_athena(self, sql, database, s3_output, ctas_approach=True):
        """Read athena DB by given SQL query
        :param sql: SQL query
        :param database: DB to read from
        :param s3_output: S3 athena output bucket and path
        :return: df: DataFrame of data read
        """
        logger.info(f"Validating sql with sqlglot")
        try:
            sqlglot.parse(sql, read="athena")
        except sqlglot.errors.ParseError as e:
            logger.error(f"Error in given sql query {sql}")
            logger.error(e.errors)
            raise e
        logger.info(f"Reading athena with query {sql} on db {database} to {s3_output}")
        try:
            df = self.wr_client.athena.read_sql_query(sql, database, s3_output=s3_output, ctas_approach=ctas_approach)
            return df
        except Exception as ex:
            logger.error(f"Failed to read Athena with query {sql} on db {database}")
            raise ex

    def check_db_table_exists(self, database, table_name):
        """Check if athena table exists in the database
        :param database: DB name
        :param table_name: Table name
        :return: boolean True if table exists, False otherwise
        """
        logger.info(f"Checking if athena table {table_name} exists in db {database}")
        try:
            return self.wr_client.catalog.does_table_exist(database=database, table=table_name)
        except Exception as ex:
            logger.error(f"Failed to validate if table {table_name} exists in db {database}")
            raise ex

    def get_table_columns(self, database, table):
        logger.info(f"Reading catalog (column types) for table {table} in db {database}")
        try:
            types = self.wr_client.catalog.get_table_types(database, table)
            logger.info("Reading catalog completed")
            return types
        except Exception as ex:
            logger.error(f"Failed to read catalog for able {table} in db {database}")
            raise ex

    def copy_s3_objects(self, s3_source, s3_target, paths=[]):
        """Wrapper to call copy_objects batch API
        :param s3_source: source path to copy from
        :param s3_target: target path to copy to
        :param paths: the list of the paths to be copied
        :return:
        """
        try:
            self.wr_client.s3.copy_objects(paths, source_path=s3_source, target_path=s3_target)
        except Exception as ex:
            logger.error(f"Failed to copy s3 data in copy_s3_objects, error {ex}")
            raise ex

    async def copy_s3_objects_async(self, s3_target, paths=[], content_type=None):
        """Asynchronously copy s3 objects from source to target provided a list of paths
        :param s3_target: target path to copy to
        :param paths: the list of tuples containing (source s3 path, property id)
        :return:
        """
        tasks = []
        for path in paths:
            s3_path = path[0]
            property_id = path[1]
            task = self.copy_object(s3_path, property_id, s3_target, content_type)
            tasks.append(task)

        if tasks:
            logger.info(f"Executing {len(tasks)} asynchronous copy tasks...")
            tasks_start_time = time.time()
            results = await asyncio.gather(*tasks)
            logger.info(
                f"Successfully copied {results.count(True)} objects and pulled and pushed {results.count(False)} objects.")
            tasks_end_time = time.time()
            processing_time = tasks_end_time - tasks_start_time
            logger.info(f"All asynchronous copy tasks completed. Total processing time:"
                        f" {processing_time} seconds.")

    def push_s3_content(self, data, s3_bucket, s3_path, content_type=None):
        """Wrapper used to push content into s3 bucket
        :param data: Byte data to push
        :param s3_bucket: Bucket to push to
        :param s3_path: Path within the bucket where to store new data
        :param content_type: Data content type
        :return:
        """
        s3_params = {
            'Body': data,
            'Bucket': s3_bucket,
            'Key': s3_path
        }
        if content_type:
            s3_params['ContentType'] = content_type
        client = boto3.client("s3")
        try:
            client.put_object(**s3_params)
        except Exception as ex:
            logger.error(f"Unable to push content to s3 bucket {s3_bucket} due to {ex}")
            raise ex

    async def push_s3_content_async(self, data, s3_bucket, s3_path, content_type=None):
        """Asynchronously push content into s3 bucket
        :param data: Byte data to push
        :param s3_bucket: Bucket to push to
        :param s3_path: Path within the bucket where to store new data
        :param content_type: Data content type
        :return:
        """
        s3_params = {
            'Body': data,
            'Bucket': s3_bucket,
            'Key': s3_path
        }
        if content_type:
            s3_params['ContentType'] = content_type
        async with self.async_boto_session.client("s3") as s3_client:
            try:
                await s3_client.put_object(**s3_params)
            except Exception as ex:
                logger.error(f"Unable to push content to s3 bucket {s3_bucket} due to {ex}")
                raise ex

    @staticmethod
    def list_objects_v2(s3_bucket, s3_prefix):
        """Wrapper used to list objects in s3 bucket
        :param s3_bucket: Bucket to list from
        :param s3_prefix: Prefix to filter objects
        :return: list of objects
        """
        client = boto3.client("s3")
        try:
            response = client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
            return response.get('Contents')
        except Exception as ex:
            logger.error(f"Unable to list objects in s3 bucket {s3_bucket}/{s3_prefix} due to {ex}")
            raise ex

    async def check_s3_file_exists(self, s3_bucket, s3_path):
        try:
            async with self.async_boto_session.resource("s3") as S3:
                obj = await S3.Object(s3_bucket, s3_path)
                await obj.load()
                return True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            else:
                logger.error(f"ClientError when checking if file {s3_path} exists in bucket {s3_bucket}: {e}")
                raise e
        except botocore.exceptions.BotoCoreError as e:
            # This catches errors that are not caught by ClientError.
            # It includes things like connection errors.
            logger.error(f"BotoCoreError when checking if file {s3_path} exists in bucket {s3_bucket}: {e}")
            raise e
        except Exception as e:
            # This is a catch-all for any other exception that might occur, which you weren't explicitly checking for.
            logger.error(f"Unexpected error when checking if file {s3_path} exists in bucket {s3_bucket}: {e}")
            raise e

    async def upload_to_s3(self, bucket, data, path, content_type):
        try:
            async with self.async_boto_session.client('s3') as s3:
                await s3.put_object(Bucket=bucket, Key=path,
                                    Body=data, ContentType=content_type)
        except ClientError as e:
            # Handle client-side error (e.g., S3 bucket not found, access denied)
            logger.error(f"Client error occurred during s3 upload to {bucket}/{path}: {e}")
            raise e
        except BotoCoreError as e:
            # Handle errors that are not caught by ClientError
            logger.error(f"BotoCore error occurred during s3 upload to {bucket}/{path}: {e}")
            raise e
        except Exception as e:
            # Handle all other exceptions
            logger.error(f"Unexpected error occurred during s3 upload to {bucket}/{path}: {e}")
            raise e

    def get_table_location(self, database, table):
        """Get the location of the table in the Glue Catalog
        :param database: DB name
        :param table: Table name
        :return: location of the table
        """
        try:
            return self.wr_client.catalog.get_table_location(database, table)
        except Exception as ex:
            logger.error(f"Failed to get table location for table {table} in db {database}")
            raise ex

    def parse_s3_path(self, s3_path):
        """Parse s3 path into bucket and prefix
        :param s3_path: s3 path
        """
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        bucket_name, prefix = s3_path.split("/", 1)
        return bucket_name, prefix

    def is_iceberg_table(self, database, table):
        """Check if the table is an Iceberg table by checking the table type in the Glue Catalog
        :param database: DB name
        :param table: Table name
        :return: boolean True if table is Iceberg, False otherwise
        """
        # Describe the table to get metadata
        table_metadata = self.wr_client.catalog.get_table_parameters(database=database, table=table)

        table_type = table_metadata.get('table_type', None)
        return table_type and table_type.lower() == 'iceberg'

    def get_table_partition_accumulated_size(self, database, table):
        """Get the size of the table partitions per partition, and total number of files, directory.
        :return: size of the table partitions
        """
        location = self.get_table_location(database, table)
        bucket_name, prefix = self.parse_s3_path(location)

        if self.is_iceberg_table(database, table):
            prefix = f"{prefix.rstrip('/')}/data/" if not prefix.endswith('/data/') else prefix
            # prefix = f"{prefix.rstrip('/')}/metadata" if not prefix.endswith('/metadata/') else prefix

        # Initialize a boto3 client
        s3 = boto3.client('s3')

        # Initialize the paginator for the list_objects_v2 method
        paginator = s3.get_paginator('list_objects_v2')

        # Initialize a list to store directory info
        directory_info = []

        # Use the paginator to retrieve objects within the specified path
        # Delimiter is used to treat '/' as a directory separator
        operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix, 'Delimiter': '/'}
        page_iterator = paginator.paginate(**operation_parameters)

        num_directories = 0

        for page in page_iterator:
            # Handling directories
            if 'CommonPrefixes' in page:
                for directory in page['CommonPrefixes']:
                    num_directories += 1
                    directory_prefix = directory['Prefix']

                    # Initialize counters for the directory
                    file_count = 0
                    total_size = 0

                    # List all objects in the current directory
                    for obj_page in paginator.paginate(Bucket=bucket_name, Prefix=directory_prefix):
                        if 'Contents' in obj_page:
                            for obj in obj_page['Contents']:
                                file_count += 1
                                total_size += obj['Size']

                    # Append directory info to the list
                    info = {
                        'prefix': directory_prefix,
                        'file_count': file_count,
                        'total_size_bytes': total_size,
                        'total_size_mb': total_size / (1024 * 1024)
                    }
                    directory_info.append(info)

        # Sort the list by total size in descending order
        sorted_directory_info = sorted(directory_info, key=lambda x: x['total_size_bytes'], reverse=True)

        return {
            'total_directory_count': num_directories,
            'total_file_count': sum(x['file_count'] for x in sorted_directory_info),
            'total_size_bytes': sum(x['total_size_bytes'] for x in sorted_directory_info),
            'total_size_mb': sum(x['total_size_mb'] for x in sorted_directory_info),
            'directory_info': sorted_directory_info,
        }
