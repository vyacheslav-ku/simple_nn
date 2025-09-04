import os
minio_endpoint=os.environ.get('minio_endpoint')
temp_access_key=os.environ.get('minio_access_key')
temp_secret_key=os.environ.get('minio_secret_key')
bucket_name=os.environ.get('minio_bucket_name')
epoch=os.environ.get('epoch', 2)
batch_size=os.environ.get('minio_batch_size', 64)