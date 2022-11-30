import boto3
from aac_ts_anomaly.services import file

credentials_dir = "/home/alexv84/Documents/AWS_stuff"

aws_cred = file.TOMLservice(root_path = credentials_dir, path = "aws_config.toml").doRead()

bucket_name = aws_cred['Credentials']['bucket_name']

s3 = boto3.resource(
    service_name='s3',
    region_name = aws_cred['Credentials']['region_name'],
    aws_access_key_id = aws_cred['Credentials']['aws_access_key_id'],
    aws_secret_access_key = aws_cred['Credentials']['aws_secret_access_key']
)

bucket_obj = s3.Bucket(bucket_name); print(f"AWS S3 bucket object initialized.")
