import boto3
import os
from botocore.exceptions import ClientError
from aac_ts_anomaly.services import file

tm = file.TOMLservice(root_path = "/home/alexv84/Documents/AWS_stuff", path = "aws_config.toml")
aws_cred = tm.doRead()

bucket_name = aws_cred['Credentials']['bucket_name']

s3 = boto3.resource(
    service_name='s3',
    region_name = aws_cred['Credentials']['region_name'],
    aws_access_key_id = aws_cred['Credentials']['aws_access_key_id'],
    aws_secret_access_key = aws_cred['Credentials']['aws_secret_access_key']
)


# Print out bucket names
for bucket in s3.buckets.all():
    print(bucket.name)

bucket.name


#s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)
bucket

# Iterates through all the objects, doing the pagination for you. Each obj
# is an ObjectSummary, so it doesn't contain the body. You'll need to call
# get to get the whole body.
for obj in bucket.objects.all():
    print(obj)
    key = obj.key
    print(key)
    #body = obj.get()['Body'].read()

[obj.key for obj in bucket.objects.all()]

filename = "agg_time_series_52.csv"


from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.services import file_aws
from importlib import reload
import pandas as pd

reload(file_aws)

from aac_ts_anomaly.config.aws_config import bucket_obj as bucket 

[obj.key for obj in bucket.objects.all()]

path = "agg_time_series_52.csv"
root_path = "anomaly_detection"

obj = bucket.Object(os.path.join(root_path, path)).get()

#foo = pd.read_csv(obj['Body'], index_col=0)

# Import data:
#---------------
csv = file_aws.CSVService(root_path=root_path, path="agg_time_series_52.csv", delimiter=',')

data_orig = csv.doRead() ; data_orig.shape
data_orig.head()


from io import StringIO # python3; python2: BytesIO 
import boto3

foo = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})

csv_buffer = StringIO()
foo.to_csv(csv_buffer)

s3.Object(bucket_name, 'df.csv').put(Body=csv_buffer.getvalue())
