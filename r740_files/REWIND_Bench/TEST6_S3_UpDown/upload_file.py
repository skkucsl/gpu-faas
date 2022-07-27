import boto3
import time

def s3_connection():
    s3 = boto3.client(
        service_name="s3",
        region_name="ap-northeast-2", # 자신이 설정한 bucket region
        aws_access_key_id="AKIARPQUZ627QGCVDIWQ",
        aws_secret_access_key="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",
    )
    return s3

def s3_put_object(s3, bucket, filepath, access_key):
    s3.upload_file(
        Filename=filepath,
        Bucket=bucket,
        Key=access_key
    )

def s3_get_object(s3, bucket, filepath, access_key):
    s3.download_file(
        Filename=filepath,
        Bucket=bucket,
        Key=access_key
    )


s3 = s3_connection()
s3_put_object(s3, 'bskim-s3-test', './dog.PNG', 'dog.PNG')
    
