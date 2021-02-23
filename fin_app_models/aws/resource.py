import boto3

from ..utils.config import AWSConfig
from ..utils.logger import Logger


try:
    _aws_session = boto3.session.Session(
        region_name=AWSConfig.REGION_NAME,
        aws_access_key_id=AWSConfig.ACCESS_KEY_ID,
        aws_secret_access_key=AWSConfig.SECRET_ACCESS_KEY,
    )
except Exception as e:
    Logger.e(__file__, e)
    _aws_session = boto3.session.Session()

DYNAMO_DB = _aws_session.resource('dynamodb')
S3 = _aws_session.resource('s3')
