import pandas as pd

from .stock import Stock
from .stocklist import StockList


def check_aws_access() -> bool:
    try:
        pd.read_csv('s3://fin-app/stocklist/stocklist.csv')
        return True
    except Exception as e:
        print(e)
        return False


if not check_aws_access():
    raise Exception("You don't have access right to aws dataset")
