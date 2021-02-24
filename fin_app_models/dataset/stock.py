import os
from datetime import datetime, date

from overrides import overrides
import pandas as pd

from .s3_csv_cached_data import S3CSVCachedData
from ..utils.config import DataLocationConfig


class Stock(S3CSVCachedData):

    def __init__(
        self,
        code: int = None,
        company_name: str = None,
    ):
        self._code = code
        self._company_name = company_name
        # if code is None and company_name is not None:
        #     self._code = company_name2code(company_name)
        self._DATETIME_COL_NAME = 'Date'
        self._OPEN_COL_NAME = 'Open'
        self._CLOSE_COL_NAME = 'Close'

    @property
    def code(self) -> int:
        return self._code

    @property
    def company_name(self) -> str:
        return self._company_name

    @property
    @overrides
    def data(self) -> pd.DataFrame:
        df = super().data
        if self._DATETIME_COL_NAME in df.columns:
            df[self._DATETIME_COL_NAME] = pd.to_datetime(df[self._DATETIME_COL_NAME])
            df.set_index(self._DATETIME_COL_NAME, inplace=True)
            df.sort_index(inplace=True)
        return df

    @overrides
    def _local_cache_path(self) -> str:
        local_cache_path = os.path.join(
            DataLocationConfig.LOCAL_CACHE_DIR,
            'stock',
            f'{self._code}.csv'
        )
        return local_cache_path

    @overrides
    def _source_path(self) -> str:
        source_path = os.path.join(
            DataLocationConfig.STOCKPRICE_STOOQ_CONCAT_BASEDIR,
            f'{self._code}.csv'
        )
        return source_path

    def __getitem__(self, dt):
        if not isinstance(dt, datetime):
            raise Exception('Only datetime type index accessing is supported.')

        df = self._data()

        if self._DATETIME_COL_NAME in df.columns:
            df[self._DATETIME_COL_NAME] = pd.to_datetime(df[self._DATETIME_COL_NAME])
            df.set_index(self._DATETIME_COL_NAME, inplace=True)

        try:
            dt_idx = datetime(dt.year, dt.month, dt.day)
            stock_price = df.loc[dt_idx, self._OPEN_COL_NAME]
            stock_price += df.loc[dt_idx, self._CLOSE_COL_NAME]
        except Exception:
            raise StockDataNotFoundException(f'stock data at {dt_idx} not found')
        return stock_price // 2


class StockDataNotFoundException(Exception):
    pass