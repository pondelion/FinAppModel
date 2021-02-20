import os
from datetime import datetime, date

from overrides import overrides
import pandas as pd

from .s3_csv_cached_data import S3CSVCachedData
from ..utils.config import DataLocationConfig


class StockList(S3CSVCachedData):

    @overrides
    def _local_cache_path(self) -> str:
        local_cache_path = os.path.join(
            DataLocationConfig.LOCAL_CACHE_DIR,
            'stocklist',
            'stocklist.csv'
        )
        return local_cache_path

    @overrides
    def _source_path(self) -> str:
        source_path = DataLocationConfig.STOCKLIST_FILE
        return source_path
