import os
from enum import Enum

from overrides import overrides

from .s3_csv_cached_data import S3CSVCachedData
from ..utils.config import DataLocationConfig


class CondleChartConfig(Enum):
    DAILY_WINDOW120D_STRIDE30D_WIDTH05 = 'DAILY_WINDOW-120d_STRIDE-30d_WIDTH-0.5'


class CandleChartMetaData(S3CSVCachedData):

    def __init__(self, candle_chart_config: CondleChartConfig):
        self._config = candle_chart_config

    @overrides
    def _local_cache_path(self) -> str:
        local_cache_path = os.path.join(
            DataLocationConfig.LOCAL_CACHE_DIR,
            'metadata',
            f'{self._config.value}.csv'
        )
        return local_cache_path

    @overrides
    def _source_path(self) -> str:
        source_path = os.path.join(
            DataLocationConfig.STOCKPRICE_CANDLECHART_BASEDIR,
            'metadata',
            self._config.value,
            'stockprice_metadata.csv'
        )
        return source_path
