import os
from datetime import datetime, date
from enum import Enum

from overrides import overrides
import pandas as pd

from .s3_csv_cached_data import S3CSVCachedData
from ..utils.config import DataLocationConfig


class IndicatorTypeJA(Enum):
    CPI_JA = 'cpi'
    GDP_JA = 'gdp'
    PPI_JA = 'ppi'
    INDUSTRIAL_PRODUCTION_JA = 'industrial_production'
    RETAIL_SALES_JA = 'retail_sales'
    INTERNATIONAL_TRADE_EXPORTS_JA = 'international_trade_exports'
    INTERNATIONAL_TRADE_IMPORTS_JA = 'international_trade_imports'
    INTEREST_RATE_JA = 'interest_rate'
    NIKKEI225_JA = 'nikkei225_stock_average'
    GOVERNMENT_DEBT_JA = 'central_government_debt'
    REAL_NET_EXPORTS_GOOD_SERVICES_JA = 'real_net_exports_of_good_and_services'
    RESIDENTAL_PROPERTY_PRICE_JA = 'residental_property_price'
    TOTAL_INDUSTRY_PRODUCTION = 'total_industry_production'
    UNEMPLOYMENT_RATE_JA = 'unemployment_rate'
    WORKING_AGE_POPULATION_JA = 'working_age_population'
    REAL_EFFECTIVE_EXCHANGE_RATE_JA = 'real_effective_exchange_rate'


class EconomicIndicatorJA(S3CSVCachedData):

    def __init__(
        self,
        indicator_type: IndicatorTypeJA
    ):
        self._indicator_type = indicator_type
        self._DATETIME_COL_NAME = 'DATE'

    @overrides
    def _local_cache_path(self) -> str:
        local_cache_path = os.path.join(
            DataLocationConfig.LOCAL_CACHE_DIR,
            'ja',
            f'{self._indicator_type.value}.csv'
        )
        return local_cache_path

    @overrides
    def _source_path(self) -> str:
        source_path = os.path.join(
            DataLocationConfig.ECONOMIC_INDICATOR_BASEDIR,
            'ja',
            f'{self._indicator_type.value}.csv'
        )
        return source_path

    @property
    @overrides
    def data(self) -> pd.DataFrame:
        df = super().data
        if self._DATETIME_COL_NAME in df.columns:
            df[self._DATETIME_COL_NAME] = pd.to_datetime(df[self._DATETIME_COL_NAME])
            df.set_index(self._DATETIME_COL_NAME, inplace=True)
            df.sort_index(inplace=True)
        return df
