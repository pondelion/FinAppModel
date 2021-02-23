from fin_app_models.dataset import (
    Stock,
    StockList,
    EconomicIndicatorJA,
    IndicatorTypeJA,
)
from fin_app_models.entity.timeseries_data import TimeseriesData


# print(Stock(code=4120).data)
# print(StockList().data)

print(EconomicIndicatorJA(indicator_type=IndicatorTypeJA.CPI_JA).data)

print(TimeseriesData(EconomicIndicatorJA(indicator_type=IndicatorTypeJA.CPI_JA).data)())