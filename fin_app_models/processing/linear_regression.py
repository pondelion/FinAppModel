from typing import Tuple, Union

from overrides import overrides
import pandas as pd

from .base_processing import IStructuredDataProcessing
from ..entity.timeseries_data import TimeseriesData


class DefaultTrendLinearRegressionProcessing(IStructuredDataProcessing):

    @overrides
    def preprocess(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        ts_y_train = TimeseriesData(y_train)()
        return (X_train, ts_y_train)

    @overrides
    def postprocess(
        self,
        y_train: pd.Series
    ) -> pd.Series:
        return y_train
