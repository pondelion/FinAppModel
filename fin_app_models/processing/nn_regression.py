from typing import Tuple, Union

from overrides import overrides
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from .base_processing import IStructuredDataProcessing
from ..entity.timeseries_data import TimeseriesData


class SKMLPDataProcessing(IStructuredDataProcessing):

    def __init__(self):
        self._ss = StandardScaler()
        self._mms = MinMaxScaler((0, 1))

    @overrides
    def preprocess(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        if y_train is not None:
            ts_y_train = TimeseriesData(y_train)().to_numpy().reshape(-1, 1)
            self._mms.fit(ts_y_train)
            ts_y_train = self._mms.transform(ts_y_train)
        if X_train is not None:
            ts_X_train = TimeseriesData(X_train)().to_numpy()
            self._ss.fit(ts_X_train)
            ts_X_train = self._ss.transform(ts_X_train)
        return (ts_X_train, ts_y_train)

    @overrides
    def postprocess(
        self,
        y_train: pd.Series
    ) -> pd.Series:
        ts_y_train = TimeseriesData(y_train)().to_numpy().reshape(-1, 1)
        ts_y_train = self._mms.inverse_transform(ts_y_train).reshape(-1)
        return y_train
