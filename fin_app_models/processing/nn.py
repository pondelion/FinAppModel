from typing import Tuple, Union, List

from overrides import overrides
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from .base_processing import IStructuredDataProcessing
from ..entity.timeseries_data import TimeseriesData


class SKMLPDataProcessing(IStructuredDataProcessing):

    def __init__(self, target_X_cols: List[str] = None):
        super().__init__(target_X_cols)
        self._ss = StandardScaler()
        self._mms = MinMaxScaler((0, 1))

    @overrides
    def preprocess(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        ts_y_train = None
        if y_train is not None:
            ts_y_train = TimeseriesData(y_train)().to_numpy().reshape(-1, 1)
            self._mms.fit(ts_y_train)
            ts_y_train = self._mms.transform(ts_y_train)
        ts_X_train = None
        if X_train is not None:
            ts_X_train = TimeseriesData(X_train)().to_numpy()
            self._ss.fit(ts_X_train)
            ts_X_train = self._ss.transform(ts_X_train)
        return (ts_X_train, ts_y_train)

    @overrides
    def postprocess(
        self,
        y_train: np.ndarray
    ) -> pd.Series:
        y_train = y_train.reshape(-1, 1)
        y_train = self._mms.inverse_transform(y_train).reshape(-1)
        return y_train
