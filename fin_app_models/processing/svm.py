from typing import Tuple, Union

from overrides import overrides
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base_processing import IStructuredDataProcessing
from ..entity.timeseries_data import TimeseriesData


class KernelSVRDataProcessing(IStructuredDataProcessing):

    def __init__(self):
        self._ss = StandardScaler()

    @overrides
    def preprocess(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        ts_y_train = TimeseriesData(y_train)().to_numpy().reshape(-1, 1) if y_train is not None else None
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
        return y_train
