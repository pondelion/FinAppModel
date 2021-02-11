from typing import Tuple, Union

from overrides import overrides
import pandas as pd

from .base_processing import IStructuredDataProcessing
from ..entity.timeseries_data import TimeseriesData


class RidgeRegressionDataProcessing(IStructuredDataProcessing):

    @overrides
    def preprocess(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        ts_y_train = TimeseriesData(y_train)().to_numpy().reshape(-1, 1) if y_train is not None else None
        ts_X_train = TimeseriesData(X_train)() if X_train is not None else None
        return (ts_X_train, ts_y_train)

    @overrides
    def postprocess(
        self,
        y_train: pd.Series
    ) -> pd.Series:
        return y_train


class LassoRegressionDataProcessing(IStructuredDataProcessing):

    @overrides
    def preprocess(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        ts_y_train = TimeseriesData(y_train)().to_numpy().reshape(-1, 1) if y_train is not None else None
        ts_X_train = TimeseriesData(X_train)() if X_train is not None else None
        return (ts_X_train, ts_y_train)

    @overrides
    def postprocess(
        self,
        y_train: pd.Series
    ) -> pd.Series:
        return y_train


class ElasticNetRegressionDataProcessing(IStructuredDataProcessing):

    @overrides
    def preprocess(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        ts_y_train = TimeseriesData(y_train)().to_numpy().reshape(-1, 1) if y_train is not None else None
        ts_X_train = TimeseriesData(X_train)() if X_train is not None else None
        return (ts_X_train, ts_y_train)

    @overrides
    def postprocess(
        self,
        y_train: pd.Series
    ) -> pd.Series:
        return y_train
