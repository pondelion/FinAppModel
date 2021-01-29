from abc import ABCMeta, abstractmethod
from enum import Enum
from datetime import datetime
from typing import Union, Dict

import pandas as pd

from ...processing import (
    IStructuredDataProcessing,
    DefaultStructuredDataProcessing
)
from ...param_tuning import (
    IParamTuber,
    DefaultTuner
)
from ...entity.timeseries_data import TimeseriesData


class BaseRegressionModel(metaclass=ABCMeta):

    def __init__(
        self,
        data_processors: IStructuredDataProcessing = [DefaultStructuredDataProcessing()],
        param_tuner: IParamTuber = DefaultTuner(),
    ):
        self._data_processors = data_processors
        self._param_tuner = param_tuner

    def train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        auto_param_tuning: bool = True,
        **kwargs: Dict,
    ) -> None:
        X_train_preprocessed = X_train.copy() if X_train is not None else None
        y_train_preprocessed = y_train.copy()
        for dp in self._data_processors:
            X_train_preprocessed, y_train_preprocessed = dp.preprocess_cols(
                X_train_preprocessed, y_train_preprocessed
            )

        if auto_param_tuning:
            self._best_params = self._param_tuner.param_tuning(
                X_train_preprocessed, y_train_preprocessed, dt_now, 
            )
        else:
            self._best_params = model_params

        self._train(
            y_train=y_train_preprocessed,
            X_train=X_train_preprocessed,
            dt_now=dt_now,
            model_params=self._best_params
            **kwargs
        )

    def predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        pred_days: int = 30,
        **kwargs: Dict,
    ) -> pd.Series:
        X_preprocessed = X.copy()
        y_preprocessed = y.copy()
        for dp in self._data_processors:
            X_preprocessed, y_preprocessed = dp.preprocess_cols(
                X_preprocessed, y_preprocessed
            )
        sr_pred = self._predict(
            y=y_preprocessed,
            X=X_preprocessed,
            pred_days=pred_days
            **kwargs
        )
        for dp in self._data_processors[::-1]:
            sr_pred = dp.postprocess(sr_pred)
        return sr_pred

    @abstractmethod
    def _train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs: Dict,
    ) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        pred_days: int = 30,
        **kwargs: Dict,
    ) -> pd.Series:
        raise NotImplementedError


class BinClassificationResult(Enum):
    UP = 0
    DOWN = 1


class BaseBinClassificationModel(metaclass=ABCMeta):

    @abstractmethod
    def train(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, pd.Series],
        dt: datetime
    ) -> BinClassificationResult:
        raise NotImplementedError
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, pd.Series],
        dt: datetime
    ) -> float:
        raise NotImplementedError('not implemented')


class ThreeClassificationResult(Enum):
    UP = 0
    NO_CHANGE = 1
    DOWN = 2


class BaseThreeClassificationModel(metaclass=ABCMeta):

    @abstractmethod
    def train(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, pd.Series],
        dt: datetime
    ) -> ThreeClassificationResult:
        raise NotImplementedError

    def predict_proba(
        self,
        X: Union[pd.DataFrame, pd.Series],
        dt: datetime
    ) -> float:
        raise NotImplementedError('not implemented')
