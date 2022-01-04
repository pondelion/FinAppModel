from abc import ABCMeta, abstractmethod
from enum import Enum
from datetime import datetime
from typing import Union, Dict, List

import pandas as pd
import numpy as np

from ...processing import (
    IStructuredDataProcessing,
    DefaultStructuredDataProcessing
)
from ...param_tuning import (
    IParamTuber,
    DefaultTuner
)
from ...entity.timeseries_data import TimeseriesData
from ...utils import Logger


class BaseRegressionModel(metaclass=ABCMeta):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [DefaultStructuredDataProcessing()],
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
        daily_interpolation: bool = True,
        **kwargs,
    ) -> None:
        X_train_preprocessed = X_train.copy() if X_train is not None else None
        y_train_preprocessed = y_train.copy()

        self._X_col_names = X_train_preprocessed.columns

        if dt_now is not None:
            X_train_preprocessed = X_train_preprocessed[
                X_train_preprocessed.index <= dt_now
            ] if X_train is not None else None
            y_train_preprocessed = y_train_preprocessed[
                y_train_preprocessed.index <= dt_now
            ]

        for dp in self._data_processors:
            X_train_preprocessed, y_train_preprocessed = dp.preprocess_cols(
                X_train_preprocessed, y_train_preprocessed
            )
        if auto_param_tuning:
            Logger.d(self.__class__.__name__, 'Start tuning parameters')
            self._best_params = self._param_tuner.param_tuning(
                y_train=y_train_preprocessed,
                X_train=X_train_preprocessed,
                dt_now=dt_now,
            )
            Logger.d(self.__class__.__name__, 'Done tuning parameters')
            Logger.d(self.__class__.__name__, f'best_pramas : {self._best_params}')
        else:
            self._best_params = model_params

        self._train(
            y_train=y_train_preprocessed,
            X_train=X_train_preprocessed,
            dt_now=dt_now,
            model_params=self._best_params,
            daily_interpolation=daily_interpolation,
            **kwargs
        )

    def predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        X_preprocessed = X.copy() if X is not None else None
        y_preprocessed = y.copy() if y is not None else None
        self._X_index = X_preprocessed.index
        for dp in self._data_processors:
            X_preprocessed, y_preprocessed = dp.preprocess_cols(
                X_preprocessed, y_preprocessed
            )
        sr_pred = self._predict(
            y=y_preprocessed,
            X=X_preprocessed,
            **kwargs
        )
        for dp in self._data_processors[::-1]:
            sr_pred = dp.postprocess(sr_pred)
        return sr_pred

    @abstractmethod
    def _train(
        self,
        y_train: Union[pd.Series, np.ndarray],
        X_train: Union[Union[pd.DataFrame, pd.Series], np.ndarray] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(
        self,
        y: Union[pd.Series, np.ndarray],
        X: Union[Union[pd.DataFrame, pd.Series], np.ndarray] = None,
        **kwargs,
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


class BaseTimeseriesModel(metaclass=ABCMeta):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [DefaultStructuredDataProcessing()],
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
        **kwargs,
    ) -> None:
        X_train_preprocessed = X_train.copy() if X_train is not None else None
        y_train_preprocessed = y_train.copy()

        if dt_now is not None:
            X_train_preprocessed = X_train_preprocessed[
                X_train_preprocessed.index <= dt_now
            ] if X_train is not None else None
            y_train_preprocessed = y_train_preprocessed[
                y_train_preprocessed.index <= dt_now
            ]

        for dp in self._data_processors:
            X_train_preprocessed, y_train_preprocessed = dp.preprocess_cols(
                X_train_preprocessed, y_train_preprocessed
            )

        if auto_param_tuning:
            Logger.d(self.__class__.__name__, 'Start tuning parameters')
            self._best_params = self._param_tuner.param_tuning(
                X_train_preprocessed, y_train_preprocessed, dt_now, 
            )
            Logger.d(self.__class__.__name__, 'Done tuning parameters')
            Logger.d(self.__class__.__name__, f'best_pramas : {self._best_params}')
        else:
            self._best_params = model_params

        self._train(
            y_train=y_train_preprocessed,
            X_train=X_train_preprocessed,
            dt_now=dt_now,
            model_params=self._best_params,
            **kwargs
        )

    def predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        pred_periods: int = 30,
        **kwargs,
    ) -> pd.Series:
        X_preprocessed = X.copy() if X is not None else None
        y_preprocessed = y.copy() if y is not None else None
        for dp in self._data_processors:
            X_preprocessed, y_preprocessed = dp.preprocess_cols(
                X_preprocessed, y_preprocessed
            )
        sr_pred = self._predict(
            y=y_preprocessed,
            X=X_preprocessed,
            pred_periods=pred_periods,
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
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        pred_periods: int = 30,
        **kwargs,
    ) -> pd.Series:
        raise NotImplementedError
