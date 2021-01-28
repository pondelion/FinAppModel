from abc import ABCMeta, abstractmethod
from enum import Enum
from datetime import datetime
from typing import Union, Dict

import pandas as pd

from ...processing import (
    IStructuredDataProcessing,
    DefaultStructuredDataProcessing
)


class BaseRegressionModel(metaclass=ABCMeta):

    def __init__(
        self,
        data_processor: IStructuredDataProcessing = DefaultStructuredDataProcessing()
    ):
        self._data_processor = data_processor

    def train(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series,
        kwargs: Dict = {},
    ) -> None:
        X_train_preprocessed, y_train_preprocessed = self._data_processor.preprocess(
            X_train, y_train
        )
        self._train(
            X_train=X_train_preprocessed,
            y_train=y_train_preprocessed,
            **kwargs
        )

    def predict(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: pd.Series,
        pred_days: int = 30,
        kwargs: Dict = {},
    ) -> pd.Series:
        X_preprocessed, y_preprocessed = self._data_processor.preprocess(X, y)
        sr_pred = self._predict(
            X=X_preprocessed,
            y=y_preprocessed,
            pred_days=pred_days
            **kwargs
        )
        sr_pred_postprocessed = self._data_processor.postprocess(sr_pred)
        return sr_pred_postprocessed

    @abstractmethod
    def _train(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _predict(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: pd.Series,
        pred_days: int = 30,
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
