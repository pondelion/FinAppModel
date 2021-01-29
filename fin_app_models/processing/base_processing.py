from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, List

import pandas as pd


class IStructuredDataProcessing(metaclass=ABCMeta):

    def __init__(self, target_X_cols: List[str] = None):
        self._target_X_cols = target_X_cols

    def preprocess_cols(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        if (self._target_X_cols is None) or (X_train is None):
            return self.preprocess(X_train, y_train)
        else:
            X_train_preprocessed, y_train_preprocessed = self.preprocess(
                X_train=X_train[self._target_X_cols],
                y_train=y_train
            )
            X_train_cp = X_train.copy()
            X_train_cp.loc[:, self._target_X_cols] = X_train_preprocessed
            return (X_train_cp, y_train_preprocessed)

    @abstractmethod
    def preprocess(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        y: pd.Series
    ) -> pd.Series:
        raise NotImplementedError
