from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import pandas as pd


class IStructuredDataProcessing(metaclass=ABCMeta):

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
        y_train: pd.Series
    ) -> pd.Series:
        raise NotImplementedError
