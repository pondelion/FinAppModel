from typing import Tuple, Union

from overrides import overrides
import pandas as pd

from .base_processing import IStructuredDataProcessing


class DefaultStructuredDataProcessing(IStructuredDataProcessing):

    @overrides
    def preprocess(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series
    ) -> Tuple[Union[pd.DataFrame, pd.Series], pd.Series]:
        return (X_train, y_train)

    @overrides
    def postprocess(
        self,
        y_train: pd.Series
    ) -> pd.Series:
        return y_train
