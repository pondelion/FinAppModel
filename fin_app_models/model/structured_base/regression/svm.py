from datetime import datetime, timedelta
from typing import Union, Tuple, List, Dict

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from overrides import overrides

from ..base_model import BaseRegressionModel
from ....processing import (
    IStructuredDataProcessing,
    KernelSVRDataProcessing,
)
from ....param_tuning import (
    IParamTuber,
    KernelSVRRegressionTuner
)


class KernelSVRRegression(BaseRegressionModel):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [KernelSVRDataProcessing()],
        param_tuner: IParamTuber = KernelSVRRegressionTuner(),
    ):
        super(KernelSVRRegression, self).__init__(data_processors, param_tuner)

    @overrides
    def _train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs,
    ) -> None:
        self._model = SVR(**model_params)
        self._model.fit(X_train, y_train)

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        **kwargs,
    ) -> np.ndarray:
        return self._model.predict(X).flatten()
