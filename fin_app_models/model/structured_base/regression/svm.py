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
        random_state = kwargs.get('random_state', 42)
        model_params['random_state'] = random_state
        self._model = SVR(**model_params)

        self._model.fit(X_train, y_train)
        self._X_col_names = X_train.columns

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        pred_days: int = 30,
        **kwargs,
    ) -> pd.Series:
        sr_pred = pd.Series(
            index=X.index,
            data=self._model.predict(X).flatten()
        )
        return sr_pred
