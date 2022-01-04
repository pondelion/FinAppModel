from datetime import datetime, timedelta
from typing import Union, Tuple, List, Dict

import pandas as pd
import numpy as np
from sklearn import linear_model
from overrides import overrides

from ..base_model import BaseRegressionModel
from ....processing import (
    IStructuredDataProcessing,
    LinearRegressionDataProcessing,
)
from ....param_tuning import (
    IParamTuber,
    DefaultTuner
)


class LinearRegression(BaseRegressionModel):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [LinearRegressionDataProcessing()],
        param_tuner: IParamTuber = DefaultTuner(),
    ):
        super(LinearRegression, self).__init__(data_processors, param_tuner)

    @overrides
    def _train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs,
    ) -> None:
        self._model = linear_model.LinearRegression(**model_params)
        self._model.fit(X_train, y_train)

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        **kwargs,
    ) -> np.ndarray:
        return self._model.predict(X).flatten()

    def __str__(self):
        if self._model is None:
            return 'Not trained yet.'
        eq = 'y = '
        for i, (col_name, coeff) in enumerate(zip(self._X_col_names, self._model.coef_[0])):
            if (coeff < 0) and (i > 0):
                eq = eq[:-2]
            eq += f'{coeff:.3f}{col_name} + '
        if self._model.intercept_[0] < 0:
            eq = eq[:-2]
        eq += f'{self._model.intercept_[0]:.3f}'
        return eq
