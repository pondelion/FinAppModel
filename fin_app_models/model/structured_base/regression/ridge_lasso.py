from datetime import datetime, timedelta
from typing import Union, Tuple, List, Dict

import pandas as pd
import numpy as np
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet
)
from overrides import overrides

from ..base_model import BaseRegressionModel
from ....processing import (
    IStructuredDataProcessing,
    RidgeRegressionDataProcessing,
    LassoRegressionDataProcessing,
    ElasticNetRegressionDataProcessing,
)
from ....param_tuning import (
    IParamTuber,
    RidgeRegressionTuner,
    LassoRegressionTuner,
    ElasticNetRegressionTuner,
)


class RidgeRegression(BaseRegressionModel):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [RidgeRegressionDataProcessing()],
        param_tuner: IParamTuber = RidgeRegressionTuner(),
    ):
        super(RidgeRegression, self).__init__(data_processors, param_tuner)

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
        self._model = Ridge(**model_params)
        self._model.fit(X_train, y_train)

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        **kwargs,
    ) -> np.ndarray:
        return self._model.predict(X).flatten()


class LassoRegression(BaseRegressionModel):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [LassoRegressionDataProcessing()],
        param_tuner: IParamTuber = LassoRegressionTuner(),
    ):
        super(LassoRegression, self).__init__(data_processors, param_tuner)

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
        self._model = Lasso(**model_params)
        self._model.fit(X_train, y_train)

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        **kwargs,
    ) -> np.ndarray:
        return self._model.predict(X).flatten()


class ElasticNetRegression(BaseRegressionModel):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [ElasticNetRegressionDataProcessing()],
        param_tuner: IParamTuber = ElasticNetRegressionTuner(),
    ):
        super(ElasticNetRegression, self).__init__(data_processors, param_tuner)

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
        self._model = ElasticNet(**model_params)
        self._model.fit(X_train, y_train)

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        **kwargs,
    ) -> np.ndarray:
        return self._model.predict(X).flatten()
