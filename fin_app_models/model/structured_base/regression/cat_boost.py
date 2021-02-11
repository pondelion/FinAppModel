from datetime import datetime, timedelta
from typing import Union, Tuple, List, Dict

import pandas as pd
import numpy as np
import catboost
from overrides import overrides

from ..base_model import BaseRegressionModel
from ....processing import (
    IStructuredDataProcessing,
    CatBoostDataProcessing,
)
from ....param_tuning import (
    IParamTuber,
    CatBoostRegressionTuner,
)


class CatBoostRegression(BaseRegressionModel):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [CatBoostDataProcessing()],
        param_tuner: IParamTuber = CatBoostRegressionTuner(),
    ):
        super(CatBoostRegression, self).__init__(data_processors, param_tuner)

    @overrides
    def _train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs,
    ) -> None:
        self._model = catboost.CatBoostRegressor(**model_params)
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 50)

        self._model.fit(
            X_train, y_train,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        self._X_col_names = X_train.columns

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        sr_pred = pd.Series(
            index=X.index,
            data=self._model.predict(X).flatten()
        )
        return sr_pred
