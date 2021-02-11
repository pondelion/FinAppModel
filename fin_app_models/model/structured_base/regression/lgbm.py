from datetime import datetime, timedelta
from typing import Union, Tuple, List, Dict

import pandas as pd
import numpy as np
import lightgbm as lgb
from overrides import overrides

from ..base_model import BaseRegressionModel
from ....processing import (
    IStructuredDataProcessing,
    LGBMDataProcessing,
)
from ....param_tuning import (
    IParamTuber,
    LGBMRegressionTuner,
)


class LGBMRegression(BaseRegressionModel):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [LGBMDataProcessing()],
        param_tuner: IParamTuber = LGBMRegressionTuner(),
    ):
        super(LGBMRegression, self).__init__(data_processors, param_tuner)

    @overrides
    def _train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs,
    ) -> None:
        lgb_train = lgb.Dataset(X_train, y_train)
        num_boost_round = kwargs.get('num_boost_round', 100)
        self._model = lgb.train(model_params, lgb_train, num_boost_round=num_boost_round)

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
