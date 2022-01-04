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

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        **kwargs,
    ) -> np.ndarray:
        return self._model.predict(X).flatten()

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=self._model.feature_importance(),
            index=self._X_col_names,
            columns=['importance']
        ).sort_values(by='importance', ascending=False)
