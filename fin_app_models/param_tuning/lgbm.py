from typing import Tuple, Union, List, Dict
from datetime import datetime
from copy import copy

import pandas as pd
from overrides import overrides
import lightgbm as lgb
import optuna
from optuna.integration import lightgbm as lgb_optuna
optuna.logging.disable_default_handler()

from .base_tuner import IParamTuber


class LGBMRegressionTuner(IParamTuber):

    def __init__(self):
        self._fixed_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
        }

    @overrides
    def param_tuning(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
    ) -> Dict[str, Union[float, str, int]]:
        lgb_train = lgb.Dataset(
            X_train[:int(len(X_train)*0.8)],
            y_train[:int(len(y_train)*0.8)]
        )
        lgb_eval = lgb.Dataset(
            X_train[int(len(X_train)*0.8):],
            y_train[int(len(y_train)*0.8):]
        )

        opt = lgb_optuna.train(
            self._fixed_params,
            lgb_train, valid_sets=lgb_eval,
            verbose_eval=False,
            num_boost_round=100,
            # early_stopping_rounds=5,
            show_progress_bar=False,
        )
        return opt.params
