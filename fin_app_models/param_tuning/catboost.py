from typing import Tuple, Union, List, Dict
from datetime import datetime
from copy import copy

import pandas as pd
from overrides import overrides
import catboost
import optuna
optuna.logging.disable_default_handler()

from .optuna_tuner import RMSERegressionOptunaTuner


class CatBoostRegressionTuner(RMSERegressionOptunaTuner):

    def __init__(self):
        fixed_params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'use_best_model': True,
            'od_type' : 'Iter',
            'od_wait' : 100,
            'random_seed': 42,
            'one_hot_max_size': 1024,
            'verbose': False,
        }
        tuning_param = [
            'depth',
            'l2_leaf_reg',
            'eval_metric',
        ]
        super(CatBoostRegressionTuner, self).__init__(
            fixed_params,
            tuning_param,
            CatBoostRegressionTuner
        )

    @overrides
    def _get_trial_params(self, trial) -> Dict:
        tuning_params = {}
        tuning_params['depth'] = trial.suggest_int('depth', 2, 10)
        tuning_params['l2_leaf_reg'] = trial.suggest_loguniform('l2_leaf_reg', 1e-8, 100)
        tuning_params['eval_metric'] = trial.suggest_categorical('eval_metric', ['MAE','RMSE'])
        return tuning_params
