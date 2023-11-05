from typing import Tuple, Union, List, Dict
from datetime import datetime
from copy import copy

import pandas as pd
from overrides import overrides
from sklearn.ensemble import RandomForestRegressor
import optuna
optuna.logging.disable_default_handler()

from .optuna_tuner import RMSERegressionOptunaTuner


class RandomForestRegressionTuner(RMSERegressionOptunaTuner):

    def __init__(self):
        fixed_params = {
            'random_state': 42,
        }
        tuning_param = [
            'n_estimators',
            'max_depth',
            'max_features',
        ]
        super(RandomForestRegressionTuner, self).__init__(
            fixed_params,
            tuning_param,
            RandomForestRegressor
        )

    @overrides
    def _get_trial_params(self, trial) -> Dict:
        tuning_params = {}
        tuning_params['n_estimators'] = trial.suggest_int('n_estimators', 2, 200)
        tuning_params['max_depth'] = int(trial.suggest_loguniform('max_depth', 1, 96))
        tuning_params['max_features'] = trial.suggest_float('n_estimators', 0.1, 1.0)
        return tuning_params
