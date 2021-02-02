from typing import Tuple, Union, List, Dict
from datetime import datetime
from copy import copy

import pandas as pd
from overrides import overrides
from sklearn.ensemble import RandomForestRegressor
import optuna
optuna.logging.disable_default_handler()

from .regression_optuna_tuner import RMSERegressionOptunaTuner


class RandomForestRegressionTuner(RMSERegressionOptunaTuner):

    def __init__(self):
        fixed_params = {
            'random_state': 42,
        }
        tuning_param = [
            'n_estimators',
            'max_depth'
        ]
        super(RandomForestRegressionTuner, self).__init__(
            fixed_params,
            tuning_param,
            RandomForestRegressor
        )

    @overrides
    def _get_trial_params(self, trial) -> Dict:
        tuning_params = {}
        tuning_params['n_estimators'] = trial.suggest_int('n_estimators', 2, 20)
        tuning_params['max_depth'] = int(trial.suggest_loguniform('max_depth', 1, 32))
        return tuning_params
