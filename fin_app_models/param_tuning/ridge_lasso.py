from typing import Tuple, Union, List, Dict
from datetime import datetime
from copy import copy

import pandas as pd
from overrides import overrides
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet
)
import optuna
optuna.logging.disable_default_handler()

from .optuna_tuner import RMSERegressionOptunaTuner


class RidgeRegressionTuner(RMSERegressionOptunaTuner):

    def __init__(self):
        fixed_params = {
            'fit_intercept': True,
            'max_iter': 1000,
            'random_state': 42,
        }
        tuning_param = [
            'alpha'
        ]
        super(RidgeRegressionTuner, self).__init__(
            fixed_params,
            tuning_param,
            Ridge
        )

    @overrides
    def _get_trial_params(self, trial) -> Dict:
        tuning_params = {}
        tuning_params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e5)
        return tuning_params


class LassoRegressionTuner(RMSERegressionOptunaTuner):

    def __init__(self):
        fixed_params = {
            'fit_intercept': True,
            'max_iter': 1000,
            'random_state': 42,
        }
        tuning_param = [
            'alpha'
        ]
        super(LassoRegressionTuner, self).__init__(
            fixed_params,
            tuning_param,
            Lasso
        )

    @overrides
    def _get_trial_params(self, trial) -> Dict:
        tuning_params = {}
        tuning_params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e5)
        return tuning_params


class ElasticNetRegressionTuner(RMSERegressionOptunaTuner):

    def __init__(self):
        fixed_params = {
            'random_state': 42,
        }
        tuning_param = [
            'alpha',
            'l1_ratio'
        ]
        super(ElasticNetRegressionTuner, self).__init__(
            fixed_params,
            tuning_param,
            ElasticNet
        )

    @overrides
    def _get_trial_params(self, trial) -> Dict:
        tuning_params = {}
        tuning_params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e3)
        tuning_params['l1_ratio'] = trial.suggest_discrete_uniform('l1_ratio', 0.1, 0.9, 0.1)
        return tuning_params
