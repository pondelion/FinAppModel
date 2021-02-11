from typing import Tuple, Union, List, Dict
from datetime import datetime
from copy import copy

import pandas as pd
from overrides import overrides
from sklearn.neural_network import MLPRegressor
import optuna
optuna.logging.disable_default_handler()

from .optuna_tuner import RMSERegressionOptunaTuner


class SKMLPRegressionTuner(RMSERegressionOptunaTuner):

    def __init__(self):
        fixed_params = {
            'solver': 'adam',
            'random_state': 42
        }
        tuning_param = [
            'hidden_layer_sizes',
            'activation',
            'alpha',
            'learning_rate',
        ]
        super(SKMLPRegressionTuner, self).__init__(
            fixed_params,
            tuning_param,
            MLPRegressor
        )

    @overrides
    def _get_trial_params(self, trial) -> Dict:
        tuning_params = {}
        tuning_params['hidden_layer_sizes'] = trial.suggest_int('hidden_layer_sizes', 8, 100)
        tuning_params['activation'] = trial.suggest_categorical('activation', ['relu','tanh','logistic'])
        tuning_params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e1)
        tuning_params['learning_rate'] = trial.suggest_categorical('learning_rate', ['constant','adaptive'])
        return tuning_params
