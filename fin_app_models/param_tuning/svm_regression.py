from typing import Tuple, Union, List, Dict
from datetime import datetime
from copy import copy

import pandas as pd
from overrides import overrides
from sklearn.svm import SVR
import optuna
optuna.logging.disable_default_handler()

from .regression_optuna_tuner import RMSERegressionOptunaTuner


class KernelSVRRegressionTuner(RMSERegressionOptunaTuner):

    def __init__(self):
        fixed_params = {
            'max_iter': 1000,
        }
        tuning_param = [
            'kernel',
            'gamma',
            'C'
        ]
        super(KernelSVRRegressionTuner, self).__init__(
            fixed_params,
            tuning_param,
            SVR
        )

    @overrides
    def _get_trial_params(self, trial) -> Dict:
        tuning_params = {}
        tuning_params['kernel'] = trial.suggest_categorical('kernel',['poly','rbf'])
        tuning_params['gamma'] = trial.suggest_loguniform('gamma',1e-4,1e4)
        tuning_params['C'] = trial.suggest_loguniform('C',1e-4,1e4)
        return tuning_params
