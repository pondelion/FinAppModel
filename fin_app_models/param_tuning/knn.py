from typing import Tuple, Union, List, Dict
from datetime import datetime
from copy import copy

import pandas as pd
from overrides import overrides
from sklearn.neighbors import KNeighborsRegressor
import optuna
optuna.logging.disable_default_handler()

from .optuna_tuner import RMSERegressionOptunaTuner


class KNNRegressionTuner(RMSERegressionOptunaTuner):

    def __init__(self):
        fixed_params = {
        }
        tuning_param = [
            'n_neighbors',
        ]
        super(KNNRegressionTuner, self).__init__(
            fixed_params,
            tuning_param,
            KNeighborsRegressor
        )

    @overrides
    def _get_trial_params(self, trial) -> Dict:
        tuning_params = {}
        tuning_params['n_neighbors'] = trial.suggest_int('n_neighbors', 2, 15)
        return tuning_params
