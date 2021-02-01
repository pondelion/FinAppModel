from typing import Tuple, Union, List, Dict
from datetime import datetime
from copy import copy

import pandas as pd
from overrides import overrides
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge, Lasso
import optuna
optuna.logging.disable_default_handler()

from .base_tuner import IParamTuber


class RidgeRegressionTuner(IParamTuber):

    def __init__(self):
        self._fixed_params = {
            'fit_intercept': True,
            'max_iter': 1000,
            'random_state': 42,
        }
        self._tuning_param = [
            'alpha'
        ]

    @overrides
    def param_tuning(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
    ) -> Dict[str, Union[float, str, int]]:

        self._X_train = X_train
        self._y_train = y_train

        study = optuna.create_study()
        optuna.logging.disable_default_handler()
        study.optimize(self._objective, n_trials=100)

        best_params = study.best_params

        return best_params

    def _objective(self, trial):

        tscv = TimeSeriesSplit(n_splits=5)

        params = copy(self._fixed_params)
        params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e5)

        model = Ridge(**params)

        scores = cross_validate(
            model, self._X_train, self._y_train, cv=tscv, scoring='neg_root_mean_squared_error'
        )
        rmse_mean = scores['test_score'].mean()

        return rmse_mean


class LassoRegressionTuner(IParamTuber):

    def __init__(self):
        self._fixed_params = {
            'fit_intercept': True,
            'max_iter': 1000,
            'random_state': 42,
        }
        self._tuning_param = [
            'alpha'
        ]

    @overrides
    def param_tuning(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
    ) -> Dict[str, Union[float, str, int]]:

        self._X_train = X_train
        self._y_train = y_train

        study = optuna.create_study()
        optuna.logging.disable_default_handler()
        study.optimize(self._objective, n_trials=100)

        best_params = study.best_params

        return best_params

    def _objective(self, trial):

        tscv = TimeSeriesSplit(n_splits=5)

        params = copy(self._fixed_params)
        params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e5)

        model = Lasso(**params)

        scores = cross_validate(
            model, self._X_train, self._y_train, cv=tscv, scoring='neg_root_mean_squared_error'
        )
        rmse_mean = scores['test_score'].mean()

        return rmse_mean
