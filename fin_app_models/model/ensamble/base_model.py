from abc import ABCMeta, abstractmethod
from typing import Dict, List

import pandas as pd

from ..structured_base.models import (
    RegressionModel
)


class BaseRegressionEnsambleModel(metaclass=ABCMeta):

    @abstractmethod
    def train(
        self,
        X_train_ohlc_df_dict: Dict[str, pd.DataFrame],
        X_train_single_ts_sr_dict: Dict[str, pd.Series],
        y_train: pd.Series,
        auto_generate_feats: bool = True,
        n_estimators: int = 100,
        estimator_models: List[RegressionModel] = [
            RegressionModel.LINEAR,
            RegressionModel.KNN,
            RegressionModel.RIDGE,
            RegressionModel.LASSO,
            RegressionModel.ELASTIC_NET,
            RegressionModel.KERNEL_SVR,
            RegressionModel.RANDOM_FOREST,
            RegressionModel.SKMLP,
            RegressionModel.LGBM,
            RegressionModel.CAT_BOOST,
        ],
        open_col_name: str = 'open',
        close_col_name: str = 'close',
        high_col_name: str = 'high',
        low_col_name: str = 'low',
    ) -> None:
        raise NotImplementedError

    def predict(
        self,
        X_ohlc_df_dict: Dict[str, pd.DataFrame],
        X_single_ts_sr_dict: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        preds = self._predict(
            X_ohlc_df_dict, X_single_ts_sr_dict
        )
        return self._aggregate(preds)

    @abstractmethod
    def _predict(
        self,
        X_ohlc_df_dict: Dict[str, pd.DataFrame],
        X_single_ts_sr_dict: Dict[str, pd.Series],
    ) -> List[pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def _aggregate(
        self,
        preds: List[pd.DataFrame],
    ) -> pd.DataFrame:
        raise NotImplementedError
