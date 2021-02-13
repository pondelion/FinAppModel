import random
from typing import Dict, List

from overrides import overrides
import pandas as pd

from ..base_model import BaseRegressionEnsambleModel
from ...structured_base.models import (
    RegressionModel
)
from ....feature.selection.random_selection import random_feat_select
from ....feature.creation.ohlc import create_ohlc_features
from ....feature.creation.single_ts import create_single_ts_features


class Broccoli(BaseRegressionEnsambleModel):

    @overrides
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
        model_clss = random.choices(estimator_models, n_estimators)

        self._estimators = [model.value() for model in model_clss]

        self._X_cols = []
        for estimator in self._estimators:
            if auto_generate_feats:
                df_train_feat = random_feat_select(
                    ohlc_df_dict=X_train_ohlc_df_dict,
                    single_ts_sr_dict=X_train_single_ts_sr_dict,
                    open_col_name=open_col_name,
                    close_col_name=close_col_name,
                    high_col_name=high_col_name,
                    low_col_name=low_col_name,
                )
                self._X_cols.append(list(df_train_feat.columns))
            else:
                raise NotImplementedError

            estimator.train(
                y_train=y_train,
                X_train=df_train_feat,
            )

    @overrides
    def _predict(
        self,
        X_ohlc_df_dict: Dict[str, pd.DataFrame],
        X_single_ts_sr_dict: Dict[str, pd.Series],
        open_col_name: str = 'open',
        close_col_name: str = 'close',
        high_col_name: str = 'high',
        low_col_name: str = 'low',
    ) -> List[pd.DataFrame]:
        ohlc_feat_dfs = [create_ohlc_features(
            sr_close=df[close_col_name],
            sr_open=df[open_col_name],
            sr_high=df[high_col_name],
            sr_low=df[low_col_name],
            col_name_prefix=str(key),
        ) for key, df in X_ohlc_df_dict.items()]
        single_ts_feat_dfs = [create_single_ts_features(
            sr_ts=sr,
            col_name_prefix=str(key),
        ) for key, sr in X_single_ts_sr_dict.items()]
        df_feats = pd.concat([
            pd.concat(ohlc_feat_dfs, axis=1),
            pd.concat(single_ts_feat_dfs, axis=1)
        ], axis=1)
        preds = [
            estimator.predict(df_feats[cols]) for estimator, cols in zip(self._estimators, self._X_cols)
        ]
        return preds

    @overrides
    def _aggregate(
        self,
        preds: List[pd.DataFrame],
    ) -> pd.DataFrame:
        return pd.concat(preds, axis=1).mean(axis=1)
