from typing import Dict, List
from copy import copy

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from fastprogress import progress_bar as pb

from .random_selection import random_feat_select
from ...model.structured_base.regression import LGBMRegression


def lgbm_reg_random_feats_search(
    sr_y: pd.Series,
    ohlc_ts_X_df_dict: Dict[str, pd.DataFrame],
    single_ts_X_sr_dict: Dict[str, pd.Series],
    min_select_tss: int = 2,
    max_select_tss: int = 40,
    min_select_feats: int = 2,
    max_select_feats: int = 200,
    close_col_name: str = 'close',
    open_col_name: str = 'open',
    high_col_name: str = 'high',
    low_col_name: str = 'low',
    n_repeats: int = 20,
    replace_rate: float = 0.5,
) -> pd.DataFrame:

    df_random_feats = random_feat_select(
        ohlc_df_dict=ohlc_ts_X_df_dict,
        single_ts_sr_dict=single_ts_X_sr_dict,
        min_select_tss=min_select_tss,
        max_select_tss=max_select_tss,
        min_select_feats=min_select_feats,
        max_select_feats=max_select_feats,
        close_col_name=close_col_name,
        open_col_name=open_col_name,
        high_col_name=high_col_name,
        low_col_name=low_col_name,
    )
    df_merged = pd.merge(
        sr_y.rename('target'), df_random_feats,
        left_index=True, right_index=True,
    ).dropna()
    X = df_merged[df_random_feats.columns]
    y = df_merged['target']
    feat_len = len(X.columns)
    inportance_dfs = []
    rmses = []
    prev_selected_feats = None

    for _ in pb(range(n_repeats)):
        model = LGBMRegression()
        model.train(
            y_train=y,
            X_train=X
        )
        y_pred = model.predict(X=X)
        df_pred = pd.merge(
            y.to_frame('y'), y_pred.to_frame('y_pred'),
            how='inner',
            left_index=True, right_index=True
        )
        rmse = np.sqrt(
            mean_squared_error(df_pred['y'].to_numpy(), df_pred['y_pred'].to_numpy())
        )
        rmses.append(rmse)
        df_importance = model.feature_importance().sort_values(
            by='importance', ascending=False
        )
        inportance_dfs.append(df_importance[:int(feat_len*(1-replace_rate))])
        if (df_importance['importance']==0).all() and prev_selected_feats is not None:
            selected_feats = copy(prev_selected_feats)
        else:
            # Remain feature_num*(1-replace_rate) high importance features as next candisates.
            selected_feats = list(df_importance[:int(feat_len*(1-replace_rate))].index)
        df_condiate_feats = pd.merge(
            sr_y.rename('target'), df_merged[selected_feats],
            left_index=True, right_index=True,
        ).dropna()
        prev_selected_feats = copy(selected_feats)
        # Adopt feature_num*(replace_rate) newly created features as next candidates.
        df_new_random_feats = random_feat_select(
            ohlc_df_dict=ohlc_ts_X_df_dict,
            single_ts_sr_dict=single_ts_X_sr_dict,
            min_select_tss=min_select_tss,
            max_select_tss=max_select_tss,
            min_select_feats=int(feat_len*(replace_rate)),
            max_select_feats=int(feat_len*(replace_rate)),
            close_col_name=close_col_name,
            open_col_name=open_col_name,
            high_col_name=high_col_name,
            low_col_name=low_col_name,
        )
        new_feats = list(set(df_new_random_feats.columns)-set(selected_feats))  # delete duplicates
        df_merged = pd.merge(
            df_condiate_feats, df_new_random_feats[new_feats],
            left_index=True, right_index=True,
        ).dropna()
        X = df_merged[selected_feats+list(df_new_random_feats.columns)]
        y = df_merged['target']

    return inportance_dfs[::-1], rmses
