from typing import Dict, List

import pandas as pd

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

    for _ in range(n_repeats):
        model = LGBMRegression()
        model.train(
            y_train=X,
            X_train=y
        )
        df_importance = model.feature_importance().sort_values(
            by='importance', ascending=False
        )
        # Remain feature_num*(1-replace_rate) high importance features as next candisates.
        selected_cols = list(df_importance[:int(len(df_importance)*(1-replace_rate))].index)
        df_condiate_feats = pd.merge(
            sr_y.rename('target'), df_merged[selected_cols],
            left_index=True, right_index=True,
        ).dropna()
        # Adopt feature_num*(replace_rate) newly created features as next candidates.
        df_new_random_feats = random_feat_select(
            ohlc_df_dict=ohlc_ts_X_df_dict,
            single_ts_sr_dict=single_ts_X_sr_dict,
            min_select_tss=min_select_tss,
            max_select_tss=max_select_tss,
            min_select_feats=int(len(df_importance)*(replace_rate)),
            max_select_feats=int(len(df_importance)*(replace_rate)),
            close_col_name=close_col_name,
            open_col_name=open_col_name,
            high_col_name=high_col_name,
            low_col_name=low_col_name,
        )
        df_merged = pd.merge(
            df_condiate_feats, df_new_random_feats,
            left_index=True, right_index=True,
        ).dropna()
        X = df_merged[selected_cols+list(df_new_random_feats.columns)]
        y = df_merged['target']

    return df_importance[:int(len(df_importance)*(1-replace_rate))]
