from copy import copy
from typing import Dict

from fastprogress import progress_bar as pb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .random_selection import random_feat_select


def permutation_importance_reg(X_train, y_train, X_val = None, y_val = None, model = None, metric = mean_squared_error, n_repeat: int = 8):
    if model is None:
        model = RandomForestRegressor()
    if X_val is None:
        X_val = X_train
    if y_val is None:
        y_val = y_train
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    base_score = metric(np.array(y_val).flatten(), val_preds.flatten())
    if isinstance(X_val, pd.DataFrame):
        cols = X_val.columns.tolist()
    elif isinstance(X_val, np.ndarray):
        cols = list(map(str, range(X_val.shape[1])))
        X_val = pd.DataFrame(data=X_val, columns=cols)
    else:
        raise TypeError(f'unsupported X_val type : {X_val.type}')

    perm_scores = {}
    for i in range(n_repeat):
        ith_perm_scores = []
        replace_vals = np.random.randn(len(X_val))
        for col in X_val.columns:
            X_val_perm = X_val.copy()
            # replace_vals = X_val_perm[col].sample(frac=1).values
            X_val_perm[col] = replace_vals
            val_preds = model.predict(X_val_perm)
            perm_score = metric(np.array(y_val).flatten(), val_preds.flatten())
            ith_perm_scores.append(perm_score)
        perm_scores[f'{i}th_perm_score'] = ith_perm_scores

    df_perm_score = pd.DataFrame(
        index=X_val.columns.tolist(),
        data=perm_scores
    )
    df_perm_score['base_score'] = [base_score]*len(X_val.columns)
    df_perm_score['mean_perm_score'] = df_perm_score[list(perm_scores.keys())].mean(axis=1)
    df_perm_score['diff_score'] = df_perm_score['mean_perm_score'] - df_perm_score['base_score']
    df_perm_score = df_perm_score.sort_values(by='diff_score', ascending=False)
    return df_perm_score


def repeative_high_importance_feats_search_reg(
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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)
    prev_selected_feats = None
    inportance_dfs = []
    for _ in pb(range(n_repeats)):
        df_perm_importance = permutation_importance_reg(X_train, y_train, X_val, y_val, n_repeat=3)
        inportance_dfs.append(df_perm_importance[:int(feat_len*(1-replace_rate))])
        if (df_perm_importance['mean_perm_score']==0).all() and prev_selected_feats is not None:
            selected_feats = copy(prev_selected_feats)
        else:
            # Remain feature_num*(1-replace_rate) high importance features as next candisates.
            selected_feats = list(df_perm_importance[:int(feat_len*(1-replace_rate))].index)
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)

    return inportance_dfs[::-1]
