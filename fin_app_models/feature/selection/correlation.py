from typing import List

import numpy as np
import pandas as pd


def _calc_lag_corr_df(
    df_ts: pd.DataFrame,
    target_col: str,
    max_search_lag: int=300,
    min_periods: int=None
):
    df_ts_lag = df_ts.copy()
    df_corrs = []
    corr_args = {'min_periods': min_periods} if min_periods is not None else {}
    for lag in range(max_search_lag):
        df_ts_lag['target_lag'] = df_ts[target_col].shift(-(lag+1))
        df_corr = df_ts_lag.corr(**corr_args)['target_lag'].to_frame(f'lag{lag+1}').T
        df_corrs.append(df_corr)
    return pd.concat(df_corrs).drop(['target_lag', target_col], axis=1)


def find_corr_max_lags(
    df_ts: pd.DataFrame,
    target_col: str,
    max_search_lag: int=300,
    min_periods: int=None
):
    df_lag_corr = _calc_lag_corr_df(
        df_ts, target_col, max_search_lag, min_periods
    )
    return {col: {'max_corr_lag': df_lag_corr[col].argmax()+1, 'max_corr': df_lag_corr[col].max()} for col in df_lag_corr.columns}


def delete_high_corr_feats(
    df_X: pd.DataFrame,
    sr_y: pd.Series,
    corr_thres: 0.6,
) -> List[str]:
    df_Xy = df_X.copy()
    df_Xy['_y_'] = sr_y
    df_corr = df_Xy.corr()
    drop_cols = set()
    for i, col1 in enumerate(df_X.columns):
        for j, col2 in enumerate(df_X.columns):
            if i >= j:
                continue
            corr = df_corr.loc[col1, col2]
            if corr_thres > corr:
                continue
            if df_corr.loc[col1, '_y_'] > df_corr.loc[col2, '_y_']:
                drop_cols.add(col1)
                # print(f'{col1} > {col2}')
            else:
                drop_cols.add(col2)
                # print(f'{col1} < {col2}')
    return list(set(df_X.columns) - drop_cols)
