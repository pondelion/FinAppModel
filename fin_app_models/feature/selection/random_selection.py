from typing import Dict, List
import random
from datetime import timedelta, date, MINYEAR, MAXYEAR

import pandas as pd

from ..creation.ohlc import create_ohlc_features
from ..creation.single_ts import create_single_ts_features


def random_feat_select(
    ohlc_df_dict: Dict[str, pd.DataFrame],
    single_ts_sr_dict: Dict[str, pd.Series],
    min_select_tss: int = 1,
    max_select_tss: int = 40,
    min_select_feats: int = 1,
    max_select_feats: int = 200,
    close_col_name: str = 'close',
    open_col_name: str = 'open',
    high_col_name: str = 'high',
    low_col_name: str = 'low',
) -> pd.DataFrame:
    max_select_tss = min(
        max_select_tss, len(ohlc_df_dict)+len(single_ts_sr_dict)
    )
    n_select_tss = random.randint(
        min_select_tss, max_select_tss
    )
    random_selected_ts_keys = random.sample(
        set(ohlc_df_dict.keys()) | set(single_ts_sr_dict.keys()), n_select_tss
    )
    random_ohlc_dfs = [(key, ohlc_df_dict[key]) for key in random_selected_ts_keys if key in list(ohlc_df_dict.keys())]
    random_single_ts_srs = [(key, single_ts_sr_dict[key]) for key in random_selected_ts_keys if key in list(single_ts_sr_dict.keys())]

    ohlc_feat_dfs = [create_ohlc_features(
        sr_close=df[close_col_name],
        sr_open=df[open_col_name],
        sr_high=df[high_col_name],
        sr_low=df[low_col_name],
        col_name_prefix=str(key),
    ) for key, df in random_ohlc_dfs]
    single_ts_feat_dfs = [create_single_ts_features(
        sr_ts=sr,
        col_name_prefix=str(key),
    ) for key, sr in random_single_ts_srs]

    # min_dt = min(
    #     min([df.index.min() for df in ohlc_feat_dfs]) if len(ohlc_feat_dfs) > 0 else date(MAXYEAR, 1, 1),
    #     min([df.index.min() for df in single_ts_feat_dfs]) if len(single_ts_feat_dfs) > 0 else date(MAXYEAR, 1, 1),
    # )
    # max_dt = max(
    #     max([df.index.max() for df in ohlc_feat_dfs]) if len(ohlc_feat_dfs) > 0 else date(MINYEAR, 1, 1),
    #     max([df.index.max() for df in single_ts_feat_dfs]) if len(single_ts_feat_dfs) > 0 else date(MINYEAR, 1, 1),
    # )

    # df_random_feats = pd.DataFrame(
    #     index=pd.date_range(min_dt, max_dt)
    # )
    df_random_feats = None

    for df in ohlc_feat_dfs+single_ts_feat_dfs:
        # df_random_feats = pd.merge(
        #     df_random_feats, df,
        #     left_index=True, right_index=True
        # )
        if df_random_feats is None:
            df_random_feats = df
        else:
            df_random_feats = pd.merge(
                df_random_feats, df, how='outer',
                left_index=True, right_index=True
            )

    max_select_feats = min(
        max_select_feats, len(df_random_feats.columns)
    )
    min_select_feats = min(min_select_feats, max_select_feats)
    n_select_feats = random.randint(
        min_select_feats, max_select_feats
    )

    random_feat_cols = random.sample(list(df_random_feats.columns), n_select_feats)

    return df_random_feats[random_feat_cols]


def random_time_window_select(
    dfs: List[pd.DataFrame],
    min_interval_days: int = 15,
    max_interval_days: int = 365*20,
    fix_latest_date: bool = False,
) -> List[pd.DataFrame]:
    min_dt = dfs[0].index.min()
    max_dt = dfs[0].index.max()

    if (max_dt - min_dt).days <= min_interval_days:
        return dfs

    max_interval_days = min(
        max_interval_days, (max_dt - min_dt).days
    )

    random_interval_days = random.randint(
        min_interval_days, max_interval_days
    )
    random_time_indices = [random.randint(
        0, len(df)-random_interval_days
    ) for df in dfs]

    if fix_latest_date:
        sliced_dfs = [df[-random_interval_days:] for df in dfs]
    else:
        sliced_dfs = [df[ti:ti+random_interval_days] for df, ti in zip(dfs, random_time_indices)]

    return sliced_dfs


def random_lagged_select(
    df: pd.DataFrame,
) -> pd.DataFrame:
    raise NotImplementedError
