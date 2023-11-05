from typing import Dict, List, Optional
import random

import pandas as pd

from ..creation.ohlc import create_ohlc_features
from ..creation.single_ts import create_single_ts_features


def random_feat_select(
    ohlc_df_dict: Dict[str, pd.DataFrame],
    single_ts_sr_dict: Dict[str, pd.Series],
    min_select_tss: int = None,
    max_select_tss: int = None,
    min_select_feats: int = 20,
    max_select_feats: int = 200,
    close_col_name: str = 'close',
    open_col_name: str = 'open',
    high_col_name: str = 'high',
    low_col_name: str = 'low',
    # for single/ohlc ts feature creation
    macd_fastperiod: int = 12,
    macd_slowperiod: int = 26,
    macd_signalperiod: int = 9,
    bb_periods: int = [7, 20, 30, 60],
    basic_stats_period: int = 14,
    atr_period: int = 14,
    return_lags: List[int] = [1, 3, 7, 10, 20, 30, 60],
    include_deviation: bool = True,
    reproduce_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if max_select_tss is not None:
        max_select_tss = min(
            max_select_tss, len(ohlc_df_dict)+len(single_ts_sr_dict)
        )
    else:
        max_select_tss = len(ohlc_df_dict)+len(single_ts_sr_dict)
    if min_select_tss is None:
        min_select_tss = max_select_tss
    min_select_tss = min(min_select_tss, max_select_tss)
    n_select_tss = random.randint(
        min_select_tss, max_select_tss
    )
    if reproduce_cols is None:
        selected_ts_keys = random.sample(
            set(ohlc_df_dict.keys()) | set(single_ts_sr_dict.keys()), n_select_tss
        )
    else:
        selected_ts_keys = set(ohlc_df_dict.keys()) | set(single_ts_sr_dict.keys())
    random_ohlc_dfs = [(key, ohlc_df_dict[key]) for key in selected_ts_keys if key in list(ohlc_df_dict.keys())]
    random_single_ts_srs = [(key, single_ts_sr_dict[key]) for key in selected_ts_keys if key in list(single_ts_sr_dict.keys())]

    ohlc_feat_dfs = [create_ohlc_features(
        sr_close=df[close_col_name],
        sr_open=df[open_col_name],
        sr_high=df[high_col_name],
        sr_low=df[low_col_name],
        macd_fastperiod=macd_fastperiod,
        macd_slowperiod=macd_slowperiod,
        macd_signalperiod=macd_signalperiod,
        bb_periods=bb_periods,
        basic_stats_period=basic_stats_period,
        atr_period=atr_period,
        return_lags=return_lags,
        col_name_prefix=str(key),
    ) for key, df in random_ohlc_dfs]
    # ohlc_feat_dfs += [
    #     df[[close_col_name, open_col_name, high_col_name, low_col_name]].add_prefix(f'{key}_')
    #     for key, df in random_ohlc_dfs
    # ]
    single_ts_feat_dfs = [create_single_ts_features(
        sr_ts=sr,
        col_name_prefix=str(key),
        macd_fastperiod=macd_fastperiod,
        macd_slowperiod=macd_slowperiod,
        macd_signalperiod=macd_signalperiod,
        bb_periods=bb_periods,
        basic_stats_period=basic_stats_period,
        atr_period=atr_period,
        return_lags=return_lags,
        include_deviation=include_deviation,
    ) for key, sr in random_single_ts_srs]
    # single_ts_feat_dfs += [sr.to_frame(str(key)) for key, sr in random_single_ts_srs]

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

    if reproduce_cols is None:
        # randomly choose features
        max_select_feats = min(
            max_select_feats, len(df_random_feats.columns)
        )
        min_select_feats = min(min_select_feats, max_select_feats)
        n_select_feats = random.randint(
            min_select_feats, max_select_feats
        )

        random_feat_cols = random.sample(list(df_random_feats.columns), n_select_feats)
        selected_cols = random_feat_cols
    else:
        diff_cols = set(reproduce_cols) - set(list(df_random_feats.columns))
        if len(diff_cols) > 0:
            raise ValueError(
                f'reproduce_cols {diff_cols} not found in created features, '
                'maybe data or parameters are mismatched.'
            )
        selected_cols = reproduce_cols

    return df_random_feats[selected_cols]


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
