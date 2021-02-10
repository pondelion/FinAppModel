from copy import copy
from typing import List

import pandas as pd
import numpy as np
from talib import RSI, BBANDS, MACD, NATR, ATR, PPO, APO, CMO


def create_ohlc_features(
    sr_close: pd.Series,
    sr_open: pd.Series,
    sr_high: pd.Series,
    sr_low: pd.Series,
    macd_fastperiod: int=12,
    macd_slowperiod: int=26,
    macd_signalperiod: int=9,
    bb_period: int=20,
    basic_stats_period: int=14,
    atr_period: int=14,
    return_lags: List[int]=[1, 3, 7, 10, 20, 30, 60],
    col_name_prefix: str = None,
) -> pd.DataFrame:
    feature_dfs = [pd.DataFrame(
        index=sr_close.index,
        data={
            'close': sr_close,
            'open': sr_open,
            'high': sr_high,
            'low': sr_low
        }
    )]

    feature_dfs.append(rsi(sr_close).to_frame('rsi'))
    feature_dfs.append(macd(sr_close, macd_fastperiod, macd_slowperiod, macd_signalperiod).to_frame('macd'))
    high, mid, low = bollinger_band(sr_close, bb_period)
    feature_dfs.append(pd.DataFrame({
        'bb_high': high,
        'bb_mid': mid,
        'bb_low': low,
        'bb_std': 0.5*(high-low)
    }))
    feature_dfs.append(basic_stats(sr_close, window=basic_stats_period))
    atr_sr, natr_sr = atr(sr_close, high, low, atr_period)
    feature_dfs.append(pd.DataFrame({
        'atr': atr_sr,
        'natr': natr_sr,
    }))
    feature_dfs.append(oscillator(sr_close))
    df_return = returns(sr_close, return_lags)
    feature_dfs.append(df_return)
    feature_dfs.append(momentum(df_return, return_lags))
    feature_dfs.append(periodic(sr_close))

    df_feats = pd.concat(feature_dfs, axis=1)

    if col_name_prefix is not None:
        df_feats.columns = [f'{col_name_prefix}_{col_name}' for col_name in df_feats.columns]

    return df_feats


def rsi(ts: pd.Series, window: int=14) -> pd.Series:
    return RSI(ts, window)


def macd(
    ts: pd.Series,
    fastperiod: int=12,
    slowperiod: int=26,
    signalperiod: int=9
) -> pd.Series:
    macd, macdsignal, macdhist = MACD(
        ts,
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod
    )
    return macd.sub(macd.mean()).div(macd.std())


def bollinger_band(
    ts: pd.Series,
    period: int=20,
) -> pd.Series:
    high, mid, low = BBANDS(ts, timeperiod=period)
    return high, mid, low


def basic_stats(ts: pd.Series, window: int=14) -> pd.DataFrame:
    return pd.DataFrame(
        index=ts.index,
        data={
            f'max_{window}': ts.rolling(window=window).max(),
            f'min_{window}': ts.rolling(window=window).min(),
            f'median_{window}': ts.rolling(window=window).median(),
            f'skew_{window}': ts.rolling(window=window).skew(),
            f'kurt_{window}': ts.rolling(window=window).kurt(),
        }
    )


def atr(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int=14,
) -> pd.Series:
    atr = ATR(
        high=high, 
        low=low, 
        close=close, 
        timeperiod=period
    )
    natr = NATR(
        high=high, 
        low=low, 
        close=close
    ) 
    return atr.sub(atr.mean()).div(atr.std()), natr


def oscillator(
    ts: pd.Series,
    ppo_fastperiod: int=12,
    ppo_slowperiod: int=26,
    apo_fastperiod: int=12,
    apo_slowperiod: int=26,
    cmo_period: int=14,
) -> pd.DataFrame:
    return pd.DataFrame(
        index=ts.index,
        data={
            f'ppo_{ppo_fastperiod}_{ppo_slowperiod}': PPO(ts, ppo_fastperiod, ppo_slowperiod),
            f'apo_{apo_fastperiod}_{apo_slowperiod}': APO(ts, apo_fastperiod, apo_slowperiod),
            f'cmo_{cmo_period}': CMO(ts, cmo_period)
        }
    )


def returns(
    ts: pd.Series,
    lags: List[int],
) -> pd.DataFrame:

    return_srs = {}

    for lag in lags:
        return_srs[f'return_lag{lag}'] = ts.pct_change(lag).add(1).pow(1/lag).sub(1)

    return pd.DataFrame(return_srs)


def momentum(
    df_return: pd.DataFrame,
    lags: List[int],
    col_name_fmt: str='return_lag{lag}'
) -> pd.DataFrame:
    lags_cp = copy(lags)
    momentum_srs = {}
    base_lag = min(lags_cp)
    lags_cp.remove(base_lag)

    for lag in lags_cp:
        momentum_srs[f'momentum_{base_lag}_{lag}'] = df_return[col_name_fmt.format(lag=lag)] - df_return[col_name_fmt.format(lag=base_lag)]

    return pd.DataFrame(momentum_srs)


def periodic(
    df: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(
        index=df.index,
        data={
            'weekday': df.index.weekday,
            'year': df.index.year,
            'month': df.index.month
        }
    )
