from typing import Optional, Union

import pandas as pd


def return_days(ts: Union[pd.Series, pd.DataFrame], days: int = 1):
    return (ts.shift(-days) - ts) / ts


def return_lags(
    base_ts: Union[pd.Series, pd.DataFrame],
    target_ts: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lag: int = 1,
    order_lag: int = 1,
) -> Union[pd.Series, pd.DataFrame]:
    """現時点tにおけるリターンラグを、
    (現時点+オーダーラグ)時点のbase_ts→(現時点+オーダーラグ+リターンラグ)時点のtarget_tsに対する変化率として計算する
    return(t; lag, order_lag) = target_ts(t+lag+order_lag) / base_ts(t+order_lag) - 1

    Args:
        base_ts (Union[pd.Series, pd.DataFrame]): リターン計算の始点とする時系列データ
        target_ts (Optional[Union[pd.Series, pd.DataFrame]], optional): リターン計算の終点とする時系列データ. Defaults to None.
        lag (int, optional): リターンラグ. Defaults to 1.
        order_lag (int, optional): オーダーラグ. 例)1の場合現時点から1期先時にオーダーを出す想定. Defaults to 1.

    Returns:
        Union[pd.Series, pd.DataFrame]: リターン
    """
    if target_ts is None:
        target_ts = base_ts
    return (target_ts.shift(-(lag + order_lag)) / base_ts.shift(-order_lag)) - 1.0
