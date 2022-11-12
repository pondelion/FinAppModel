import pandas as pd


def min_max_return(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    df_max_return = (df.rolling(window=periods, min_periods=1).max().shift(-periods) - df) / df
    df_min_return = (df.rolling(window=periods, min_periods=1).min().shift(-periods) - df) / df
    return df_max_return, df_min_return
