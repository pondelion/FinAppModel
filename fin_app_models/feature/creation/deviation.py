from typing import List, Optional


def deviation_from_target_feats(df, base_col: str, target_cols: Optional[List[str]] = None):
    if target_cols is None:
        target_cols = df.columns.tolist()
        target_cols.remove(base_col)
    df_deviation = df[target_cols].div(df[base_col], axis=0)
    df_deviation.columns = [f'deviation_from_{col_name}' for col_name in df_deviation.columns]


def deviation_from_sector_mean():
    raise NotImplementedError


def deviation_from_financial_data_cluster_mean():
    raise NotImplementedError


def deviation_from_timeseries_cluster_mean():
    raise NotImplementedError
