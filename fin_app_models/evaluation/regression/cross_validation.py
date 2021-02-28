import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

from ...model.structured_base.base_model import BaseRegressionModel
from ...utils.logger import Logger


def ts_cross_validate(
    model: BaseRegressionModel,
    y: pd.Series,
    X: pd.DataFrame,
    n_splits: int = 4
) -> pd.DataFrame:
    if not _validate_index(y, X):
        Logger.w('ts_cross_validate', "y and X's indices is not identilcal, merging index.")
        df_merged = pd.merge(
            y.to_frame('target'), X,
            left_index=True, right_index=True
        ).dropna()
        y = df_merged['target']
        X = df_merged[X.columns]

    rmses = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        model.train(
            y_train=y_train,
            X_train=X_train
        )
        y_pred_test = model.predict(X=X_test)
        rmse = np.sqrt(
            mean_squared_error(y_test.to_numpy(), y_pred_test.to_numpy())
        )
        rmses.append(rmse)

    return pd.DataFrame({
        'rmse': rmses
    })


def _validate_index(
    y_train: pd.Series,
    X_train: pd.DataFrame,
) -> bool:
    if len(y_train) != len(X_train):
        return False
    if y_train.index.min() != X_train.index.min():
        return False
    if y_train.index.max() != X_train.index.max():
        return False
    return True
