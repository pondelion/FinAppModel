from datetime import datetime, timedelta
from typing import Union, Tuple

import pandas as pd
import numpy as np
from overrides import overrides

from .base_model import BaseRegressionModel
from ...entity.timeseries_data import TimeseriesData


class TrendLinearRegression(BaseRegressionModel):
    """Trend Linear Regression Model
    
    Example:
    >>> tlr_model = TrendLinearRegression()
    >>> tlr_model.train(y_train=sr_ts, trend_interval=365*3)
    >>> sr_pred = tlr_model.predict(pred_days=900)
    """

    @overrides
    def _train(
        self,
        y_train: pd.Series,
        trend_interval: timedelta = None,
        dt_now: datetime = None,
    ) -> float:

        ts_y_train = TimeseriesData(y_train)()

        if dt_now is None:
            dt_now = ts_y_train.index.max()
        if trend_interval is None:
            trend_interval = dt_now - ts_y_train.index.min()

        self._dt_now = dt_now

        self._slope, self._intercept = self._calc_reg_coeffs(
            ts_y_train, trend_interval, dt_now
        )

    @overrides
    def _predict(self, pred_days: int = 30) -> pd.Series:
        dts = pd.date_range(self._dt_now, self._dt_now+timedelta(days=pred_days))

        xs = np.arange(len(dts))
        ys = self._slope * xs + self._intercept

        return pd.Series(index=dts, data=ys)

    def _calc_reg_coeffs(
        self,
        ts_y: pd.Series,
        trend_interval: timedelta,
        dt_now: datetime
    ) -> Tuple[float, float]:
        t1 = dt_now - trend_interval
        t2 = dt_now
        ys = ts_y[(ts_y.index >= t1) & (ts_y.index <= t2)]
        xs = np.arange(len(ys))
        y_now = ys[dt_now]
        x_var = np.var(xs)
        if x_var == 1.0:
            raise Exception('x variance is zero')
        xy_cov = np.cov(xs, ys)[0, 1]
        slope = xy_cov / x_var
        intercept = y_now
        return (slope, intercept)


class SimpleLinearRegression(BaseRegressionModel):
    raise NotImplementedError


class MultipleLinearRegression(BaseRegressionModel):
    raise NotImplementedError
