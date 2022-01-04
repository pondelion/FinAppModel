from datetime import datetime, timedelta
from typing import Union, Tuple, List, Dict

import pandas as pd
import numpy as np
from overrides import overrides

from ..base_model import BaseTimeseriesModel
from ....processing import (
    IStructuredDataProcessing,
    TrendLinearRegressionDataProcessing,
)
from ....param_tuning import (
    IParamTuber,
    DefaultTuner
)


class TrendLinearRegression(BaseTimeseriesModel):
    """Trend Linear Regression Model
    
    Example:
    >>> tlr_model = TrendLinearRegression()
    >>> tlr_model.train(y_train=sr_ts, trend_interval_days=365*3)
    >>> sr_pred = tlr_model.predict(pred_periods=900)
    """

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [TrendLinearRegressionDataProcessing()],
        param_tuner: IParamTuber = DefaultTuner(),
    ):
        super(TrendLinearRegression, self).__init__(data_processors, param_tuner)

    @overrides
    def _train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs,
    ) -> None:

        if dt_now is None:
            dt_now = y_train.index.max()
        if 'trend_interval_days' not in kwargs:
            kwargs['trend_interval_days'] = (dt_now - y_train.index.min()).days

        self._dt_now = dt_now

        self._slope, self._intercept = self._calc_reg_coeffs(
            y_train, timedelta(days=int(kwargs['trend_interval_days'])), dt_now
        )

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        pred_periods: int = 30,
        **kwargs,
    ) -> pd.Series:
        dts = pd.date_range(self._dt_now, self._dt_now+timedelta(days=pred_periods))

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
