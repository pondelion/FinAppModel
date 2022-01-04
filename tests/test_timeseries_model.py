import pandas as pd
import numpy as np

from fin_app_models.model.structured_base.timeseries_model import (
    TrendLinearRegression,
)


class TestTimeseriesModel:

    def test_trend_linear_regression_model(self):
        dts = pd.date_range('2015-01-01', '2020-01-27')
        xs = np.arange(len(dts))

        test_ts = 4*np.sin(0.1*xs) + 7*np.cos(0.04*xs) + 0.01*xs + 2*np.random.randn(len(xs))
        sr_ts = pd.Series(index=dts,data=test_ts)

        tlr_model = TrendLinearRegression()
        tlr_model.train(y_train=sr_ts, trend_interval_days=365*3)
        sr_pred = tlr_model.predict(pred_periods=40)
        assert isinstance(sr_pred, pd.Series)
        assert len(sr_pred)==40+1
        print(sr_pred)
