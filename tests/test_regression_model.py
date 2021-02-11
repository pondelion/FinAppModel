import pandas as pd
import numpy as np

from fin_app_models.model.structured_base.regression import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
)


class TestRegressionModel:

    def test_linear_regression(self):
        dts = pd.date_range('2015-01-01', '2020-01-27')
        xs = np.arange(len(dts))
        test_ts = 4*np.sin(0.1*xs) + 7*np.cos(0.04*xs) + 0.01*xs + 2*np.random.randn(len(xs))
        y_train = pd.Series(index=dts, data=test_ts)
        X_train = pd.DataFrame(
            index=dts,
            data={
                'X1': test_ts+2*np.random.randn(len(xs)),
                'X2': -test_ts+2*np.random.randn(len(xs)),
                'X3': -test_ts+2*np.random.randn(len(xs)),
            }
        )
        lr_model = LinearRegression()
        lr_model.train(
            y_train=y_train,
            X_train=X_train
        )
        pred = lr_model.predict(X=X_train)
        print(pred)
        print(lr_model)

    def test_ridge_regression(self):
        dts = pd.date_range('2015-01-01', '2020-01-27')
        xs = np.arange(len(dts))
        test_ts = 4*np.sin(0.1*xs) + 7*np.cos(0.04*xs) + 0.01*xs + 2*np.random.randn(len(xs))
        y_train = pd.Series(index=dts, data=test_ts)
        X_train = pd.DataFrame(
            index=dts,
            data={
                'X1': test_ts+2*np.random.randn(len(xs)),
                'X2': -test_ts+2*np.random.randn(len(xs)),
                'X3': -test_ts+2*np.random.randn(len(xs)),
            }
        )
        ridge_model = RidgeRegression()
        ridge_model.train(
            y_train=y_train,
            X_train=X_train
        )
        pred = ridge_model.predict(X=X_train)
        print(pred)

    def test_lasso_regression(self):
        dts = pd.date_range('2015-01-01', '2020-01-27')
        xs = np.arange(len(dts))
        test_ts = 4*np.sin(0.1*xs) + 7*np.cos(0.04*xs) + 0.01*xs + 2*np.random.randn(len(xs))
        y_train = pd.Series(index=dts, data=test_ts)
        X_train = pd.DataFrame(
            index=dts,
            data={
                'X1': test_ts+2*np.random.randn(len(xs)),
                'X2': -test_ts+2*np.random.randn(len(xs)),
                'X3': -test_ts+2*np.random.randn(len(xs)),
            }
        )
        lasso_model = LassoRegression()
        lasso_model.train(
            y_train=y_train,
            X_train=X_train
        )
        pred = lasso_model.predict(X=X_train)
        print(pred)
