import pandas as pd
import numpy as np
import pytest

from fin_app_models.model.structured_base.regression import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    LGBMRegression,
    CatBoostRegression,
)


@pytest.fixture
def dummy_data():
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
    return {
        'y': y_train,
        'X': X_train
    }


class TestRegressionModel:

    def test_linear_regression(self, dummy_data):
        lr_model = LinearRegression()
        lr_model.train(
            y_train=dummy_data['y'],
            X_train=dummy_data['X']
        )
        pred = lr_model.predict(X=dummy_data['X'])
        print(pred)
        print(lr_model)

    def test_ridge_regression(self, dummy_data):
        ridge_model = RidgeRegression()
        ridge_model.train(
            y_train=dummy_data['y'],
            X_train=dummy_data['X']
        )
        pred = ridge_model.predict(X=dummy_data['X'])
        print(pred)

    def test_lasso_regression(self, dummy_data):
        lasso_model = LassoRegression()
        lasso_model.train(
            y_train=dummy_data['y'],
            X_train=dummy_data['X']
        )
        pred = lasso_model.predict(X=dummy_data['X'])
        print(pred)

    def test_lgbm_regression(self, dummy_data):
        lgbm_model = LGBMRegression()
        lgbm_model.train(
            y_train=dummy_data['y'],
            X_train=dummy_data['X']
        )
        pred = lgbm_model.predict(X=dummy_data['X'])
        print(pred)
        print(lgbm_model.feature_importance())

    def test_catboost_regression(self, dummy_data):
        catboost_model = CatBoostRegression()
        catboost_model.train(
            y_train=dummy_data['y'],
            X_train=dummy_data['X']
        )
        pred = catboost_model.predict(X=dummy_data['X'])
        print(pred)
