import pandas as pd
import numpy as np
import pytest

from fin_app_models.model.structured_base.regression import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    LGBMRegression,
    CatBoostRegression,
    SKMLPRegression,
    KernelSVRRegression,
)
from fin_app_models.model.structured_base.timeseries_model.dnn.lstm import (
    BILSTMRegression,
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
            X_train=dummy_data['X'],
            auto_param_tuning=False,
        )
        pred = lgbm_model.predict(X=dummy_data['X'])
        print(pred)
        print(lgbm_model.feature_importance())

    def test_catboost_regression(self, dummy_data):
        catboost_model = CatBoostRegression()
        catboost_model.train(
            y_train=dummy_data['y'],
            X_train=dummy_data['X'],
            auto_param_tuning=False,
        )
        pred = catboost_model.predict(X=dummy_data['X'])
        print(pred)

    def test_kernel_svm_regression(self, dummy_data):
        svr_model = KernelSVRRegression()
        svr_model.train(
            y_train=dummy_data['y'],
            X_train=dummy_data['X']
        )
        pred = svr_model.predict(X=dummy_data['X'])
        print(pred)

    def test_skmlp_regression(self, dummy_data):
        skmlp_model = SKMLPRegression()
        skmlp_model.train(
            y_train=dummy_data['y'],
            X_train=dummy_data['X']
        )
        pred = skmlp_model.predict(X=dummy_data['X'])
        print(pred)

    def test_bilstm_regression(self, dummy_data):
        bilstm_model = BILSTMRegression()
        bilstm_model.train(
            y_train=dummy_data['y'],
            X_train=dummy_data['X'],
            seq_len=64,
            n_epoch=5,
        )
        pred = bilstm_model.predict(
            # y=dummy_data['y'],
            X=dummy_data['X']
        )
        print(pred)
