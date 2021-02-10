import pandas as pd
import numpy as np
import pytest

from fin_app_models.feature.creation.ohlc import create_ohlc_features
from fin_app_models.feature.creation.single_ts import create_single_ts_features
from fin_app_models.feature.selection.random_selection import random_feat_select


@pytest.fixture
def ohlc_dummy_data():
    dts = pd.date_range('2015-01-01', '2020-01-27')
    xs = np.arange(len(dts))
    test_ts = 4*np.sin(0.1*xs) + 7*np.cos(0.04*xs) + 0.01*xs + 2*np.random.randn(len(xs))
    X = pd.DataFrame(
        index=dts,
        data={
            'open': test_ts,
            'close': test_ts+2*(np.random.rand(len(xs))-0.5),
            'high': test_ts+5,
            'low': test_ts-4,
        }
    )
    return X


@pytest.fixture
def single_ts_dummy_data():
    dts = pd.date_range('2015-01-01', '2020-01-27')
    xs = np.arange(len(dts))
    test_ts = 4*np.sin(0.1*xs) + 7*np.cos(0.04*xs) + 0.01*xs + 2*np.random.randn(len(xs))
    X = pd.Series(index=dts, data=test_ts, name='close')
    return X


class TestFeature:

    def test_feature_creation_ohlc(self, ohlc_dummy_data):
        df_ohlc_feats = create_ohlc_features(
            sr_close=ohlc_dummy_data['close'],
            sr_open=ohlc_dummy_data['open'],
            sr_high=ohlc_dummy_data['high'],
            sr_low=ohlc_dummy_data['low'],
            col_name_prefix='ohlc_testdata'
        ).dropna()
        print(df_ohlc_feats)
        assert(isinstance(df_ohlc_feats, pd.DataFrame))
        assert(len(df_ohlc_feats)>100)

    def test_feature_creation_single_ts(self, single_ts_dummy_data):
        df_single_ts_feats = create_single_ts_features(
            sr_ts=single_ts_dummy_data,
            col_name_prefix='singlets_testdata'
        ).dropna()
        print(df_single_ts_feats)
        assert(isinstance(df_single_ts_feats, pd.DataFrame))
        assert(len(df_single_ts_feats)>100)

    def test_feature_selection_random(self, ohlc_dummy_data, single_ts_dummy_data):
        ohlc_df_dict = {f'ohlc_no{i}': ohlc_dummy_data for i in range(20)}
        single_ts_sr_dict = {f'single_ts_no{i}': single_ts_dummy_data for i in range(20)}
        df_random_feats = random_feat_select(
            ohlc_df_dict=ohlc_df_dict,
            single_ts_sr_dict=single_ts_sr_dict,
            close_col_name='close',
            open_col_name='open',
            high_col_name='high',
            low_col_name='low'
        ).dropna()
        print(df_random_feats)
