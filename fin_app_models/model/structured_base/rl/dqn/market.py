from typing import List

import numpy as np
import torch.nn as nn
import pandas as pd

from .actions import Action, Position


class Market:

    def __init__(
        self,
        df_X_train: pd.DataFrame,
        sr_y_train: pd.Series,
        window: int,
        cost: float,
        asset: int=30000,
    ):
        if not self._validate_xy_data(df_X_train, sr_y_train):
            raise Exception("df_X_train and sr_y_train's date index must be same.")
        self._df_X_train = df_X_train
        self._sr_y_train = sr_y_train
        self._position = Position.NO_POSI
        self._window = window  # 過去何個分のデータを入力データ(状態)とするか幅
        self._cost = cost  # 決済コスト
        self._t = window
        self._asset = asset
        self._init_asset = asset

    def _get_state(self, t):
        try:
            df_state = self._df_X_train.iloc[t-self._window:t, :]
        except Exception as e:
            raise e
        
        # 標準化 -> ?
        means = self._df_X_train.mean(axis=0)
        df_state = (df_state.div(means) - 1.0) * 100

        return df_state

    def get_valid_actions(self, position: Position) -> List[Action]:
        if position == Position.NO_POSI:
            return [Action.NO_POSI, Action.BUY]
        else:
            return [Action.SELL, Action.HOLD]

    def _get_reward(self, t: int, action: Action) -> float:
        reward = None

        if action == Action.NO_POSI:
            reward = 0
        elif action == Action.BUY:
            reward = self._sr_y_train[t+1] - self._sr_y_train[t] - self._cost
        elif action == Action.SELL:
            reward = -(self._sr_y_train[t+1] - self._sr_y_train[t]) - self._cost
        elif action == Action.HOLD:
            reward = self._sr_y_train[t+1] - self._sr_y_train[t] - self._cost
        else:
            raise ValueError(f'Invalid action : {action}')

        return reward

    def step(self, action: Action):

        reward = self._get_reward(self._t, action)

        if (action == Action.NO_POSI) or (action ==Action.SELL):
            self._position = Position.NO_POSI
        else:
            self._position = Position.HOLD

        if action == Action.BUY:
            self._asset -= (self._sr_y_train[self._t] + self._cost)
        elif action == Action.SELL:
            self._asset += (self._sr_y_train[self._t]  - self._cost)

        self._t += 1

        # アクションを取った後(タイムステップを1進めた)の状態を取得
        df_state = self._get_state(self._t)

        done = len(self._sr_y_train) == (self._t + 1)

        return df_state, reward, done, self.get_valid_actions(self._position), self._asset

    def reset(self):
        self._position = Position.NO_POSI
        self._t = self._window
        self._asset = self._init_asset
        df_state = self._get_state(self._t)
        valid_actions = self.get_valid_actions(self._position)
        return df_state, valid_actions

    def _validate_xy_data(
        self, df_X: pd.DataFrame, sr_y: pd.Series,
    ) -> bool:
        if len(df_X) != len(sr_y):
            return False
        if df_X.index.min() != sr_y.index.min():
            return False
        if df_X.index.max() != sr_y.index.max():
            return False
        return True
