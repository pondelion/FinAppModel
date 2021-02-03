from typing import List

import numpy as np
import torch.nn as nn


class Market:

    ACTIONS = (
        0,  # ノーポジ
        1,  # 買い
        2,  # 売り
        3  # 保持
    )

    def __init__(
        self,
        timeseries: np.array,
        window: int,
        model: nn.Module,
        cost: float,
        asset: int=30000,
        reward_data_idx: int=0
    ):
        self._timeseries = timeseries  # 学習に使う時系列データ。T(期間)xN(特徴量数)
        self._no_posi_flg = True  # ノーポジフラグ
        self._window = window  # 過去何個分のデータを入力データ(状態)とするか幅
        self._model = model
        self._cost = cost  # 決済コスト
        self._reward_data_idx = reward_data_idx  # ターゲットインデックス(何個目の特徴量を報酬計算の教師データとするかのインデックス)
        self._t = window
        self._asset = asset
        self._init_asset = asset

    def _get_state(self, t):
        try:
            state = self._timeseries[t-self._window:t, :]
        except Exception as e:
            raise e
        
        # 標準化 -> ?
        means = np.mean(state, axis=0)
        state = (state/means - 1.0) * 100

        return state

    def _get_valid_actions(self) -> List[int]:
        if self._no_posi_flg:
            return [Market.ACTIONS[0], Market.ACTIONS[1]]  # ノーポジor買い
        else:
            return [Market.ACTIONS[2], Market.ACTIONS[3]]  # 売りor保持

    def _get_reward(self, t: int, action: int) -> float:
        reward = None

        if action == 0:  # ノーポジ
            reward = 0
        elif action == 1:  # 買い
            reward = self._timeseries[t+1, self._reward_data_idx] - self._timeseries[t, self._reward_data_idx] - self._cost
        elif action == 2:  # 売り
            reward = -(self._timeseries[t+1, self._reward_data_idx] - self._timeseries[t, self._reward_data_idx]) - self._cost
        elif action == 3:  # 保持
            reward = self._timeseries[t+1, self._reward_data_idx] - self._timeseries[t, self._reward_data_idx]
        else:
            raise ValueError(f'Invalid action : {action}')

        return reward

    def step(self, action: int):

        reward = self._get_reward(self._t, action)

        if (action == Market.ACTIONS[0]) or (action == Market.ACTIONS[2]):
            self._no_posi_flg = True
        else:
            self._no_posi_flg = False

        if action == Market.ACTIONS[1]:  # 買い
            self._asset -= (self._timeseries[self._t, self._reward_data_idx] + self._cost)
        elif action == Market.ACTIONS[2]:  # 売り
            self._asset += (self._timeseries[self._t, self._reward_data_idx] - self._cost)

        self._t += 1

        # アクションを取った後(タイムステップを1進めた)の状態を取得
        state = self._get_state(self._t)

        done = self._timeseries.shape[0] == (self._t + 1)

        return state, reward, done, self._get_valid_actions(), self._asset

    def reset(self):
        self._no_posi_flg = True
        self._t = self._window
        self._asset = self._init_asset
