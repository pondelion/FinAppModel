import random
from typing import List

import numpy as np
import torch.nn as nn


class Agent:

    def __init__(
        self,
        model: nn.Module,
        batch_size: int=32,
        discount_factor: float=0.95,
        eps: float=0.05
    ):
        self._model = model
        self._memory = []
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._eps = eps

    def replay(self):
        """過去の状態・行動等の履歴からバッチサイズ分ランダムサンプリングし、
        各データのQ値を更新し、入力を状態(時系列スライスデータ)、出力を
        更新したQ値でモデルを学習する
        """
        batch = random.sample(
            self._memory,
            min(len(self._memory), self._batch_size)
        )
        for state, action, reward, next_state, done, next_valid_actions in batch:
            q = reward
            if not done:
                q += self._discount_factor * np.nanmax(
                    self._get_q_values(next_state, next_valid_actions)
                )  # 現状態のQ値 = 現状態のQ値 + γ*max(次の状態でとれるアクションのQ最大値)
            self._model.train(
                X=state,
                y=q,
                action=action
            )  # 現状態を入力、出力を更新したQ値としてモデルを学習

    def _get_q_values(
        self,
        state,
        next_valid_actions
    ) -> List[float]:
        """指定した状態をモデルに入力してQ値を予測し、
        有効アクションに対するQ値リストを返す。
        非有効なアクションに対するQ値はnp.nan。
        """
        q = self._model.predict(state)  # 全アクションのQ値
        q_valid = [np.nan] * len(q)  # len(q) -> 全アクション数
        for action in next_valid_actions:
            q_valid[action] = q[action]
        return q_valid

    def remenber(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_valid_actions
    ) -> None:
        self._memory.append(
            (state, action, reward, next_state, done, next_valid_actions)
        )

    def act(
        self,
        state,
        valid_actions
    ) -> int:
        """状態と可能なアクションから次にとるアクションを決める"""
        action = None
        if np.random.random() > self._eps:
            q = self._get_q_values(state, valid_actions)
            if np.nanmin(q) != np.nanmax(q):
                action = np.nanargmax(q)  # 最もQ値が大きいアクションをとる
        else:  # epsの確率でランダムなアクションをとる
            action = random.sample(valid_actions, 1)[0]

        return action