import random
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .actions import Action
from .model.models import QModelType


class Agent:

    def __init__(
        self,
        model_type: QModelType,
        model_params: Dict[str, Union[str, int, float]] = {},
        batch_size: int = 32,
        discount_factor: float = 0.95,
        eps: float = 0.05,
        lr: float = 0.005,
    ):
        self._model = model_type.value(**model_params)
        self._memory = []
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._eps = eps
        self._mse_loss = nn.MSELoss()
        self._optim = torch.optim.Adam(
            params=self._model.parameters(),
            lr=lr
        )

    def replay(self):
        """過去の状態・行動等の履歴からバッチサイズ分ランダムサンプリングし、
        各データのQ値を更新し、入力を状態(時系列スライスデータ)、出力を
        更新したQ値でモデルを学習する
        """
        batch = random.sample(
            self._memory,
            min(len(self._memory), self._batch_size)
        )
        q_tensors = []
        state_tensors = []
        for df_state, action, reward, df_next_state, done, next_valid_actions in batch:
            q = reward
            if not done:
                q += self._discount_factor * np.nanmax(
                    self._get_q_values(df_next_state, next_valid_actions)
                )  # 現状態のQ値 = 現状態のQ値 + γ*max(次の状態でとれるアクションのQ最大値)

            q_tensors.append(torch.Tensor(q))
            state_tensors.append(torch.Tensor(df_state.to_numpy()))

        self._train(
            model=self._model,
            state_batch=torch.stack(state_tensors, axis=0).unsqueeze(1),
            q_batch=torch.stack(q_tensors, axis=0),
            action=action
        )  # 現状態を入力、出力を更新したQ値としてモデルを学習

    def _get_q_values(
        self,
        df_state: pd.DataFrame,
        next_valid_actions: List[Action],
    ) -> List[float]:
        """指定した状態をモデルに入力してQ値を予測し、
        有効アクションに対するQ値リストを返す。
        非有効なアクションに対するQ値はnp.nan。
        """
        # q = self._model.predict(df_state)  # 全アクションのQ値
        q = self._predict(
            mdoel=self._model,
            df_state=df_state
        )
        q_valid = [np.nan] * len(q)  # len(q) -> 全アクション数
        for action in next_valid_actions:
            q_valid[action.value] = q[action.value]
        return q_valid

    def remember(
        self,
        df_state: pd.DataFrame,
        action: Action,
        reward: float,
        df_next_state: pd.DataFrame,
        done: bool,
        next_valid_actions: List[Action]
    ) -> None:
        self._memory.append(
            (df_state, action, reward, df_next_state, done, next_valid_actions)
        )

    def act(
        self,
        df_state: pd.DataFrame,
        valid_actions: List[Action],
    ) -> int:
        """状態と可能なアクションから次にとるアクションを決める"""
        action = None
        if np.random.random() > self._eps:
            q = self._get_q_values(df_state, valid_actions)
            if np.nanmin(q) != np.nanmax(q):
                action_idx = np.nanargmax(q)  # 最もQ値が大きいアクションをとる
                action = Action.idx2action(action_idx)
        else:  # epsの確率でランダムなアクションをとる
            action = random.sample(valid_actions, 1)[0]

        return action

    def _train(
        self,
        model: nn.Module,
        state_batch: torch.Tensor,
        q_batch: torch.Tensor,
        action: Action,
    ) -> None:
        out = self._model(state_batch)
        loss = self._mse_loss(out, q_batch)

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def _predict(
        self,
        mdoel: nn.Module,
        df_state: pd.DataFrame
    ) -> np.array:
        state_tensor = torch.Tensor(df_state.to_numpy())
        pred = self._model(state_tensor.unsqueeze(0).unsqueeze(0)).detach().numpy().flatten()
        return pred
