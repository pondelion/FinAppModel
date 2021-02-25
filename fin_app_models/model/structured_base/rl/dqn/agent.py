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
        self._model_type = model_type
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
        state_tensors = []
        df_states = []
        action_qs = []
        for df_state, action, reward, df_next_state, done, next_valid_actions in batch:
            action_q = reward
            if not done:
                action_q += self._discount_factor * np.nanmax(
                    self._get_q_values(df_next_state, next_valid_actions)
                )  # 現状態のQ値 = 現状態のQ値 + γ*max(次の状態でとれるアクションのQ最大値)

            df_states.append(df_state)
            action_qs.append(action_q)
            state_tensors.append(torch.Tensor(df_state.to_numpy()))

        q_values = self._predict(df_states)
        for i, (df_state, action, reward, df_next_state, done, next_valid_actions) in enumerate(batch):
            q_values[i, action.value] = action_qs[i]  # 対象の行動のQ値だけ更新する。

        state_batch = torch.stack(state_tensors, axis=0)  # (batch_size, seq_len, n_feats)
        if self._model_type == QModelType.CNN:
            state_batch = state_batch.unsqueeze(1)  # (batch_size, seq_len, n_feats) => (batch_size, channel(1), seq_len, n_feats)

        q_batch = torch.Tensor(q_values)
        self._train(
            model=self._model,
            state_batch=state_batch,
            q_batch=q_batch,
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
        df_state: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> np.array:
        if isinstance(df_state, pd.DataFrame):
            state_tensor = torch.Tensor(df_state.to_numpy()).unsqueeze(0)  # (batch_size(1), seq_len, n_feats)
            if self._model_type == QModelType.CNN:
                state_tensor = state_tensor.unsqueeze(1)  # (batch_size(1), seq_len, n_feats) => (batch_size(1), channel(1), seq_len, n_feats)
            pred = self._model(state_tensor).detach().numpy().flatten()
        elif isinstance(df_state, list) and isinstance(df_state[0], pd.DataFrame):
            state_tensor = torch.stack([torch.Tensor(ds.to_numpy()) for ds in df_state])  # (batch_size, seq_len, n_feats)
            if self._model_type == QModelType.CNN:
                state_tensor = state_tensor.unsqueeze(1)  # (batch_size, seq_len, n_feats) => (batch_size, channel(1), seq_len, n_feats)
            pred = self._model(state_tensor).detach().numpy().reshape((len(df_state), -1))
        else:
            raise Exception(f'_predict() received invalid type input {type(df_state)}')
        return pred
