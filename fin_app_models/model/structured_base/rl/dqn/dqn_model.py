from typing import Dict, Union
import copy

import pandas as pd

from .market import Market
from .agent import Agent
from .model.models import QModelType


class DQNModel:

    def train(
        self,
        df_X_train: pd.DataFrame,
        sr_y_train: pd.Series,
        n_episodes: int,
        state_window: int,
        model_type: QModelType = QModelType.CNN,
        model_params: Dict[str, Union[str, int, float]] = {},
        cost: int = 0,
        asset: int = 1000000,
        batch_size: int = 32,
        discount_factor: float = 0.95,
        eps: float = 0.05,
    ):
        model_params['n_feats'] = len(df_X_train.columns)
        model_params['n_actions'] = 4
        self._market = Market(
            df_X_train=df_X_train,
            sr_y_train=sr_y_train,
            window=state_window,
            cost=cost,
            asset=asset
        )
        self._agent = Agent(
            model_type=model_type,
            model_params=model_params,
            batch_size=batch_size,
            discount_factor=discount_factor,
            eps=eps,
        )

        self._histories = {}

        for episode in range(n_episodes):
            
            t_idx = state_window
            
            rewards = []
            actions = []
            states = []
            assets = []
            done = False
            
            df_state, valid_actions = self._market.reset()
            print('episode : ', episode)

            while not done:
                if t_idx % 100 == 0:
                    print('t_idx : ', t_idx)

                # decide next action from current state and valid actions.
                action = self._agent.act(df_state, valid_actions)

                # take action and move to next state.
                df_next_state, reward, done, valid_actions, asset = self._market.step(action)

                rewards.append(reward)
                actions.append(action)
                states.append(df_next_state)
                assets.append(asset)

                # save state and execute training
                self._agent.remember(df_state, action, reward, df_next_state, done, valid_actions)
                self._agent.agent.replay()

                df_state = df_next_state

                t_idx += 1

            self._histories[f'episode{episode}'] = {}
            self._histories[f'episode{episode}']['actions'] = actions
            self._histories[f'episode{episode}']['rewards'] = rewards
            self._histories[f'episode{episode}']['assets'] = assets

        return copy.deepcopy(self._histories)