from typing import List, Union
from functools import reduce

import pandas as pd
import torch
import torch.nn as nn 
from sklearn.preprocessing import MinMaxScaler
from fastprogress import progress_bar as pb

from .trader import Trader
from .models.models import TraderModel


class SimpleTraderMarket:
    """All trader has same characteristics.
    (Same prediction model type and make trading decision from same input data.)
    """

    def __init__(
        self,
        n_traders: int = 100,
        device: str = 'cpu',
        trader_model_type: TraderModel = TraderModel.BILSTM,
        **kwargs,
    ):
        model_params = {}
        if trader_model_type == TraderModel.BILSTM:
            hidden_dim = kwargs.get('hidden_dim', 32)
            num_layers = kwargs.get('num_layers', 2)
            model_params['hidden_dim'] = hidden_dim
            model_params['num_layers'] = num_layers

        self._n_traders = n_traders
        self._traders = [
            Trader(
                model_type=TraderModel.BILSTM,
                model_params=model_params,
                action_thresh=0.3,
                device=device
            ) for _ in range(n_traders)
        ]
        self._device = device
        self._x_batch_mms = MinMaxScaler((-0.8, 0.8))
        self._y_mms = MinMaxScaler((-3, 3))

    def train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        **kwargs,
    ) -> None:
        self._y_mms.fit(y_train.to_numpy().reshape(-1, 1))

        try:
            self._seq_len = kwargs['seq_len']
        except Exception:
            raise Exception('Paramerer [seq_len] must be specified.')
        n_epoch = kwargs.get('n_epoch', 100)
        lr = kwargs.get('lr', 0.005)
        batch_size = kwargs.get('batch_size', 16)

        xs_train, ys_train = self._create_xy_dataset(X_train, y_train, self._seq_len)

        params = list(reduce(lambda x, y: x+y, [list(trader.parameters()) for trader in self._traders]))
        mse_loss = nn.MSELoss()
        optim = torch.optim.Adam(
            params=params,
            lr=lr
        )

        [trader.train() for trader in self._traders]
        loss_history = []

        for n in pb(range(n_epoch)):
            losses = []
            for t in range(xs_train.size(0) // batch_size):
                x = xs_train[t*batch_size:(t+1)*batch_size, :, :]
                y = ys_train[t*batch_size:(t+1)*batch_size, :, :]

                preds = [trader.predict(x) for trader in self._traders]
                preds = torch.stack(preds, -1)  # (batch_size, n_tranders)
                # Market trader's average decision (1 -> buy = y(stock price) rises, -1 -> sell = y(stock price) drops)
                preds = preds.mean(axis=-1)  # (batch_size, n_tranders) => (batch_size)
                loss = mse_loss(preds, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())

            loss_history.append(sum(losses) / len(losses))

    def predict(
        self,
        df_state: pd.DataFrame
    ) -> float:
        if df_state.shape(0) != self._seq_len:
            raise Exception(f'Expected time window size {self._seq_len}, but received {df_state.shape(0)}')
        x = self._preprocess_x_sequence(
            df_x_seq=df_state,
            seq_len=self._seq_len
        )
        [trader.eval() for trader in self._traders]
        preds = [trader.predict(x) for trader in self._traders]
        market_trend = torch.stack(preds).mean().cpu().detach.numpy()
        # market_trend -> 1 means trader's 
        return market_trend

    def _create_xy_dataset(
        self,
        df_X: pd.DataFrame,
        sr_y: pd.Series,
        seq_len: int
    ) -> List[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for t in range(len(df_X)-seq_len):
            xs.append(self._preprocess_x_sequence(df_X.iloc[t:t+seq_len, :], seq_len))
            ys.append(self._preprocess_y_sequence(sr_y.iloc[t:t+seq_len], seq_len))
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def _preprocess_x_sequence(self, df_x_seq: pd.DataFrame, seq_len: int) -> torch.Tensor:
        x = self._x_batch_mms.fit_transform(df_x_seq.to_numpy().reshape(-1, len(df_x_seq.columns))).reshape(seq_len, len(df_x_seq.columns))
        x = torch.Tensor(x).view(seq_len, len(df_x_seq.columns))
        return x

    def _preprocess_y_sequence(self, sr_y: pd.Series, seq_len: int) -> torch.Tensor:
        y = self._y_mms.transform(sr_y.to_numpy().reshape(-1, 1)).flatten()
        y = torch.Tensor(y).view(seq_len, 1)
        y = torch.Tanh()(y)
        return y
