from datetime import datetime, timedelta
from typing import Union, Tuple, List, Dict

import pandas as pd
import numpy as np
from overrides import overrides
import torch
import torch.nn as nn
from sklearn.preprocess import MinMaxScaler
from fastprogress import progress_bar as pb

from ..base_model import BaseTimeseriesModel
from ....processing import (
    IStructuredDataProcessing,
    LSTMRegressionDataProcessing,
)
from ....param_tuning import (
    IParamTuber,
    DefaultTuner
)


class _BILSTM(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super(_BILSTM, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._output_dim = output_dim

        self._lstm = nn.LSTM(
            input_size=self._input_dim,
            hidden_size=self._hidden_dim,
            num_layers=self._num_layers,
            batch_first=True,
            bidirectional=True
        )

        self._linear = nn.Linear(
            self._hidden_dim*2,
            self._output_dim
        )

    def forward(self, x):
        self._lstm.flatten_parameters()
        out, hidden = self._lstm(x)
        return self._linear(out[:, :, :])


class BILSTMRegression(BaseTimeseriesModel):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [LSTMRegressionDataProcessing()],
        param_tuner: IParamTuber = DefaultTuner(),
    ):
        super(BILSTMRegression, self).__init__(data_processors, param_tuner)
        self._x_mms = MinMaxScaler((-0.5, 0.5))
        self._y_mms = MinMaxScaler((-0.5, 0.5))

    @overrides
    def _train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs,
    ) -> None:

        if dt_now is None:
            dt_now = y_train.index.max()
        self._dt_now = dt_now

        try:
            seq_len = kwargs['seq_len']
        except Exception:
            raise Exception('Paramerer [seq_len] must be specified.')
        n_epoch = kwargs.get('n_epoch', 100)
        hidden_dim = kwargs.get('n_epoch', 32)
        num_layers = kwargs.get('n_epoch', 2)
        lr = kwargs.get('lr', 0.005)
        batch_size = kwargs.get('batch_size', 16)

        xs_train, ys_train = self._create_xy_dataset(X_train, y_train, seq_len)

        self._model = _BILSTM(
            input_dim=len(X_train.columns),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1
        )

        mse_loss = nn.MSELoss()
        optim = torch.optim.Adam(
            params=self._model.parameters(),
            lr=lr
        )

        loss_history = []

        for n in pb(range(n_epoch)):
            losses = []
            for t in range(xs_train.size(0) // batch_size):
                x = xs_train[t*batch_size:(t+1)*batch_size, :, :]
                y = ys_train[t*batch_size:(t+1)*batch_size, :, :]

                out = self._model(x)
                loss = mse_loss(out, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())

            loss_history.append(sum(losses) / len(losses))

    @overrides
    def _predict(
        self,
        y: pd.Series = None,
        X: Union[pd.DataFrame, pd.Series] = None,
        pred_days: int = 30,
        **kwargs,
    ) -> pd.Series:
        raise NotImplementedError

    def _create_xy_dataset(
        self,
        df_X: pd.DataFrame,
        sr_y: pd.Series,
        seq_len: int
    ) -> List[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for t in range(len(df_X)-seq_len):
            xs.append(self._preprocess_input_x_sequence(df_X.iloc[t:t+seq_len, :], seq_len))
            ys.append(self._preprocess_input_y_sequence(sr_y.iloc[t:t+seq_len], seq_len))
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def _preprocess_input_x_sequence(self, df_x_seq: pd.DataFrame, seq_len: int) -> torch.Tensor:
        x = self._x_mms.fit_transform(df_x_seq.to_numpy().reshape(-1, len(df_x_seq.columns))).reshape(seq_len, len(df_x_seq.columns))
        x = torch.Tensor(x).view(seq_len, len(df_x_seq.columns))
        return x

    def _preprocess_input_y_sequence(self, sr_y_seq: pd.DataFrame, seq_len: int):
        y = self._y_mms.fit_transform(sr_y_seq.to_numpy().reshape(-1, 1)).flatten()
        y = torch.Tensor(y).view(seq_len, 1)
        return y
