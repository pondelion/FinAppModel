from datetime import datetime, timedelta
from functools import reduce
from typing import Union, Tuple, List, Dict, Tuple

import pandas as pd
import numpy as np
from overrides import overrides
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from fastprogress import progress_bar as pb

from ...base_model import BaseTimeseriesModel
from .....processing import (
    IStructuredDataProcessing,
    LSTMDataProcessing,
)
from .....param_tuning import (
    IParamTuber,
    DefaultTuner
)
from .....utils.logger import Logger


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
        data_processors: List[IStructuredDataProcessing] = [LSTMDataProcessing()],
        param_tuner: IParamTuber = DefaultTuner(),
    ):
        super(BILSTMRegression, self).__init__(data_processors, param_tuner)
        self._x_mms = MinMaxScaler((-0.5, 0.5))
        self._y_mms = MinMaxScaler((-0.5, 0.5))
        self._model = None

    @overrides
    def _train(
        self,
        y_train: Union[pd.Series, np.ndarray],
        X_train: Union[Union[pd.DataFrame, pd.Series], np.ndarray] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs,
    ) -> None:

        if dt_now is None and (isinstance(y_train, pd.Series)):
            dt_now = y_train.index.max()
        self._dt_now = dt_now

        try:
            self._seq_len = kwargs['seq_len']
        except Exception:
            raise Exception('Paramerer [seq_len] must be specified.')
        n_epoch = kwargs.get('n_epoch', 100)
        try:
            n_epoch = kwargs['n_epoch']
        except Exception:
            n_epoch = 30
            Logger.w(self.__class__, f'Parameter [n_epoch] not specified, using default n_epoch : 30')
        try:
            self._hidden_dim = kwargs['hidden_dim']
        except Exception:
            self._hidden_dim = 32
            Logger.w(self.__class__, f'Parameter [hidden_dim] not specified, using default hidden_dim : 32')
        try:
            self._num_layers = kwargs['num_layers']
        except Exception:
            self._num_layers = 2
            Logger.w(self.__class__, f'Parameter [num_layers] not specified, using default num_layers : 2')
        try:
            lr = kwargs['lr']
        except Exception:
            lr = 0.005
            Logger.w(self.__class__, f'Parameter [lr] not specified, using default lr : 0.005')
        try:
            batch_size = kwargs['batch_size']
        except Exception:
            batch_size = 16
            Logger.w(self.__class__, f'Parameter [batch_size] not specified, using default batch_size : 16')

        if not isinstance(y_train, np.ndarray):
            y_train = y_train.to_numpy()
        if not isinstance(X_train, np.ndarray):
            X_train = X_train.to_numpy()
        self._y_mms.fit(y_train.reshape(-1, 1))
        xs_train, ys_train = self._create_xy_dataset(X_train, y_train, self._seq_len)

        self._model = _BILSTM(
            input_dim=X_train.shape[1],
            hidden_dim=self._hidden_dim,
            num_layers=self._num_layers,
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
        y: Union[pd.Series, np.ndarray],
        X: Union[Union[pd.DataFrame, pd.Series], np.ndarray] = None,
        pred_periods: int = 1,
        **kwargs,
    ) -> np.ndarray:
        if pred_periods > 1:
            Logger.w(
                self.__class__.__name__,
                'Multivariable LSTM model does not support forcast for pred_periods > 1, setting pred_periods = 1'
            )
        if self._model is None:
            raise Exception('Model is not yet fit.')
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        xs, ys = self._create_xy_dataset(X, y, self._seq_len)
        self._model.eval()
        return self._model(xs).cpu().detach().numpy()

    def _create_xy_dataset(
        self,
        X: np.ndarray,
        sr_y: np.ndarray,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for t in range(len(X)-seq_len):
            xs.append(self._preprocess_input_x_sequence(X[t:t+seq_len, :], seq_len))
            if sr_y is not None:
                ys.append(self._preprocess_input_y_sequence(sr_y[t:t+seq_len], seq_len))
        ys = torch.stack(ys, dim=0) if len(ys) > 0 else None
        xs = torch.stack(xs, dim=0)
        return xs, ys

    def _preprocess_input_x_sequence(self, x_seq: np.ndarray, seq_len: int) -> torch.Tensor:
        assert x_seq.ndim == 2
        x = self._x_mms.fit_transform(x_seq.reshape(-1, x_seq.shape[1])).reshape(seq_len, x_seq.shape[1])
        x = torch.Tensor(x).view(seq_len, x_seq.shape[1])
        return x

    def _preprocess_input_y_sequence(self, y_seq: pd.DataFrame, seq_len: int):
        y = self._y_mms.transform(y_seq.reshape(-1, 1)).flatten()
        y = torch.Tensor(y).view(seq_len, 1)
        return y


class BILSTMMultiTimescaleRegression(BaseTimeseriesModel):

    def __init__(
        self,
        data_processors: List[IStructuredDataProcessing] = [LSTMDataProcessing()],
        param_tuner: IParamTuber = DefaultTuner(),
    ):
        super(BILSTMMultiTimescaleRegression, self).__init__(data_processors, param_tuner)
        self._x_mms = MinMaxScaler((-0.5, 0.5))
        self._y_mms = MinMaxScaler((-0.5, 0.5))
        self._models = None

    @overrides
    def _train(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
        model_params: Dict = {},
        **kwargs,
    ) -> None:

        if dt_now is None and (isinstance(y_train, pd.Series)):
            dt_now = y_train.index.max()
        self._dt_now = dt_now

        try:
            time_window_sizes = kwargs['time_window_sizes']
        except Exception:
            time_window_sizes = [7, 20, 30, 60]
            Logger.w(self.__class__, f'Parameter [time_window_sizes] not specified, using default time_window_sizes : {time_window_sizes}')
        try:
            n_epoch = kwargs['n_epoch']
        except Exception:
            n_epoch = 30
            Logger.w(self.__class__, f'Parameter [n_epoch] not specified, using default n_epoch : 30')
        try:
            hidden_dim = kwargs['hidden_dim']
        except Exception:
            hidden_dim = 32
            Logger.w(self.__class__, f'Parameter [hidden_dim] not specified, using default hidden_dim : 32')
        try:
            num_layers = kwargs['num_layers']
        except Exception:
            num_layers = 2
            Logger.w(self.__class__, f'Parameter [num_layers] not specified, using default num_layers : 2')
        try:
            lr = kwargs['lr']
        except Exception:
            lr = 0.005
            Logger.w(self.__class__, f'Parameter [lr] not specified, using default lr : 0.005')
        try:
            batch_size = kwargs['batch_size']
        except Exception:
            batch_size = 16
            Logger.w(self.__class__, f'Parameter [batch_size] not specified, using default batch_size : 16')
        try:
            lstm_output_dim = kwargs['lstm_output_dim']
        except Exception:
            lstm_output_dim = 8
            Logger.w(self.__class__, f'Parameter [lstm_output_dim] not specified, using default lstm_output_dim : 8')


        self._y_mms.fit(y_train.to_numpy().reshape(-1, 1))
        xs_train_list = []
        ys_train_list = []
        for seq_len in time_window_sizes:
            xs_train, ys_train = self._create_xy_dataset(X_train, y_train, seq_len)
            xs_train_list.append(xs_train)
            ys_train_list.append(ys_train)

        min_xs_len = min([xs_train.shape[0] for xs_train in xs_train_list])
        min_ys_len = min([ys_train.shape[0] for ys_train in ys_train_list])

        for i, (xs_train, ys_train) in enumerate(zip(xs_train_list, ys_train_list)):
            xs_train_list[i] = xs_train[-min_xs_len:, :, :]
            ys_train_list[i] = ys_train[-min_ys_len:, :, :]

        self._models = []
        for _ in range(len(time_window_sizes)):
            self._models.append(
                _BILSTM(
                    input_dim=len(X_train.columns),
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=lstm_output_dim,
                )
            )
        self._linear = nn.Linear(
            len(time_window_sizes)*lstm_output_dim,
            1
        )  # final output fc layer.

        params = list(reduce(lambda x, y: x+y, [list(model.parameters()) for model in self._models]))
        params += self._linear.parameters()
        mse_loss = nn.MSELoss()
        optim = torch.optim.Adam(
            params=params,
            lr=lr
        )

        loss_history = []

        for n in pb(range(n_epoch)):
            losses = []
            for t in range(xs_train_list[0].size(0) // batch_size):
                outs = []
                for xs_train, model in zip(xs_train_list, self._models):
                    x = xs_train[t*batch_size:(t+1)*batch_size, :, :]
                    out = model(x)[:, -min(time_window_sizes):, :]  # (batch_size, min(time_window_sizes), lstm_output_dim)
                    outs.append(out)
                outs = torch.cat(outs, axis=1)  # list(batch_size, min(time_window_sizes), lstm_output_dim) => (batch_size, len(time_window_sizes)*min(time_window_sizes), lstm_output_dim)
                pred = self._linear(outs.view(outs.shape[0], min(time_window_sizes), -1))  # (batch_size, min(time_window_sizes), len(time_window_sizes)*lstm_output_dim) => (batch_size, min(time_window_sizes), 1)
                y = ys_train_list[0][t*batch_size:(t+1)*batch_size, -min(time_window_sizes):, :]  # (batch_size, min(time_window_sizes), 1)
                loss = mse_loss(pred, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())

            loss_history.append(sum(losses) / len(losses))

    @overrides
    def _predict(
        self,
        y: Union[pd.Series, np.ndarray],
        X: Union[Union[pd.DataFrame, pd.Series], np.ndarray] = None,
        pred_periods: int = 1,
        **kwargs,
    ) -> np.ndarray:
        if pred_periods > 1:
            Logger.w(
                self.__class__.__name__,
                'Multivariable LSTM model does not support forcast for pred_periods > 1, setting pred_periods = 1'
            )
        if self._models is None:
            raise Exception('Model is not yet fit.')
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        xs, ys = self._create_xy_dataset(X, y, self._seq_len)
        self._model.eval()
        return self._model(xs).cpu().detach().numpy()

    def _create_xy_dataset(
        self,
        X: np.ndarray,
        sr_y: np.ndarray,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for t in range(len(X)-seq_len):
            xs.append(self._preprocess_input_x_sequence(X[t:t+seq_len, :], seq_len))
            if sr_y is not None:
                ys.append(self._preprocess_input_y_sequence(sr_y[t:t+seq_len], seq_len))
        ys = torch.stack(ys, dim=0) if len(ys) > 0 else None
        xs = torch.stack(xs, dim=0)
        return xs, ys

    def _preprocess_input_x_sequence(self, x_seq: np.ndarray, seq_len: int) -> torch.Tensor:
        assert x_seq.ndim == 2
        x = self._x_mms.fit_transform(x_seq.reshape(-1, x_seq.shape[1])).reshape(seq_len, x_seq.shape[1])
        x = torch.Tensor(x).view(seq_len, x_seq.shape[1])
        return x

    def _preprocess_input_y_sequence(self, y_seq: pd.DataFrame, seq_len: int):
        y = self._y_mms.transform(y_seq.reshape(-1, 1)).flatten()
        y = torch.Tensor(y).view(seq_len, 1)
        return y