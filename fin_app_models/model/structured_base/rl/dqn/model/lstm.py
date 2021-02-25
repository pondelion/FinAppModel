import torch.nn as nn


class BILSTM(nn.Module):

    def __init__(
        self,
        n_feats: int,
        timeseries_len: int,
        n_actions: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ):
        super(BILSTM, self).__init__()
        self._input_dim = n_feats
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._output_dim = n_actions

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
        return self._linear(out[:, -1, :])
