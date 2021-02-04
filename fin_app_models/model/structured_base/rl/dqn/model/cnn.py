from typing import List
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(
        self,
        n_feats: int,
        n_actions: int,
        cnn_out_channels: List[int] = [16, 16],
        cnn_time_kernel_sizes: List[int] = [5, 5],
        fc_out_dims: List[int] = [48, 24],
        activation_cls = nn.ReLU,
        use_pool: bool = True,
        pool_kernel_size: int = 2,
    ):
        super(CNN, self).__init__()
        self._n_feats = n_feats
        self._n_actions = n_actions
        self._in_channel = 1
        self._cnn_out_channels = cnn_out_channels
        self._cnn_in_channels = [self._in_channel] + self._cnn_out_channels[:-1]
        self._cnn_layers_num = len(cnn_out_channels)
        self._cnn_time_kernel_sizes = cnn_time_kernel_sizes
        self._cnn_kernel_sizes = [(tks, n_feats) for tks in cnn_time_kernel_sizes]
        self._fc_out_dims = fc_out_dims
        self._fc_layers_num = len(fc_out_dims)
        self._activation_cls = activation_cls
        self._use_pool = use_pool
        self._pool_kernel_size = pool_kernel_size

        self._layers = nn.ModuleDict({
            'cnn_layers': self._create_cnn_layers(),
            'fc_layers':  self._create_fc_layers(),
        })

    def forward(self, x):
        x = self._layers['cnn_layers'](x)
        x = x.view(x.shape[0], -1)
        x = self._layers['fc_layers'](x)
        return x

    def _create_cnn_layers(self) -> nn.Sequential:
        layers = []
        for in_ch, out_ch, kernel_size in zip(self._cnn_in_channels, self._cnn_out_channels, self._cnn_kernel_sizes):
            layers.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=(0, 1)))
            layers.append(self._activation_cls())
            if self._use_pool:
                layers.append(nn.MaxPool2d(kernel_size=self._pool_kernel_size, padding=1))
        return nn.Sequential(*layers)

    def _create_fc_layers(self) -> nn.Sequential:
        layers = []
        in_dims = self._get_fc_indims()
        for in_dim, out_dim in zip(in_dims, self._fc_out_dims):
            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            layers.append(self._activation_cls())
        layers.append(nn.Linear(in_features=self._fc_out_dims[-1], out_features=self._n_actions))
        return nn.Sequential(*layers)

    def _get_fc_indims(self) -> List[int]:
        window_size = 100
        dummy_data = torch.randn(5, self._in_channel, window_size, self._n_feats)
        cnn_out = self._create_cnn_layers()(dummy_data)
        fc_in_dim = cnn_out.view((cnn_out.shape[0], -1)).shape[1]
        fc_in_dims = [fc_in_dim] + self._fc_out_dims[:-1]
        return fc_in_dims
