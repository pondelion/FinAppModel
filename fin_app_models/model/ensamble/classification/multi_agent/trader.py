from typing import Dict, Union
from enum import Enum
from typing import List, Dict, Union

import pandas as pd
import torch

from .models.models import TraderModel


class TraderAction(Enum):
    BUY = -1
    SELL = 1
    DO_NOTHING = 0


class Trader:

    def __init__(
        self,
        model_type: TraderModel = TraderModel.BILSTM,
        model_params: Dict[str, Union[str, int, float]] = {},
        action_thresh: float = 0.3,
        device: str = 'cpu',
    ):
        self._model_type = model_type
        self._model = model_type.value(**model_params)
        self._action_thresh = action_thresh
        self._device = device

    def predict(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self._model(state.to(self._device))

    def make_decision(
        self,
        state: torch.Tensor,
    ) -> Union[TraderAction, List[TraderAction]]:
        preds = self._model(state.to(self._device)).cpu().detach().numpy()  # (batch_size)
        action = [self._pred2action(pred) for pred in preds]
        return action

    def _pred2action(self, pred_value: float):
        if pred_value <= self._action_thresh:
            return TraderAction.SELL
        elif pred_value < self._action_thresh:
            return TraderAction.DO_NOTHING
        else:
            return TraderAction.BUY
