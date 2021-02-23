from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Union
from enum import Enum


class ModelType(Enum):

    Regression = 'regression'
    Classification = 'classification'
    Timeseries = 'timeseries'
    Reinforcement = 'reinforcement'


class PredTarget(Enum):

    Stock = 'stock'
    Other = 'other'


@dataclass
class PredictionResult:

    model_name: str
    model_type: ModelType
    pred_target_type: PredTarget
    stock_code: int
    train_dt: datetime 
    now_dt: datetime
    pred_target_dt: datetime
    pred_value: Union[int, float]
    model_params: Dict[str, Union[str, int, float]]
    feature_names: List[str]
    actual_value: int = None

    def json(self):
        return {
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'pred_target_type': self.pred_target_type.value,
            'stock_code': self.stock_code,
            'train_dt': self.train_dt,
            'now_dt': self.now_dt,
            'pred_target_dt': self.pred_target_dt,
            'pred_value': self.pred_value,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'actual_value': self.actual_value
        }
