from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, List, Dict
from datetime import datetime

import pandas as pd


class IParamTuber(metaclass=ABCMeta):

    @abstractmethod
    def param_tuning(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
    ) -> Dict[str, Union[float, str, int]]:
        raise NotImplementedError
