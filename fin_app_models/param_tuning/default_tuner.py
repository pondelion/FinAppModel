from typing import Tuple, Union, List, Dict
from datetime import datetime

import pandas as pd
from overrides import overrides

from .base_tuner import IParamTuber


class DefaultTuner(IParamTuber):

    @overrides
    def param_tuning(
        self,
        y_train: pd.Series,
        X_train: Union[pd.DataFrame, pd.Series] = None,
        dt_now: datetime = None,
    ) -> Dict[str, Union[float, str, int]]:
        return {}
