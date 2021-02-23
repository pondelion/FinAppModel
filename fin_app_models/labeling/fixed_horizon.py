from typing import Union

import pandas as pd


def return_days(ts: Union[pd.Series, pd.DataFrame], days: int = 1):
    return (ts.shift(-days) - ts) / ts
