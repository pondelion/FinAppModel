from typing import Union

import pandas as pd

class TimeseriesData:

    def __init__(
        self,
        ts: Union[pd.DataFrame, pd.Series],
        do_daily_interpolation: bool = True,
        interpolate_method: str = 'pad'
    ):
        if self._validate_index_dtype(ts) is False:
            raise Exception('Only datetime index DataFrame/Series are supported')

        if isinstance(ts, pd.DataFrame):
            pd_ts = pd.DataFrame
        elif isinstance(ts, pd.Series):
            pd_ts = pd.Series
        else:
            raise Exception('ts datatype error')

        self._ts = ts.asfreq('D')

        if do_daily_interpolation:
            self._ts = self._daily_interpolate(
                self._ts, interpolate_method
            )

    def _daily_interpolate(
        self,
        ts: Union[pd.DataFrame, pd.Series],
        interpolate_method: str,
    ) -> Union[pd.DataFrame, pd.Series]:
        return ts.interpolate(method=interpolate_method)

    def _validate_index_dtype(self, ts: Union[pd.DataFrame, pd.Series]) -> bool:
        try:
            pd.date_range(ts.index.min(), ts.index.max())
            return True
        except Exception:
            return False

    def __call__(self) -> Union[pd.DataFrame, pd.Series]:
        return self._ts
