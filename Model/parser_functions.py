import pandas as pd
import datetime


def df_time_interpolate(df: pd.DataFrame, time_interval_sec: int):
    opts = dict(closed="left", label="left")
    time_interval_str = f"{time_interval_sec}S"
    return resample_time_weighted_mean(
        df,
        pd.DatetimeIndex(df.resample(time_interval_str, **opts).groups.keys(), freq="infer"),
        **opts
    )


'''
Function that converts unequal-distributed time data to equally distributed time data. 
The value will be the weighted average of of the samples in the given time frame.
https://stackoverflow.com/questions/46030055/python-time-weighted-average-pandas-grouped-by-time-interval
'''


def resample_time_weighted_mean(x, target_index, closed=None, label=None):
    shift = 1 if closed == "right" else -1
    fill = "bfill" if closed == "right" else "ffill"
    # Determine length of each interval (daylight saving aware)
    extended_index = target_index.union(
        [target_index[0] - target_index.freq, target_index[-1] + target_index.freq]
    )
    interval_lengths = -extended_index.to_series().diff(periods=shift)

    # Create a combined index of the source index and target index and reindex to combined index
    combined_index = x.index.union(extended_index)
    x = x.reindex(index=combined_index, method=fill)
    interval_lengths = interval_lengths.reindex(index=combined_index, method=fill)

    # Determine weights of each value and multiply source values
    weights = -x.index.to_series().diff(periods=shift) / interval_lengths
    x = x.mul(weights, axis=0)

    # Resample to new index, the final reindex is necessary because resample
    # might return more rows based on the frequency
    return (
        x.resample(target_index.freq, closed=closed, label=label)
            .sum()
            .reindex(target_index)
    )


month_number_dict = {
    'januar': 1,
    'februar': 2,
    'marts': 3,
    'april': 4,
    'maj': 5,
    'juni': 6,
    'juli': 7,
    'august': 8,
    'september': 9,
    'oktober': 10,
    'november': 11,
    'december': 12,
}


def filename_to_datetime(s: str) -> datetime.datetime:
    year, month = (s.split('/')[-1].split('.')[0].split('_')[1:])
    month_nr = month_number_dict.get(month.lower())

    return datetime.datetime(int(year), month_nr, 1)
