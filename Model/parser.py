import datetime
import re

import numpy as np
import pandas as pd
import math
import log
from pumping_station_enum import PUMPING_STATION_ENUM as PS
from meteostat import Point, Hourly
from model.PumpingStation import PumpingStation
from data_validator import validate


def df_time_interpolate(df: pd.DataFrame, time_interval_sec: int):
    opts = dict(closed="left", label="left")
    time_interval_str = f"{time_interval_sec}S"
    return resample_time_weighted_mean(
        df,
        pd.DatetimeIndex(
            df.resample(time_interval_str, **opts).groups.keys(), freq="infer"
        ),
        **opts,
    )


"""
Function that converts unequal-distributed time data to equally distributed time data. 
The value will be the weighted average of of the samples in the given time frame.
https://stackoverflow.com/questions/46030055/python-time-weighted-average-pandas-grouped-by-time-interval
"""


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
    "januar": 1,
    "februar": 2,
    "marts": 3,
    "april": 4,
    "maj": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
}


def remove_invalid_readings(df: pd.DataFrame, pump_station: PumpingStation, margin=0.1):
    power_p1 = (math.sqrt(3) * df.current_1 * 400 / 1000)
    power_p2 = (math.sqrt(3) * df.current_2 * 400 / 1000)
    power_p3 = 0
    gain_p1 = pump_station.gain[0]
    gain_p2 = pump_station.gain[1]
    gain_p3 = 0
    if 'current_3' in df.columns:
        power_p3 = (math.sqrt(3) * df.current_3 * 400 / 1000)
        gain_p3 = pump_station.gain[2]
    corrected_outflow = power_p1 * gain_p1 + power_p2 * gain_p2 + power_p3 * gain_p3
    correct = (df.outflow_level * (1 - margin) > corrected_outflow)  | (df.outflow_level * (margin + 1) < corrected_outflow * (1 + margin))
    df[~correct].outflow_level = corrected_outflow[~correct]
    return df


def filename_to_datetime(s: str) -> datetime.datetime:
    year, month = s.split("/")[-1].split(".")[0].split("_")[1:]
    month_nr = month_number_dict.get(month.lower())

    return datetime.datetime(int(year), month_nr, 1)


def parse_232(filepath, pump_station: PumpingStation, model):
    column_mapping = {'Tid ': 'time',
                      'PST-232-P1-Strøm Senest målte motorstrøm P1 (0.0-20.0 A)': 'current_1',
                      'PST-232-P2-Strøm Senest målte motorstrøm P2 (0.0-20.0 A)': 'current_2',
                      'PST-232-Niveau Niveau (0.00-10.00 m)': 'water_level',
                      'PST-232-Flow_ud Flow (0.0-250.0 m3/h)': 'outflow_level'}
    return parse_232_233_234_238_239_240(filepath, pump_station, column_mapping, model)


def parse_233(filepath, pump_station: PumpingStation, model):
    column_mapping = {'Tid ': 'time',
                      'PST-233-P1-Strøm Senest målte motorstrøm P1 (0.0-100.0 A)': 'current_1',
                      'PST-233-P2-Strøm Senest målte motorstrøm P2 (0.0-100.0 A)': 'current_2',
                      'PST-233-Niveau Niveau (0.00-10.00 m)': 'water_level',
                      'PST-233-Flow_ud Flow (0.0-500.0 m3/h)': 'outflow_level'}
    return parse_232_233_234_238_239_240(filepath, pump_station, column_mapping, model)


def parse_234(filepath, pump_station: PumpingStation, model):
    column_mapping = {'Tid ': 'time',
                      'PST-234-P1-Strøm Senest målte motorstrøm P1 (0.0-100.0 A)': 'current_1',
                      'PST-234-P2-Strøm Senest målte motorstrøm P2 (0.0-100.0 A)': 'current_2',
                      'PST-234-Niveau Niveau (0.00-10.00 m)': 'water_level',
                      'PST-234-Flow_ud Flow (0.0-500.0 m3/h)': 'outflow_level'}
    return parse_232_233_234_238_239_240(filepath, pump_station, column_mapping, model)


def parse_237(filepath, pump_station: PumpingStation, model):
    column_mapping = {'Tid ': 'time',
                      'PST-237-hist-niv Niveau kurve (0.00-5.00 m)': 'water_level',
                      'PST-237-strøm-P1 Strøm P1 kurve (0.0-100.0 A)': 'current_1',
                      'PST-237-strøm-P2 Strøm P2 kurve (0.0-100.0 A)': 'current_2',
                      'PST-237-flow-hist Flow kurve (0.0-500.0 m3/h)': 'outflow_level',
                      'PST-237-P2-Effekt Aktuel motor effekt (0.0-100.0 kW)': '_unused_1',
                      'PST-237-P1-Effekt Aktuel motor effekt (0.0-100.0 kW)': '_unused_2'}
    return parse_232_233_234_238_239_240(filepath, pump_station, column_mapping, model)


def parse_238(filepath, pump_station: PumpingStation, model):
    column_mapping = {'Tid ': 'time',
                      'PST-238-hist-niv Niveau kurve (0.00-5.00 m)': 'water_level',
                      'PST-238-strøm-P1 Strøm P1 kurve (0.0-10.0 A)': 'current_1',
                      'PST-238-strøm-P2 Strøm P2 kurve (0.0-10.0 A)': 'current_2',
                      'PST-238-flow-hist Flow kurve (0.0-500.0 m3/h)': 'outflow_level'}
    return parse_232_233_234_238_239_240(filepath, pump_station, column_mapping, model)


def parse_239(filepath, pump_station: PumpingStation, model):
    column_mapping = {'Tid ': 'time',
                      'PST-239_Niveau Niveau: (0.00-5.00 m)': 'water_level',
                      'PST-239_Flowmåler (0.0-500.0 m3/h)': 'outflow_level',
                      'PST-239_P1_Strøm (0.0-30.0 A)': 'current_1',
                      'PST-239_P2_Strøm (0.0-80.0 A)': 'current_2'}
    return parse_232_233_234_238_239_240(filepath, pump_station, column_mapping, model)


def parse_240(filepath, pump_station: PumpingStation, model):
    column_mapping = {'Tid ': 'time',
                      'PST-240_Niveau Niveau: (0.00-5.00 m)': 'water_level',
                      'PST-240_Flowmåler (0.0-500.0 m3/h)': 'outflow_level',
                      'PST-240_P1_Strøm (0.0-160.0 A)': 'current_1',
                      'PST-240_P2_Strøm (0.0-30.0 A)': 'current_2',
                      'PST-240_P2_Effekt Effektmåling pumpe 2 (0.0-60.0 kW)': '_unused_1',
                      'PST-240_P1_Effekt Effektmåling pumpe 1 (0.0-150.0 kW)': '_unused_2',
                      'PST-240_P3_Strøm (0.0-30.0 A)': 'current_3',
                      'PST-240_P3_Effekt Effektmåling pumpe 3 (0.0-60.0 kW)': '_unused_3',
                      }
    return parse_232_233_234_238_239_240(filepath, pump_station, column_mapping, model)


def parse_232_233_234_238_239_240(filepath: str, pump_station: PumpingStation, column_mapping: dict,
                                  model) -> pd.DataFrame:
    filename = filepath.split("/")[-1]
    if filename == "PST239_Februar_Graphs.CSV":
        return

    log.debug(f"{filepath}: Start parsing...")
    df = (
        pd.read_csv(filepath, encoding="cp1252", sep=";", decimal=",")
            .reset_index()
            .iloc[:, 1:]
    )
    log.debug(f"{filepath}: \t Number of samples = {len(df)}")
    df.rename(columns=column_mapping, inplace=True, errors='raise')
    # keep useful columns.
    if pump_station.name is PS.PST240:
        df = df[["time", "current_1", "current_2", "current_3", "water_level", "outflow_level"]]
    else:
        df = df[["time", "current_1", "current_2", "water_level", "outflow_level"]]

    df = df.astype(
        {
            "current_1": float,
            "current_2": float,
            "water_level": float,
            "outflow_level": float,
        }
    )

    if pump_station.name is PS.PST240:
        df = df.astype({"current_3": float, })

    # df = remove_invalid_readings(df, pump_station, margin=0.1)

    log.debug(f"{filepath}: Converting time column...")
    time_format_sample = df.iloc[0].time

    if re.match(
            "[0-9]{2}:[0-9]{2},[0-9]", time_format_sample
    ):  # see PST232_2020_Juli.CSV
        df.time = pd.to_datetime(
            df.time, format="%M:%S,%f"
        )  # converts just time, not the date and hour
        df.time = calculate_timestamp(df.time, filepath)
        df.drop_duplicates(subset=["time"], keep="first", inplace=True)
    elif re.match(
            "[0-9]{2}-[0-9]{2}-[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}",
            time_format_sample,
    ):
        df.time = pd.to_datetime(df.time, format="%d-%m-%Y %H:%M:%S,%f")
    else:
        raise Exception("Unknown time format")

    if (filename == "PST232_2020_Oktober.CSV"
            or filename == "PST239_Maj.CSV"
            or filename == "PST239_April.CSV"
            or filename == "PST239_Oktober.CSV"
            or filename == "PST234_2020_Oktober.CSV"
            or filename == "PST233_2020_Oktober.CSV"
            or filename == "PST233_2021_Juli.CSV"
            or filename == "PST240_2020_Maj.CSV"
            or pump_station.name == PS.PST237
            or pump_station.name == PS.PST238):
        df.drop_duplicates(subset=["time"], keep="first", inplace=True)
        df = df.sort_values(by=["time"])

    df = df.set_index(df.time)
    df.drop(columns=["time"], inplace=True)

    if model.include_data_validation:
        df = validate(df, pump_station.name)

    if model.time_interval is not None:
        log.debug(f"{filepath}: Resample (interpolate) data with {model.time_interval} seconds interval...")
        old_len = len(df)
        df = df_time_interpolate(df, model.time_interval)
        log.debug(f"{filepath}:\t- Resampling finished (old length = {old_len}, new length = {len(df)})")

    log.debug(f"{filepath}: Converting currents columns")
    if pump_station.name is PS.PST240:
        df["currents"] = df.apply(lambda row: [row.current_1, row.current_2, row.current_3], axis=1)
    else:
        df["currents"] = df.apply(lambda row: [row.current_1, row.current_2], axis=1)
    df["current_tot"] = df.apply(lambda row: row.current_1 + row.current_2, axis=1)
    # if model.remove_invalid_readings:
    #     df = df.apply(remove_invalid_readings, axis=1)
    #     df["current_tot"] = df.apply(lambda row: row.current_1 + row.current_2, axis=1)

    df["current_tot"] = df.apply(lambda row: row.current_1 + row.current_2, axis=1)
    df.drop(columns=["current_1", "current_2"], inplace=True)
    if pump_station.name is PS.PST240:
        df.drop(columns=["current_3"], inplace=True)
    df["pumping_station"] = pump_station.name

    if model.include_weather:
        log.debug(f"{filepath}: Adding weather data...")
        df = add_weather_data(df, pump_station.lat, pump_station.lon)

    if model.include_water_consumption:
        log.debug(f"{filepath}: Adding water consumption data...")
        df = add_water_consumption_data(df, model)



    log.update(f"{filepath}: Finished ")
    return df

def calculate_timestamp(time_series: pd.Series, filepath: str):
    current_date = filename_to_datetime(filepath)
    previous_time = datetime.time(0, 0, 0, 0)

    for ix, row in time_series.items():
        time_without_date = row.time()
        if time_without_date >= previous_time:
            previous_time = time_without_date
        else:
            added_time = datetime.timedelta(hours=1)
            current_date += added_time
            previous_time = datetime.time(0, 0, 0, 0)

        time_series.at[ix] = current_date.replace(
            minute=time_without_date.minute,
            second=time_without_date.second,
            microsecond=time_without_date.microsecond,
        )

    return time_series


def add_weather_data(pump, lat, long):
    weather = fetch_historic_weather(pump.index[0], pump.index[-1], lat, long)
    return  pump.join(weather[['temp', 'prcp', 'snow']], how='outer').apply(pd.Series.interpolate, args=('time',))



def fetch_historic_weather(start: datetime, end: datetime, long, lat):  # More details in ATTACHMENT 1
    pump_station_loc = Point(long, lat)

    # Get hourly data for 2020
    weather_data_hour = Hourly(pump_station_loc, start, end)
    weather_data_hour = weather_data_hour.normalize()  # Ensures there is one data point per hour
    weather_data_hour = weather_data_hour.fetch()  # Fetches data from given coordinates

    # print(weather_data_hour)
    return weather_data_hour


def parse_water_consumption_html(filepath):
    df = pd.read_html(filepath, encoding='cp1252')[2]
    year = filepath.split(' ')[-1].split('.')[-2].split('[')[0]
    df = df[4:-3]
    df.columns = [
        'date',
        'in_flow_total',
        'out_flow_total',
        'raw_water_flow_ponydalen',
        'raw_water_flow_maegård',
        'raw_water_flow_sandkaas',
        'raw_water_flow_sandkaas_max',
    ]
    df = df.astype(
        {
            "in_flow_total": float,
            "out_flow_total": float,
            "raw_water_flow_ponydalen": float,
            "raw_water_flow_maegård": float,
            "raw_water_flow_sandkaas": float,
            "raw_water_flow_sandkaas_max": float,
        }
    )
    df.date = df.date.apply(lambda d: d + '-' + year)
    df.date = pd.to_datetime(df.date, format="%d-%m-%Y")
    df.set_index(df.date, inplace=True)
    df.drop(columns=['date'], inplace=True)

    return df

def get_clean_water_usage(x, all_water_consumption):
    date = pd.to_datetime(x.date())
    if date in all_water_consumption.index:
        daily_consumption = all_water_consumption[all_water_consumption.index == date].iloc[0]
        return daily_consumption.out_flow_total
    else:
        return np.NAN

def add_water_consumption_data(df: pd.DataFrame, model):
    df['clean_water_usage_day'] = df.index.to_series().map(lambda x: get_clean_water_usage(x, model.all_water_consumption))
    return df