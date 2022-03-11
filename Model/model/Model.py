import os
from glob import glob
import pandas as pd
from model.PumpingStation import PumpingStation
from pumping_station_enum import PUMPING_STATION_ENUM as PS
import log
from tqdm import tqdm
from multiprocessing import Pool
from time import time
from parser_functions import df_time_interpolate,filename_to_datetime
import datetime
import re

class Model:
    def __init__(self, to_process, path_hist, path_pump_info, path_clean_water, time_interval, nr_threads):
        self.pumping_stations = {}
        self.nr_threads = nr_threads
        self.all_measurements = None
        self.time_interval = time_interval

        # Step 1: Parse Pumping Stations
        self.parse_ps_info(to_process, path_pump_info)
        # Step 2: Parse Historical Data
        self.parse_measurements(to_process, path_hist)

    def parse_ps_info(self, to_process, data_path):
        df = pd.read_csv(data_path).T[1:]
        for ps_name, ps_loc in df.iterrows():
            if PS(ps_name) in to_process:
                self.pumping_stations[ps_name] = PumpingStation(ps_name, ps_loc)
        log.update(f"- importing {len(self.pumping_stations)} pumping stations...")

    def parse_measurements(self, to_process, data_path):
        EXT = '*.CSV'
        all_csv_filepaths = [file for path, subdir, files in os.walk(data_path) for file in
                             glob(os.path.join(path, EXT))]

        csv_files = []
        for f in all_csv_filepaths:
            filename = f.split('/')[-1]
            ps_name = PS(filename.split('_')[0])
            if ps_name in to_process:
                csv_files.append((f,ps_name))

        # Parse data over available threads:
        start_time = time()
        log.update(f"\t\t- Parsing {len(csv_files)} CSV files using {self.nr_threads} Threads...")
        if self.nr_threads > 1:
            pool = Pool(processes=self.nr_threads)
            measurements = list(tqdm(pool.imap(self.worker_process_measurements_csv, csv_files),total=len(csv_files)))
            pool.close()
            pool.join()
        else:
            measurements = [self.worker_process_measurements_csv(csv_file) for csv_file in csv_files]

        self.all_measurements = pd.concat(measurements, axis=0, ignore_index=True)
        log.success(f"\t\t- Done in {time() - start_time} seconds")
        log.success(f"\t\t- imported historical data for {len(self.pumping_stations)} pumping stations")


    def worker_process_measurements_csv(self, filepath_and_ps_name) -> pd.DataFrame:
        filepath, ps_name = filepath_and_ps_name
        try:
            if ps_name == PS.PST232:
                return self.parse_232_239(filepath, ps_name)
            if ps_name == PS.PST239:
                return self.parse_232_239(filepath, ps_name)
            else:
                log.fail(f"{ps_name} not implemented")

        except Exception as e:
            log.fail(f"Could not parse '{filepath}':")
            raise e


    def parse_232_239(self, filepath, ps_name):
        if filepath.split('/')[-1] == "PST239_Februar_Graphs.CSV":
            return
        log.debug(f"{filepath}: Start parsing")
        df = pd.read_csv(filepath, encoding='cp1252', sep=';', decimal=',').reset_index().iloc[:, 1:]
        log.debug(f"{filepath}: \t Length = {len(df)}")
        df.columns = ['time', 'current_1', 'current_2', 'water_level', 'outflow_level']
        df = df.astype({
            'current_1': float,
            'current_2': float,
            'water_level': float,
            'outflow_level': float,
        })
        log.debug(f"{filepath}: Converting time column")
        time_format_sample = df.iloc[0].time

        if re.match("[0-9]{2}:[0-9]{2},[0-9]", time_format_sample):
            df.time = pd.to_datetime(df.time, format="%M:%S,%f") # converts just time, not the date and hour
            df.time = self.calculate_timestamp(df.time, filepath)
        elif re.match("[0-9]{2}-[0-9]{2}-[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}", time_format_sample):
            df.time = pd.to_datetime(df.time, format="%d-%m-%Y %H:%M:%S,%f")
        else:
            raise Exception("Unknown time format")

        df = df.set_index(df.time)
        df.drop(columns=['time'], inplace=True)

        log.debug(f"{filepath}: Resample (interpolate) data with {self.time_interval} seconds interval")
        old_len = len(df)
        df = df_time_interpolate(df, self.time_interval)
        log.debug(f"{filepath}:\t- Resampling finished (old length = {old_len}, new length = {len(df)})")

        log.debug(f"{filepath}: Converting currents columns")
        df['currents'] = df.apply(lambda row: [row.current_1, row.current_2], axis=1)
        df['current_tot'] = df.apply(lambda row: row.current_1 + row.current_2, axis=1)
        df.drop(columns=['current_1', 'current_2'], inplace=True)
        df['pumping_station'] = ps_name.name

        log.debug(f"{filepath}: Finished ")
        return df

    def calculate_timestamp(self, time_series: pd.Series, filepath: str):
        current_date = filename_to_datetime(filepath)
        previous_time = datetime.time(0, 0, 0, 0)

        for ix, row in time_series.items():
            time_without_date = row.time()
            if time_without_date >= previous_time:
                previous_time = time_without_date
                time_series.at[ix] = current_date.replace(
                    minute=time_without_date.minute,
                    second=time_without_date.second,
                    microsecond=time_without_date.microsecond,
                )
            else:
                added_time = datetime.timedelta(hours=1)
                current_date += added_time
                previous_time = datetime.time(0, 0, 0, 0)
        return time_series




