import os
from glob import glob
from multiprocessing import Pool
from parser import parse_232, parse_233, parse_234, parse_237, parse_238, parse_239, parse_240
from time import time

import pandas as pd
from tqdm import tqdm

import log
from model.PumpingStation import PumpingStation
from pumping_station_enum import PUMPING_STATION_ENUM as PS


class Model:
    def __init__(
            self,
            to_process,
            path_hist,
            path_pump_info,
            path_clean_water,
            time_interval,
            nr_threads,
    ):
        self.pumping_stations = {}
        self.nr_threads = nr_threads
        self.all_measurements: pd.DataFrame = pd.DataFrame()
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
        EXT = "*.CSV"
        all_csv_filepaths = [
            file
            for path, subdir, files in os.walk(data_path)
            for file in glob(os.path.join(path, EXT))
        ]

        csv_files = []
        for f in all_csv_filepaths:
            filename = f.split("/")[-1]
            ps_name = PS(filename.split("_")[0])
            if ps_name in to_process:
                csv_files.append((f, ps_name))

        # Parse data over available threads:
        start_time = time()
        log.update(
            f"\t\t- Parsing {len(csv_files)} CSV files using {self.nr_threads} Threads..."
        )
        if self.nr_threads > 1:
            pool = Pool(processes=self.nr_threads)
            measurements = list(
                tqdm(
                    pool.imap(self.worker_process_measurements_csv, csv_files),
                    total=len(csv_files),
                )
            )
            pool.close()
            pool.join()
        else:
            measurements = [
                self.worker_process_measurements_csv(csv_file) for csv_file in csv_files
            ]

        log.update("\t\t- Merging results...")
        self.all_measurements = pd.concat(measurements, axis=0)
        log.success(f"\t\t- Done in {time() - start_time} seconds")
        log.success(
            f"\t\t- imported historical data for {len(self.pumping_stations)} pumping stations"
        )

    def worker_process_measurements_csv(self, filepath_and_ps_name) -> pd.DataFrame:
        filepath, ps_name = filepath_and_ps_name
        try:
            if ps_name == PS.PST232:
                return parse_232(filepath, ps_name, self.time_interval)
            elif ps_name == PS.PST233:
                return parse_233(filepath, ps_name, self.time_interval)
            elif ps_name == PS.PST234:
                return parse_234(filepath, ps_name, self.time_interval)
            elif ps_name == PS.PST237:
                return parse_237(filepath, ps_name, self.time_interval)
            elif ps_name == PS.PST238:
                return parse_238(filepath, ps_name, self.time_interval)
            elif ps_name == PS.PST239:
                return parse_239(filepath, ps_name, self.time_interval)
            elif ps_name == PS.PST240:
                return parse_240(filepath, ps_name, self.time_interval)
            else:
                log.fail(f"{ps_name} not implemented")

        except Exception as e:
            log.fail(f"Could not parse '{filepath}':")
            raise e
