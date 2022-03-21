import sys
sys.path.insert(1, '../')
import os
from glob import glob
from multiprocessing import Pool
import numpy as np

from parser import parse_232, parse_233, parse_234, parse_237, parse_238, parse_239, parse_240, parse_water_consumption_html
from time import time
from math import pi

import pandas as pd
from tqdm import tqdm

import log
from model.PumpingStation import PumpingStation
from model.Pump import Pump
from pumping_station_enum import PUMPING_STATION_ENUM as PS
from model.pst_connections import link_delay_dict

class Model:
    def __init__(
            self,
            to_process,
            path_hist,
            path_ps_location,
            path_ps_info,
            path_water_consumption,
            path_pump_gain,
            time_interval,
            include_data_validation,
            include_weather,
            include_water_consumption,
            nr_threads,
            remove_invalid_readings
    ):
        self.pumping_stations: dict[PS, PumpingStation] = {}
        self.nr_threads = nr_threads
        self.all_measurements: pd.DataFrame = pd.DataFrame()
        self.all_water_consumption = pd.DataFrame()
        self.time_interval = time_interval
        self.include_data_validation = include_data_validation
        self.include_weather = include_weather
        self.include_water_consumption = include_water_consumption
        self.remove_invalid_readings = remove_invalid_readings

        # Step 1: Parse Pumping Stations Location Data
        self.parse_ps_location(to_process, path_ps_location)
        # Step 2: Parse Pumping Stations
        self.parse_ps_info(to_process, path_ps_info)
        # Step 3: Parse Pump gains
        self.parse_pump_gain(path_pump_gain)
        # Step 4: Import Water Consumption data
        self.parse_water_consumption_data(path_water_consumption)
        # Step 5: Parse Historical Data
        self.parse_measurements(to_process, path_hist)
        # Step 6: Define pipeline connections
        self.link_pumping_stations(to_process)
        log.success("Model is ready to use")

    def parse_ps_location(self, to_process, location_data_path):
        df = pd.read_csv(location_data_path).T[1:]
        for ps_name, ps_loc in df.iterrows():
            ps = PS(ps_name)
            if ps in to_process:
                self.pumping_stations[ps] = PumpingStation(ps, ps_loc)
        log.update(f"Imported information about {len(self.pumping_stations)} pumping stations.")

    def parse_ps_info(self, to_process, ps_info_path):
        df = pd.read_excel(ps_info_path, usecols='A:X').iloc[1:8]
        df.set_index(df.Item, inplace=True)
        for ps_name, ps_info in df.iterrows():
            ps = PS(ps_name)
            if ps in to_process:
                ps_to_update = self.pumping_stations[ps]
                ps_to_update.description = ps_info.Name
                ps_to_update.description = ps_info.Adress
                if ps_info.Volumen is not np.nan:
                    ps_to_update.volume = float(ps_info.Volumen[1:].replace(',','.'))
                ps_to_update.radius_est = float(ps_info['Est.'])
                ps_to_update.max_height = float(ps_info['Max'])
                ps_to_update.volume = ps_to_update.radius_est ** 2 * pi * 1000 * ps_to_update.max_height

                if ps_info.P1:
                    pump_to_add = Pump()
                    pump_to_add.max_kW = ps_info.P1
                    pump_to_add.gain_L_kWh = ps_info['Pumps [L/kWh]']
                    ps_to_update.pumps.append(pump_to_add)
                if ps_info.P2:
                    pump_to_add = Pump()
                    pump_to_add.max_kW = ps_info.P2
                    pump_to_add.gain_L_kWh = ps_info['Unnamed: 11']
                    ps_to_update.pumps.append(pump_to_add)
                if ps_info.P3:
                    pump_to_add = Pump()
                    pump_to_add.max_kW = ps_info.P3
                    pump_to_add.gain_L_kWh = ps_info['Unnamed: 12']
                    ps_to_update.pumps.append(pump_to_add)

                self.pumping_stations[ps] = ps_to_update
                log.update(f"Pump station information imported for {ps_name}")

    def parse_measurements(self, to_process, data_path):
        EXT = "*.CSV"
        all_csv_filepaths = [
            file
            for path, subdir, files in os.walk(data_path)
            for file in glob(os.path.join(path, EXT))
        ]

        csv_to_process = []
        for f in all_csv_filepaths:
            filename = f.split("/")[-1]
            ps_name = PS(filename.split("_")[0])
            if ps_name in to_process:
                csv_to_process.append((f, ps_name))

        # Parse data over available threads:
        start_time = time()
        log.update(f"Parsing {len(csv_to_process)} 'Historical Data' CSV files using {self.nr_threads} Threads...")
        if self.nr_threads > 1:
            # Multithreading approach -> Great for performance
            pool = Pool(processes=self.nr_threads)
            measurements = list(
                tqdm(
                    pool.imap(self.worker_process_measurements_csv, csv_to_process),
                    total=len(csv_to_process),
                )
            )
            pool.close()
            pool.join()
        else:
            # Conventional single threaded approach -> Great for debugging
            measurements = [
                self.worker_process_measurements_csv(csv_file) for csv_file in csv_to_process
            ]

        log.update("Merging results of historical data...")
        self.all_measurements = pd.concat(measurements, axis=0)
        log.update(f"Imported historical data for {len(self.pumping_stations)} pumping stations in {time() - start_time} seconds")

    def worker_process_measurements_csv(self, filepath_and_ps_name) -> pd.DataFrame:
        filepath, ps_name = filepath_and_ps_name
        try:
            if ps_name == PS.PST232:
                return parse_232(filepath, self.pumping_stations[ps_name], self)
            elif ps_name == PS.PST233:
                return parse_233(filepath, self.pumping_stations[ps_name], self)
            elif ps_name == PS.PST234:
                return parse_234(filepath, self.pumping_stations[ps_name], self)
            elif ps_name == PS.PST237:
                return parse_237(filepath, self.pumping_stations[ps_name], self)
            elif ps_name == PS.PST238:
                return parse_238(filepath, self.pumping_stations[ps_name], self)
            elif ps_name == PS.PST239:
                return parse_239(filepath, self.pumping_stations[ps_name], self)
            elif ps_name == PS.PST240:
                return parse_240(filepath, self.pumping_stations[ps_name], self)
            else:
                log.fail(f"{ps_name.value} not implemented")

        except Exception as e:
            log.fail(f"Could not parse '{filepath}':")
            raise e

    def parse_water_consumption_data(self, data_path):
        if self.include_water_consumption:
            html_files = [html_file for html_file in glob(os.path.join(data_path, '*.html'))]
            df_consumption_by_month = [parse_water_consumption_html(html_file) for html_file in html_files]
            self.all_water_consumption = pd.concat(df_consumption_by_month, axis=0)

    def parse_pump_gain(self, data_path):
        if self.remove_invalid_readings:
            df = pd.read_csv(data_path)
            for ps_name in df.iterrows():
                self.pumping_stations[PS(ps_name[1][0])].gain = list(map(float,ps_name[1][1].strip('[]').split(',')))

    def link_pumping_stations(self, to_process):
        links_count = 0
        for link, delay in link_delay_dict.items():
            pst_from, pst_towards = link
            if pst_from in to_process and pst_towards in to_process:
                log.debug(f"Laying pipes from '{pst_from.name}' towards '{pst_towards.name}'")
                to_update_upstream = self.pumping_stations.get(pst_from)
                to_update_upstream.pumping_stations_upstream.append((pst_towards, delay))
                to_update_downstream = self.pumping_stations.get(pst_towards)
                to_update_downstream.pumping_stations_downstream.append((pst_from, delay))
                links_count += 1
        log.update(f"{links_count} sewage pipes imported from 'pst_connections.py'")