import os
from glob import glob
from multiprocessing import Pool
import numpy as np
import math

from src.parser import parse_232, parse_233, parse_234, parse_237, parse_238, parse_239, parse_240, \
    parse_water_consumption_html, parse_data_validation_props
from time import time
from math import pi

import pandas as pd
from tqdm import tqdm
import src.log as log
import json
from src.model.PumpingStation import PumpingStation
from src.model.Pump import Pump
from src.pumping_station_enum import PUMPING_STATION_ENUM as PS
from src.model.pst_connections import link_delay_dict
from src.parser import validate


class Model:
    def __init__(
            self,
            to_process,
            path_hist,
            path_ps_location,
            path_ps_info,
            path_water_consumption,
            path_pump_gain,
            path_validator_properties,
            path_sea_level_data,
            time_interval,
            include_data_validation,
            include_weather,
            include_water_consumption,
            include_sea_level,
            apply_data_corrections,
            pst_238_include_inflow,
            nr_threads,
    ):
        self.pumping_stations: dict[PS, PumpingStation] = {}
        self.nr_threads = nr_threads
        self.all_measurements: pd.DataFrame = pd.DataFrame()
        self.all_water_consumption = pd.DataFrame()
        self.sea_level_data = pd.DataFrame()
        self.time_interval = time_interval
        self.include_data_validation = include_data_validation
        self.include_weather = include_weather
        self.include_water_consumption = include_water_consumption
        self.include_sea_level = include_sea_level
        self.apply_data_corrections = apply_data_corrections
        self.pst_238_include_inflow = pst_238_include_inflow

        if (time_interval is not None) and include_data_validation and not pst_238_include_inflow:
            log.warning("Data validation and time interpolation both enabled: ")
            log.warning("Data will be cleaned but you cannot see the errors.")
            log.warning("Set 'INCLUDE_DATA_VALIDATION' on 'True' and set 'TIME_INTERVAL' on 'None'")
            log.warning("To view error messages")

        if apply_data_corrections and not include_data_validation:
            log.fail("You cannot apply data corrections without data validation")
            log.fail("Set 'INCLUDE_DATA_VALIDATION' to 'True'")
            exit()

        if ((PS.PST238 not in to_process) or not len(to_process) == 1) and pst_238_include_inflow:
            log.fail("When 'INCLUDE_INFLOW' is set to 'True' you cannot ONLY process 'PST238'")
            log.fail("Please set exactly to 'PUMPING_STATIONS = [ps.PST238]'")
            exit()

        if pst_238_include_inflow and not include_data_validation:
            log.fail("You cannot include inflow without data validation. This is needed to set the cycles.")
            log.fail("Set 'INCLUDE_DATA_VALIDATION' to 'True'")
            exit()

        if pst_238_include_inflow and not time_interval:
            log.fail("Calculating inflow without time interpolation is not (yet) supported")
            log.fail("Set 'TIME_INTERVAL' to a integer number to fix this problem")
            exit()

        # Step 1: Parse Pumping Stations Location Data
        self.parse_ps_location(to_process, path_ps_location)
        # Step 2: Parse Pumping Stations
        self.parse_ps_info(to_process, path_ps_info)
        # Step 3: Parse Pump gains
        self.parse_pump_gain(to_process, path_pump_gain)
        # Step 4: Import Water Consumption data
        self.parse_water_consumption_data(path_water_consumption)
        # Step 5: Parse Historical Data
        self.parse_measurements(to_process, path_hist, path_validator_properties)
        # Step 6: Parse Sea Level Data
        self.parse_sea_level_data(path_sea_level_data)
        # Step 7: Define pipeline connections
        self.link_pumping_stations(to_process)
        # Step 8: Include water inflow Data
        self.include_inflow()
        log.success("src is ready to use")

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
                    ps_to_update.volume = float(ps_info.Volumen[1:].replace(',', '.'))
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

    def parse_measurements(self, to_process, path_hist, path_validator_properties):
        EXT = "*.CSV"
        all_csv_filepaths = [
            file
            for path, subdir, files in os.walk(path_hist)
            for file in glob(os.path.join(path, EXT))
        ]

        csv_to_process = []
        for f in all_csv_filepaths:
            filename = os.path.basename(f)
            ps_name = PS(filename.split("_")[0])
            if ps_name in to_process:
                csv_to_process.append((f, ps_name))

        # Load validator properties if needed
        if self.include_data_validation:
            parse_data_validation_props(self, path_validator_properties, to_process)

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
        log.update(
            f"Imported historical data for {len(self.pumping_stations)} pumping stations in {time() - start_time} seconds")

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

    def parse_pump_gain(self, to_process, data_path):
        df = pd.read_csv(data_path)
        for ps_name in df.iterrows():
            ps = PS(ps_name[1][0])
            if ps in to_process:
                self.pumping_stations[ps].gain = list(map(float, ps_name[1][1].strip('[]').split(',')))

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

    # Insert sea levels
    def split_df_chunks(self, data_df, chunk_size):
        total_length = len(data_df)
        total_chunk_num = math.ceil(total_length / chunk_size)
        normal_chunk_num = math.floor(total_length / chunk_size)
        chunks = []
        for i in range(normal_chunk_num):
            chunk = data_df[(i * chunk_size):((i + 1) * chunk_size)]
            chunks.append(chunk)
        if total_chunk_num > normal_chunk_num:
            chunk = data_df[(normal_chunk_num * chunk_size):total_length]
            chunks.append(chunk)
        return chunks

    def get_sea_levels_row(self, date):
        ix = self.sea_level_data.index.get_indexer([date], method='nearest')[0]
        return self.sea_level_data.iloc[ix - 7: ix - 1].mean().iat[0], \
               self.sea_level_data.iloc[ix: ix + 6].mean().iat[0]

    def insert_sea_levels(self, df_part):
        df_part['tides'] = df_part.index.map(self.get_sea_levels_row)
        df_part['sea_level_last_30min'] = df_part.tides.apply(lambda x: x[0])
        df_part['sea_level_next_30min'] = df_part.tides.apply(lambda x: x[1])
        return df_part

    def parse_sea_level_data(self, path_sea_level_data):
        if self.include_sea_level:
            log.debug("Including sea level data")
            ocean_data = json.load(open(path_sea_level_data))
            self.sea_level_data = pd.DataFrame(ocean_data.get('features'))[['properties']]

            self.sea_level_data.index = pd.to_datetime(
                self.sea_level_data.properties.apply(lambda x: x.get('observed')))
            self.sea_level_data.index = self.sea_level_data.index.tz_localize(None)
            self.sea_level_data = self.sea_level_data.sort_index(axis=0)
            self.sea_level_data['sea_level'] = self.sea_level_data.properties.apply(lambda x: x.get('value'))

            # assert all rows have parameterId sea_reg
            self.sea_level_data['parameter'] = self.sea_level_data.properties.apply(lambda x: x.get('parameterId'))
            assert (self.sea_level_data.parameter == 'sea_reg').all()
            self.sea_level_data.drop(columns=['parameter', 'properties'], inplace=True)

            if self.nr_threads > 1:
                log.debug(f"Imputing sea level data with {self.nr_threads} threads...")
                pool = Pool(processes=self.nr_threads)
                chunk_size = (len(self.all_measurements) // self.nr_threads) + 1
                measurements_chunks = self.split_df_chunks(self.all_measurements, chunk_size)
                measurements_list = list(
                    tqdm(
                        pool.imap(self.insert_sea_levels, measurements_chunks),
                        total=len(measurements_chunks),
                    )
                )
                pool.close()
                pool.join()
                self.all_measurements = pd.concat(measurements_list, axis=0)
                log.debug("Sea level data imported")
            else:
                log.debug("Imputing sea level data...")
                self.all_measurements = self.insert_sea_levels(self.all_measurements)
                log.debug("Sea level data imported")

            self.all_measurements.drop(columns=['tides'], inplace=True)
            log.update("Sea level data imported")

    def calc_inflow(self, df):
        alpha = (math.pi * (0.564189584 ** 2)) * 1000                                                       # Tank capacity in Liters
        df['Volume_RolAvg_Vol_7'] = df.loc[:, 'water_level'].rolling(window=4).mean() * alpha               # Tank usage in Liters
        df['Converted_Liter_diff'] = (df['water_level'].diff()) * alpha                                     # Difference with previous row in Liters
        df['CLd_RolAvg'] = df.loc[:, 'Converted_Liter_diff'].rolling(window=4).mean() / self.time_interval  # Water level: difference in L per second
        df['Converted_Outflow'] = df.loc[:, 'outflow_level'].rolling(window=4).mean() * 1000 / 3600         # Outflow from m3/h to L/s  # IMPORTANT this value needs to be scaled correctly
        df['inflow'] = df['CLd_RolAvg'] + df['Converted_Outflow']


    def include_inflow(self):
        if self.pst_238_include_inflow:
            pumping_station = self.pumping_stations[PS.PST238]
            log.debug("Doing final validation needed for inflow calculation")
            self.all_measurements['current_1'] = self.all_measurements.apply(lambda m: m.currents[0], axis=1)
            self.all_measurements['current_2'] = self.all_measurements.apply(lambda m: m.currents[1], axis=1)
            self.all_measurements = validate(self.all_measurements, pumping_station, False)
            log.debug("Done final validation needed for inflow calculation")
            self.calc_inflow(self.all_measurements)

            # Then impute the inflow for the measurements where pumps are turned on, and for following 4 measurements
            last_measurement_pumps_on = False
            start_ix, start_value, stop_delay = 0, 0, 0
            stop_delay_threshold = 6
            for ix, (current_date, m) in enumerate(self.all_measurements.iterrows()):
                if m.cycle_state and not last_measurement_pumps_on:  # Start of new cycle
                    start_ix = ix-1
                    start_value = self.all_measurements.iloc[ix - 2].inflow
                    last_measurement_pumps_on = True
                    stop_delay = 0
                elif m.cycle_state and last_measurement_pumps_on:  # Ongoing cycle
                    pass
                elif not m.cycle_state and last_measurement_pumps_on:  # End of cycle, start stop delay
                    stop_delay += 1
                    last_measurement_pumps_on = False
                elif 0 < stop_delay <= stop_delay_threshold:  # Stop delay
                    stop_delay += 1
                elif stop_delay == stop_delay_threshold+1:  # Stop delay ended
                    self.inpute_inflow(ix, start_ix, start_value)
                    start_value, start_ix, stop_delay = 0, 0, 0
                elif stop_delay == 0:  # Measurements where the pump is turned off
                    pass
                elif stop_delay != 0:
                    # Only for measurements where pumps have very low stopping time
                    self.inpute_inflow(ix, start_ix, start_value)
                    start_value, start_ix, stop_delay = 0, 0, 0
                else:
                    raise Exception(f"Unexpected state in include_inflow(): ({ix} - {current_date})")

            # Finally, we set the label
            self.all_measurements['inflow_label'] = self.flexibility_label(self.all_measurements)

    """"
    A very simple function to determine whether or not to impute the inflow or keep the (previously) calculated value.
    If we have heavy inflow, we want to keep the previously calculated value because it's safer.
    Otherwise, we want to impute the inflow to show a more accurate picture of the water inflow.
    """
    def possibility_of_heavy_inflow(self, start_ix, stop_ix):
        return False
        for ix in range(start_ix, stop_ix):
            # Assumption made here: if the calculated inflow hits zero or lower, than it's very unlikely
            # that there is heavy inflow. Therefore, return false.
            if self.all_measurements.iloc[ix].inflow <= 0:
                return False
        return True

    def inpute_inflow(self, ix, start_ix, start_value):
        stop_ix = ix - 1
        # TODO: Check for 23 may between 15:00 and 18:00
        if not self.possibility_of_heavy_inflow(start_ix, stop_ix):
            stop_value = self.all_measurements.iloc[ix].inflow
            increment_per_step = (stop_value - start_value) / (stop_ix - start_ix + 1)
            previous_date = None
            for ix, (date, _) in enumerate(self.all_measurements.iloc[start_ix:stop_ix + 1].iterrows()):
                if ix == 0:
                    self.all_measurements.at[date, 'inflow'] = start_value
                else:
                    mes_prev = self.all_measurements.loc[previous_date].inflow
                    imputed = mes_prev + increment_per_step
                    self.all_measurements.loc[date, 'inflow'] = imputed
                previous_date = date
        else:
            # When we heave heavy inflow, we want to keep the old inflow value
            pass

    def volume_to_meters_cylinder(self, volume):
        return volume / (math.pi * 0.564189584 ** 2 * 1000)

    # value t is threshold in meters. PST238 has threshold of 1 meter before a pump turns on.
    def flexibility_label(self, df, t=1):
        result = []
        for i in range(len(df)):
            if df.iloc[i].outflow_level > 0:
                catchment = 0
                water_level = df.iloc[i].water_level
                for j in range(len(df.iloc[i:])):
                    inflow = df.iloc[j + i].inflow if df.iloc[j + i].inflow > 0 else 0
                    catchment += inflow * self.time_interval
                    if self.volume_to_meters_cylinder(catchment) + water_level >= t or j == len(df.iloc[i:]) - 1:
                        result.append(j)
                        break
            else:
                result.append(0)

        return result