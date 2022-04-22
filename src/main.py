from src.model.Model import Model
from src.pumping_station_enum import PUMPING_STATION_ENUM as ps
from multiprocessing import cpu_count
from time import time

# Number of threads to use for parsing
NR_THREADS = 8

# Time interval (in seconds) between samples. 'None' means no time interpolation
TIME_INTERVAL_S = None

# Data Files
PATH_HIST = "../data/HistoricData"
PATH_PUMP_LOCATION = "../data/pump_locations.csv"
PATH_PUMP_INFO = "../data/PST Pump powers and volumes.xlsx"
PATH_CLEAN_WATER = "../data/VS__rapporter_Brogaard_VV_"
PATH_PUMP_GAIN = "../data/pump_gains.csv"
PATH_VALIDATION_PROPS = "../data/validation_properties.csv"
PATH_SEA_LEVEL = "../data/ocean_data.json"

# What to include
# PUMPING_STATIONS = [ps.PST232, ps.PST233, ps.PST234, ps.PST237, ps.PST238, ps.PST239, ps.PST240]
PUMPING_STATIONS = [ps.PST238]
INCLUDE_DATA_VALIDATION = True  # Takes long time
INCLUDE_WEATHER_DATA = False
INCLUDE_WATER_CONSUMPTION = False
INCLUDE_SEA_LEVEL = False
APPLY_DATA_CORRECTIONS = True  # Takes really long time


def load_data():
    model = Model(
        PUMPING_STATIONS,
        PATH_HIST,
        PATH_PUMP_LOCATION,
        PATH_PUMP_INFO,
        PATH_CLEAN_WATER,
        PATH_PUMP_GAIN,
        PATH_VALIDATION_PROPS,
        PATH_SEA_LEVEL,
        TIME_INTERVAL_S,
        INCLUDE_DATA_VALIDATION,
        INCLUDE_WEATHER_DATA,
        INCLUDE_WATER_CONSUMPTION,
        INCLUDE_SEA_LEVEL,
        APPLY_DATA_CORRECTIONS,
        NR_THREADS,
    )

    # Save outputs
    to_save = model.all_measurements.sort_index()
    print("Saving outputs...")
    to_save.to_pickle(f'../output/238-va-co3.pkl', compression='gzip')
    return model


if __name__ == '__main__':
    start_time = time()
    model = load_data()
    print(f"Done in {time() - start_time} seconds")