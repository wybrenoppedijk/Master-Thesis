from src.model.Model import Model
from src.pumping_station_enum import PUMPING_STATION_ENUM as ps
from multiprocessing import cpu_count

# Number of threads to use for parsing
NR_THREADS = cpu_count() * 2

# Time interval (in seconds) between samples. 'None' means no time interpolation
TIME_INTERVAL_S = 120

# Data Files
PATH_HIST = "../data/HistoricData"
PATH_PUMP_LOCATION = "../data/pump_locations.csv"
PATH_PUMP_INFO = "../data/PST Pump powers and volumes.xlsx"
PATH_CLEAN_WATER = "../data/VS__rapporter_Brogaard_VV_"
PATH_PUMP_GAIN = "../data/pump_gains.csv"

# What to include
PUMPING_STATIONS = [ps.PST232, ps.PST233, ps.PST234, ps.PST237, ps.PST238, ps.PST239, ps.PST240]
# PUMPING_STATIONS = [ps.PST232]
INCLUDE_DATA_VALIDATION = False  # Takes long time
INCLUDE_WEATHER_DATA = True
INCLUDE_WATER_CONSUMPTION = True
REMOVE_INVALID_READINGS = False


def load_data():
    model = Model(
        PUMPING_STATIONS,
        PATH_HIST,
        PATH_PUMP_LOCATION,
        PATH_PUMP_INFO,
        PATH_CLEAN_WATER,
        PATH_PUMP_GAIN,
        TIME_INTERVAL_S,
        INCLUDE_DATA_VALIDATION,
        INCLUDE_WEATHER_DATA,
        INCLUDE_WATER_CONSUMPTION,
        NR_THREADS,
        REMOVE_INVALID_READINGS
    )

    # Save outputs
    to_save = model.all_measurements.sort_index()
    to_save.to_pickle(f'../output/120s-all.pkl', compression='gzip')
    return model


if __name__ == '__main__':
    model = load_data()
    print("Done")