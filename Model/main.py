from model.Model import Model
from pumping_station_enum import PUMPING_STATION_ENUM as ps


NR_THREADS = 8 * 2
TIME_INTERVAL_S = 5
PATH_HIST = "../data/HistoricData"
PATH_PUMP_LOCATION = "../data/pump_locations.csv"
PATH_PUMP_INFO = "../data/PST Pump powers and volumes.xlsx"
PATH_CLEAN_WATER = "../data/VS__rapporter_Brogaard_VV_"
PATH_PUMP_GAIN = "../data/pump_gains.csv"
PUMPING_STATIONS = [ps.PST232]
# PUMPING_STATIONS = [ps.PST240]
INCLUDE_WEATHER_DATA = True
INCLUDE_WATER_CONSUMPTION = False
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
        INCLUDE_WEATHER_DATA,
        INCLUDE_WATER_CONSUMPTION,
        NR_THREADS,
        REMOVE_INVALID_READINGS
    )

    return model


if __name__ == '__main__':
    model = load_data()
    model.all_measurements.to_csv('../data/all_measurements.csv')
    print("Done")