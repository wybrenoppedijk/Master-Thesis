from multiprocessing import cpu_count

from model.Model import Model
from pumping_station_enum import PUMPING_STATION_ENUM as ps


NR_THREADS = 8
TIME_INTERVAL_S = 3600
PATH_HIST = "data/HistoricData"
PATH_PUMP_INFO = "data/pump_locations.csv"
PATH_CLEAN_WATER = "data/VS__rapporter_Brogaard_VV_"
PUMPING_STATIONS = [ps.PST232, ps.PST233, ps.PST234, ps.PST237, ps.PST238, ps.PST239, ps.PST240]
INCLUDE_WEATHER_DATA = True
INCLUDE_WATER_CONSUMPTION = True


def load_data():
    model = Model(
        PUMPING_STATIONS,
        PATH_HIST,
        PATH_PUMP_INFO,
        PATH_CLEAN_WATER,
        TIME_INTERVAL_S,
        INCLUDE_WEATHER_DATA,
        INCLUDE_WATER_CONSUMPTION,
        NR_THREADS,
    )

    return model


if __name__ == '__main__':
    model = load_data()
    model.all_measurements.to_csv('../data/all_measurements.csv')
    print("Done")