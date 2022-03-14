from multiprocessing import cpu_count

from model.Model import Model
from pumping_station_enum import PUMPING_STATION_ENUM as n


NR_THREADS = cpu_count() * 2

TIME_INTERVAL_S = 3600
PATH_HIST = '../data/HistoricData'
PATH_PUMP_INFO = '../data/pump_locations.csv'
PATH_CLEAN_WATER = '../data/VS__rapporter_Brogaard_VV_'
# PUMPING_STATIONS = [n.PST232,n.PST233,n.PST234,n.PST237,n.PST238,n.PST239]
PUMPING_STATIONS = [n.PST234]
NR_THREADS = cpu_count() * 2    # number of threads to use
TIME_INTERVAL_S = 3600
PATH_HIST = "../data/HistoricData"
PATH_PUMP_INFO = "../data/pump_locations.csv"
PATH_CLEAN_WATER = "../data/VS__rapporter_Brogaard_VV_"
PUMPING_STATIONS = [n.PST232, n.PST233, n.PST234, n.PST237, n.PST238, n.PST239, n.PST240]
PUMPING_STATIONS = [n.PST238]


def load_data():
    model = Model(
        PUMPING_STATIONS,
        PATH_HIST,
        PATH_PUMP_INFO,
        PATH_CLEAN_WATER,
        TIME_INTERVAL_S,
        NR_THREADS,
    )
    return model


if __name__ == '__main__':
    model = load_data()
    model.all_measurements.to_csv('../data/all_measurements.csv')
    print("Done")