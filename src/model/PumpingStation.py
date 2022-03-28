import pandas as pd
from src.pumping_station_enum import PUMPING_STATION_ENUM as PS
class PumpingStation:
    def __init__(self, name: PS, loc: list[float]):
        self.name:PS = name
        self.lat = loc[0]
        self.lon = loc[1]
        self.description = ''
        self.address = ''
        self.volume = 0
        self.gain = []
        self.pumping_stations_upstream = []
        self.pumping_stations_downstream = []
        self.pumps = []
        self.measurements = pd.DataFrame(
            columns=[
                "timestamp",
                "currents",
                "current_tot",
                "outflow_level",
                "water_level",
            ],
        )
        # Data validator values:
        self.current_pump_on = 0            # in Amperes
        self.current_change_threshold = 0   # in Amperes
        self.current_p1 = []                # in Amperes range
        self.current_p2 = []                # in Amperes range
        self.current_p3 = []                # in Amperes range
        self.outflow_pump_on = 0            # in m3/s
        self.outflow_change_threshold = 0   # in m3/s
        self.outflow_p1 = []                # in m3/s range
        self.outflow_p2 = []                # in m3/s range
        self.outflow_p1_and_p2 = []         # in m3/s range
        self.outflow_p3 = []                # in m3/s range
