import pandas as pd
from pumping_station_enum import PUMPING_STATION_ENUM as PS
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
