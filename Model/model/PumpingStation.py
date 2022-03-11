import pandas as pd

class PumpingStation:
    def __init__(self, name: str, loc: list[float]):
        self.name = name
        self.location = loc
        self.pumping_stations_upstream = []
        self.pumping_stations_downstream = []
        self.pumps = []
        self.measurements = pd.DataFrame(
            columns=['timestamp', 'currents', 'current_tot', 'outflow_level', 'water_level'],
        )

