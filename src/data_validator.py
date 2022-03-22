import numpy as np
from tqdm import tqdm
import log
import pandas as pd
import math
from src.model.PumpingStation import PumpingStation
from src.pumping_station_enum import PUMPING_STATION_ENUM as ps
pd.options.mode.chained_assignment = None

class DataValidationError(Exception):
    def __init__(self, idx, name):
        super().__init__(f"Reached an unknown state in the DataValidator. See sample '{idx}' in '{name}'")


validation_supported = [
    ps.PST232,
    ps.PST233,
    ps.PST234,
    ps.PST237,
    ps.PST238,
    ps.PST239,
    # ps.PST240, # Systems with 3 pumps are not yet supported
]


def assert_validation_supported(pumping_station):
    if pumping_station in validation_supported:
        pass
    else:
        raise Exception(f"Validation is not supported for '{pumping_station.name}'")


def validate(df: pd.DataFrame, pumping_station: PumpingStation):
    assert_validation_supported(pumping_station.name)

    # Modified dataframe optimized for validation
    dfc = df.copy()
    dfc["current_tot"] = dfc.apply(lambda row: row.current_1 + row.current_2, axis=1)
    ps_name = pumping_station.name.name

    # Parameters
    current_tolerance = pumping_station.current_tolerance
    current_change_threshold = pumping_station.current_change_threshold
    current_expected_range = pumping_station.current_expected_range
    outflow_change_threshold = pumping_station.outflow_change_threshold
    outflow_tolerance = pumping_station.outflow_tolerance
    outflow_expected_single_p = pumping_station.outflow_expected_single_p
    outflow_expected_double_p = pumping_station.outflow_expected_double_p

    # Fields added as column to DF
    cycle_nrs = []  # Which cycle number. NaN means no cycle
    cycle_steps = []  # How far are we progressed in current cycle
    cycle_states = []  # [P1], [P2], or [P1+P2]
    cycle_transitions = []  # [][P1], [][P2], [][P1+P2], [P1][P1+P2], [P2][P1+P2], [P1+P2][P1], [P1+P2][P2], [P1][], [P2][], [P1+P2][],
    errors = []

    global cycle_step
    global cycle_count
    cycle_count = 0
    cycle_step = 0

    def append(cycle_nr, cycle_step_l, cycle_state, cycle_transition):
        cycle_nrs.append(cycle_nr)
        cycle_steps.append(cycle_step_l)
        cycle_states.append(cycle_state)
        cycle_transitions.append(cycle_transition)
        errors.append(None)
        global cycle_step
        cycle_step += 1

    def next_cycle():
        global cycle_count
        global cycle_step
        cycle_count += 1
        cycle_step = 0

    def append_error(e):
        cycle_nrs.append(np.nan)
        cycle_steps.append(np.nan)
        cycle_states.append(None)
        cycle_transitions.append(None)
        errors.append(e)
        global cycle_step
        cycle_step += 1
        next_cycle()  # Start new cycle after each error

    def flowing_current(c):
        return c > current_tolerance

    def flowing_outflow(o):
        return o > outflow_tolerance

    def current_acceptable(c):
        return current_expected_range[0] <= c <= current_expected_range[1]

    # Start validating:
    log.update("Starting validation... (this takes a long time)")
    for ix, (date, now) in enumerate(dfc.iterrows()):
        # Maintain length
        assert len(cycle_nrs) == ix, f"New column length not maintained. Stopped at {ix}-'{date}'"

        # Skip first 4 rows and last 4 rows  => We cannot apply validation on them.
        if (ix < 4) | (ix >= len(dfc) - 4):
            append(np.nan, np.nan, None, None)
            continue
        previous, previous_l4 = dfc.iloc[ix - 1], dfc.iloc[ix - 4]
        next, next_l4 = dfc.iloc[ix + 1], dfc.iloc[ix + 4]

        # No outflow and motors are disabled
        if (now.current_tot == 0) & (now.outflow_level == 0):
            append(np.nan, np.nan, None, None)
            continue

        # Check if current is in range
        elif (flowing_current(now.current_1)) & (not current_acceptable(now.current_1)):
            append_error("Current is not within boundaries [P1]")
            continue
        elif (flowing_current(now.current_2)) & (not current_acceptable(now.current_2)):
            append_error("Current is not within boundaries [P2]")
            continue

        # Check for State Changes:
        # ==================================================================================
        # Transition:  [][P1], [][P2], [][P1+P2], [P1][P1+P2], [P2][P1+P2],  # CURRENT INCREASE
        elif flowing_current(now.current_tot) & (now.current_tot - previous.current_tot > current_change_threshold):
            if not flowing_outflow(next_l4.outflow_level):
                append_error("Outflow does not start after 4 samples")
                for ix2, (date, now) in enumerate(df.iloc[ix:ix+4].iterrows()):
                    water_level_diff = now.water_level - df.iloc[ix + ix2 - 1].water_level
                    if water_level_diff < 0:
                        df.loc[date, "outflow_level"] = calc_outflow(now, pumping_station)
                continue
            if (flowing_current(previous.current_1)) & (flowing_current(previous.current_2)):
                append_error("Current increased, but both pumps were already on")
                continue
            # Transition:  [][P1], [][P2], [][P1+P2]
            if not flowing_current(previous.current_tot):
                if flowing_outflow(now.outflow_level):
                    append_error("Expected delay in outflow not found")
                    continue
                if not (flowing_current(now.current_1) | flowing_current(now.current_2)):
                    append_error("Increase in current, but pumps are disabled")
                    continue
                # Transition:  [][P1]
                if (flowing_current(now.current_1)) & (not flowing_current(now.current_2)):
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1]", "[][P1]")
                    continue
                # Transition:  [][P2]
                elif (flowing_current(now.current_2)) & (not flowing_current(now.current_1)):
                    next_cycle()
                    append(cycle_count, cycle_step, "[P2]", "[][P2]")
                    continue
                # Transition:  [][P1+P2]
                elif (flowing_current(now.current_2)) & (flowing_current(now.current_1)):
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1,P2]", "[][P1,P2]")
                    continue
                else:
                    raise DataValidationError(date, ps_name)

            # Transition:  [P1][P1+P2], [P2][P1+P2]
            else:
                both_draw_current = flowing_current(now.current_1) & flowing_current(previous.current_2)
                if not both_draw_current:
                    append_error("Current increased but a second pump is not activated")
                    continue
                if now.outflow_level - previous.outflow_level > outflow_change_threshold:
                    append_error("Expected delay in outflow not found")
                    continue
                # Transition:  [P1][P1+P2]
                if (flowing_current(previous.current_1)) & (not flowing_current(previous.current_2)):
                    if now.outflow_level < outflow_expected_single_p:
                        append_error("Outflow level is too low (1)")
                        continue
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1,P2]", "[P1][P1,P2]")
                    continue
                # Transition:  [P2][P1+P2]
                elif (flowing_current(previous.current_2)) & (not flowing_current(previous.current_1)):
                    if now.outflow_level < outflow_expected_single_p:
                        append_error("Outflow level is too low (2)")
                        continue
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1,P2]", "[P2][P1,P2]")
                    continue
                else:
                    raise DataValidationError(date, ps_name)
        # Transition:  [P1+P2][P1], [P1+P2][P2]                     # CURRENT DECREASE
        elif flowing_current(now.current_tot) & (previous.current_tot - now.current_tot > current_change_threshold):
            if flowing_current(now.current_1) & flowing_current(now.current_2):
                append_error("Current decreased but both pumps still in operation")
                continue
            # Transition:  [P1+P2][P1]
            if (flowing_current(now.current_1)) & (not flowing_current(now.current_2)):
                next_cycle()
                append(cycle_count, cycle_step, "[P1]", "[P1,P2][P1]")
                continue
            # Transition:  [P1+P2][P1]
            elif (flowing_current(now.current_2)) & (not flowing_current(now.current_1)):
                next_cycle()
                append(cycle_count, cycle_step, "[P2]", "[P1,P2][P2]")
                continue
            else:
                raise DataValidationError(date, ps_name)
        # No Transition: Stable on P1, P2 or P1,P2
        elif flowing_current(now.current_tot):  # CURRENT STABLE ON
            if not abs(previous.current_tot - now.current_tot) <= current_change_threshold:
                append_error("Current is fluctuating")
                continue
            if (not flowing_outflow(now.outflow_level)) and (flowing_current(previous_l4.current_tot)):
                append_error("Pump(s) are running dry, current started 4 samples ago")
                df.iloc[ix].outflow_level = calc_outflow(df.iloc[ix], pumping_station)
                # df.iloc[ix] = calc_outflow(df.iloc[ix], pumping_station, watch_water_height=False)
                continue
            # if not next.water_level < now.water_level:
            #     append_error('Water level does not decrease while pumps are on for a while')
            #     continue
            if (flowing_current(now.current_1)) & (not flowing_current(now.current_2)):
                if not ((flowing_current(previous.current_1)) & (not flowing_current(previous.current_2))):
                    append_error("Unexpected change of selection of enabled pumps (1)")
                    continue
                if (now.outflow_level < outflow_expected_single_p) & (flowing_current(previous_l4.current_1)):
                    append_error("Outflow level is too low (3)")
                    continue
                append(cycle_count, cycle_step, "[P1]", None)
            elif (flowing_current(now.current_2)) & (not flowing_current(now.current_1)):
                if not ((flowing_current(now.current_2)) & (not flowing_current(now.current_1))):
                    append_error("Unexpected change of selection of enabled pumps (2)")
                    continue
                if (now.outflow_level < outflow_expected_single_p) & (flowing_current(previous_l4.current_2)):
                    append_error("Outflow level is too low (4)")
                    continue
                append(cycle_count, cycle_step, "[P2]", None)
            elif (flowing_current(now.current_2)) & (flowing_current(now.current_1)):
                if not ((flowing_current(now.current_2)) & (flowing_current(now.current_1))):
                    append_error("Unexpected change of selection of enabled pumps (3)")
                    continue
                if (now.outflow_level < outflow_expected_double_p) & (flowing_current(previous_l4.current_tot)):
                    append_error("Outflow level is too low (5)")
                    continue

                append(cycle_count, cycle_step, "[P1,P2]", None)
                continue
            else:
                raise DataValidationError(date, ps_name)
        # Transition:  [P1][], [P2][], [P1+P2][]
        elif not flowing_current(now.current_tot):
            # Transition:  [P1][], [P2][], [P1+P2][]
            if flowing_outflow(now.outflow_level):
                if not flowing_current(previous_l4.current_tot):
                    append_error("Pump is emitting water but operation stopped long time ago")
                    continue
                # Transition:  [P1][]
                if (flowing_current(previous_l4.current_1)) & (not flowing_current(previous_l4.current_2)):
                    append(cycle_count, cycle_step, "[P1]", "[P1][]")
                    continue
                # Transition:  [P2][]
                elif (flowing_current(previous_l4.current_2)) & (not flowing_current(previous_l4.current_1)):
                    append(cycle_count, cycle_step, "[P2]", "[P2][]")
                    continue
                # Transition:  [P1+P2][]
                elif (flowing_current(previous_l4.current_1)) & (flowing_current(previous_l4.current_2)):
                    append(cycle_count, cycle_step, "[P1,P2]", "[P1,P2][]")
                    continue
                else:
                    raise DataValidationError(date, ps_name)
            else:
                if flowing_outflow(previous.outflow_level):
                    cycle_count += 1
                    append(np.nan, np.nan, None, None)
                else:
                    if not next.water_level > now.water_level:
                        append_error("Water level should be rising")
                        continue
                    append(np.nan, np.nan, None, None)
                    continue
        else:
            raise DataValidationError(date, ps_name)
    log.update("Finished validation")

    # Set results
    df = df.assign(
        cycle_nr=cycle_nrs,
        cycle_step=cycle_steps,
        cycle_state=cycle_states,
        cycle_state_transition=cycle_transitions,
        error=errors,
    )
    df = df.astype({
        'cycle_nr': 'Int32',
        'cycle_step': 'Int32',
    })

    return df
