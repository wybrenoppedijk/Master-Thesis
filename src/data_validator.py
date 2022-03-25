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
    ps.PST240,  # Systems with 3 pumps
]


def assert_validation_supported(pumping_station):
    if pumping_station in validation_supported:
        pass
    else:
        raise Exception(f"Validation is not supported for '{pumping_station.name}'")


def calc_outflow(df: pd.DataFrame, pump_station: PumpingStation):
        power_p1 = (math.sqrt(3) * df.current_1 * 400 / 1000)
        power_p2 = (math.sqrt(3) * df.current_2 * 400 / 1000)
        power_p3 = 0
        gain_p1 = pump_station.gain[0]
        gain_p2 = pump_station.gain[1]
        gain_p3 = 0
        if pump_station.name == ps.PST240:
            power_p3 = (math.sqrt(3) * df.current_3 * 400 / 1000)
            gain_p3 = pump_station.gain[2]
        return  (power_p1 * gain_p1 + power_p2 * gain_p2 + power_p3 * gain_p3) / 1000

def calc_current(df: pd.DataFrame, pump_station: PumpingStation):
    total = 0
    if 'current_1' in df:
        if df.current_1 < 0.5:
            df.current1 = 0
        else :
            total += 1
    if 'current_2' in df:
        if df.current_2 < 0.5:
            df.current2 = 0
        else :
            total += df.current_2
    if 'current_3' in df:
        if df.current_3 < 0.5:
            df.current3 = 0
        else :
            total += 1


def validate(df, pumping_station: PumpingStation):
    ps_name = pumping_station.name

    # Custom rules for stations with 3 pumps
    p3 = True if ps_name == ps.PST240 else False

    # Modified dataframe optimized for validation
    dfc = df.copy()
    if p3:
        dfc["current_tot"] = dfc.apply(lambda row: row.current_1 + row.current_2 + row.current_3, axis=1)
    else:
        dfc["current_tot"] = dfc.apply(lambda row: row.current_1 + row.current_2, axis=1)

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
    cycle_states = []  # [P1], [P2], or [P1+P2], [P1+P3], [P2+P3], [P1+P2+P3]

    # 2p: [][P1], [][P2], [][P1+P2], [P1][P1+P2], [P2][P1+P2], [P1+P2][P1], [P1+P2][P2], [P1][], [P2][], [P1+P2][],
    # 3p: [P1+P2][P3], [P3][]   |  P3 is only used as substitute for [P1,P2].
    cycle_transitions = []
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

    def flowing_current_tot(measurement):
        return True if (
                flowing_current(measurement.current_1) or
                flowing_current(measurement.current_2) or
                (p3 and flowing_current(measurement.current_3))
                ) else False

    def flowing_outflow(o):
        return o > outflow_tolerance

    def current_acceptable(c,p):
        return current_expected_range[p-1][0] <= c <= current_expected_range[p-1][1]

    def outflow_acceptable(o,num_pumps):
        if num_pumps == 1:
            return outflow_expected_single_p * (1 - outflow_tolerance) <= o <= outflow_expected_single_p * (1 + outflow_tolerance)
        if num_pumps == 2:
            return outflow_expected_double_p * (1 - outflow_tolerance) <= o <= outflow_expected_double_p  * (1 + outflow_tolerance)

    def current_and_outflow_incorrect(df):
        num_pumps = num_pumps_running(df)
        if num_pumps == 1:
            return not current_acceptable(df.current_1, 1) and not outflow_acceptable(df.outflow_level, 1)
        elif num_pumps == 2:
            current_acceptable(df.current_1, 1)
            return not current_acceptable(df.current_1, 2) and not outflow_acceptable(df.outflow_level, 2)

    def num_pumps_running(df):
        return int((df.current_1 > 0.5) + (df.current_2 > 0.5))

    # Start validating:
    log.update("Starting validation... (this takes a long time)")
    for ix, (date, now) in enumerate(dfc.iterrows()):
        # Maintain length
        assert len(cycle_nrs) == ix, f"New column length not maintained. Stopped at {ix}-'{date}'"

        # Skip first 4 rows and last 4 rows  => We cannot apply validation on them.
        if (ix < 4) | (ix >= len(dfc) - 4):
            append(np.nan, np.nan, None, None)
            continue
        previous, previous_l2, previous_l3, previous_l4 = dfc.iloc[ix-1], dfc.iloc[ix-2], dfc.iloc[ix-3], dfc.iloc[ix-4]
        next, next_l4 = dfc.iloc[ix + 1], dfc.iloc[ix + 4]

        # Ignore measurements with no current and outflow
        # ==================================================================================
        if (not flowing_current_tot(now)) & (not flowing_outflow(now.outflow_level)):
            append(np.nan, np.nan, None, None)
            continue

        # Check for simple errors
        # ==================================================================================
        elif (flowing_current(now.current_1)) & (not current_acceptable(now.current_1, 1)):
            if now.current_1 < 0.5:
                df.loc[date, "current_1"] = 0
                append_error("Leakage Current on p1 [P1]")
                continue
            if now.outflow_level == 0:
                df.loc[date, "current_1"] = 0
                append_error("Current on [P1] is too high and no outflow")
                continue
            elif current_and_outflow_incorrect(now):
                append_error("Both Current and Outflow are incorrect")
                df.loc[date, ['current_1', 'current_2', 'outflow_level']] = 0
                continue
            elif not current_acceptable(now.current_1, 1):
                append_error("Current not within boundaries on p1 [P1]")
                df.loc[date, "current_1"] = np.mean(current_expected_range[0])
                continue
            continue
        elif (flowing_current(now.current_2)) & (not current_acceptable(now.current_2, 2)):
            if now.current_1 < 0.5:
                df.loc[date, "current_2"] = 0
                append_error("Leakage Current on [P2]")
                continue
            if now.outflow_level == 0:
                df.loc[date, "current_2"] = 0
                append_error("Current on [P2] is too high and no outflow")
                continue
            elif not current_acceptable(now.current_2, 2):
                append_error("Current not within boundaries on p1 [P2]")
                df.loc[date, "current_2"] = np.mean(current_expected_range[1])
                continue
        elif p3 and (flowing_current(now.current_3)) & (not current_acceptable(now.current_3, 3)):
            append_error("Current is not within boundaries [P3]")
            continue

        # Check for State Changes:
        # ==================================================================================
        # Transition:       [][P1], [][P2], [][P1+P2], [P1][P1+P2], [P2][P1+P2],  # CURRENT INCREASE
        # Transition P3:    [P1,P2][P3] or [P3][P3] (power increase)
        elif flowing_current_tot(now) & (now.current_tot - previous.current_tot > current_change_threshold):
            if not flowing_outflow(next_l4.outflow_level):
                append_error("Outflow does not start after 4 samples")
                for ix2, (date, now) in enumerate(df.iloc[ix:ix+4].iterrows()):
                    water_level_diff = now.water_level - df.iloc[ix + ix2 - 2].water_level
                    if water_level_diff < 0:
                        df.loc[date, "outflow_level"] = calc_outflow(now, pumping_station)
                continue
            if (flowing_current(previous.current_1)) & (flowing_current(previous.current_2)):
                append_error("Current increased, but both pumps were already on")
                continue
            # Transition:  [][P1], [][P2], [][P1+P2], [][P3]
            if not flowing_current_tot(previous):
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
                elif p3 & (flowing_current(now.current_3)) & (not flowing_current(now.current_1)) & (not flowing_current(now.current_2)):
                    next_cycle()
                    append(cycle_count, cycle_step, "[P3]", "[][P3]")
                    continue
                else:
                    raise DataValidationError(date, ps_name)

            # Transition:  [P1][P1+P2], [P2][P1+P2]
            elif (flowing_current(now.current_1)) | (flowing_current(now.current_2)):
                both_draw_current = flowing_current(now.current_1) & flowing_current(previous.current_2)
                if not both_draw_current:
                    append_error("Current increased but a second pump is not activated")
                    continue
                if now.outflow_level - previous.outflow_level > outflow_change_threshold:
                    append_error("Expected delay in outflow not found")
                    continue
                # Transition:  [P1][P1+P2]
                if (flowing_current(previous.current_1)) & (not flowing_current(previous.current_2)):
                    if now.outflow_level < (1 - outflow_tolerance) * outflow_expected_single_p:
                        append_error("Outflow level is too low (1)")
                        df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                        continue
                    if now.outflow_level > (1 + outflow_tolerance ) * outflow_expected_single_p:
                        append_error("Outflow level is too high (1)")
                        df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                        continue
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1,P2]", "[P1][P1,P2]")
                    continue
                # Transition:  [P2][P1+P2]
                elif (flowing_current(previous.current_2)) & (not flowing_current(previous.current_1)):
                    if now.outflow_level < (1 - outflow_tolerance ) * outflow_expected_single_p:
                        append_error("Outflow level is too low (2)")
                        df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                        continue
                    if now.outflow_level > (1 + outflow_tolerance ) * outflow_expected_single_p:
                        append_error("Outflow level is too High (2)")
                        df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                        continue
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1,P2]", "[P2][P1,P2]")
                    continue
                else:
                    raise DataValidationError(date, ps_name)
            # Transition P3:    [P3][P3] (power increase)
            elif (flowing_current(now.current_3)) & (not flowing_current(now.current_1)) & (not flowing_current(now.current_2)):
                if not (flowing_outflow(now.outflow_level)):
                    append_error("Expected outflow not found")
                    continue
                next_cycle()
                append(cycle_count, cycle_step, "[P3]", "[P3][P3]")
                continue
            else:
                raise DataValidationError(date, ps_name)
        # Transition:  [P1+P2][P1], [P1+P2][P2]                     # CURRENT DECREASE
        elif (not p3 or not flowing_current(now.current_3)) & flowing_current_tot(now) & (previous.current_tot - now.current_tot > current_change_threshold):
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
        # Transition: [P1,P2][P3]
        elif (p3 and (flowing_current(now.current_3) & (flowing_current(now.current_1) | flowing_current(now.current_2)))) or \
                (p3 and ((flowing_current(now.current_3) | flowing_current(previous.current_2)) & flowing_current(previous.current_2))) or \
                (p3 and ((flowing_current(now.current_3) | flowing_current(previous_l2.current_2)) & flowing_current(previous_l2.current_2)) )or \
                (p3 and ((flowing_current(now.current_3) | flowing_current(previous_l3.current_2)) & flowing_current(previous_l3.current_2))) or \
                (p3 and ((flowing_current(now.current_3) | flowing_current(previous_l4.current_2)) & flowing_current(previous_l4.current_2))):
            # Transition: [P1, P2][P3]  (First Part of transition)
            if flowing_current(now.current_3) & (flowing_current(now.current_1) | flowing_current(now.current_2)):
                if not (flowing_outflow(now.outflow_level) < flowing_outflow(previous_l4.outflow_level)):
                    append_error("Expected outflow to decrease during Transition [P1,P2][P3]")
                    continue
                if not flowing_outflow(now.outflow_level):
                    append_error("Expected outflow during Transition [P1,P2][P3]")
                    continue
                if flowing_current(previous_l4.current_3) & (flowing_current(previous_l4.current_1) | flowing_current(previous_l4.current_2)):
                    append_error("[P3] is running alongside another pump for a long time")
                    continue
                append(cycle_count, cycle_step, "[P1,P2,P3]", "[P1,P2][P3]")
                continue
            else:
                append(cycle_count, cycle_step, "[P3]", "[P1,P2][P3]")
                continue
        # Transition: [P3][P3] (power decrease), [P3][]
        elif (p3 and (flowing_current(now.current_3)) & flowing_current_tot(now) & (previous.current_tot - now.current_tot > current_change_threshold)) or \
                (p3 and (flowing_outflow(now.outflow_level) & (not flowing_current(now.current_3)) & (flowing_current(previous_l4.current_3)))) or \
                 (p3 and (flowing_outflow(now.outflow_level) & (flowing_current(now.current_3)) & (not flowing_current(next_l4.current_3)))):
            # Transition [P3][]
            if flowing_outflow(now.outflow_level) & (not flowing_current(now.current_3)) & (flowing_current(previous_l4.current_3)):
                append(cycle_count, cycle_step, "[]", "[P3][]")
                continue
            # Transition [P3][]
            elif flowing_outflow(now.outflow_level) & (flowing_current(now.current_3)) & (not flowing_current(next_l4.current_3)):
                append(cycle_count, cycle_step, "[P3]", "[P3][]")
                continue
            # Transition: [P3][P3] (power decrease)
            elif (flowing_current(now.current_3)) & flowing_current_tot(now) & (previous.current_tot - now.current_tot > current_change_threshold) :
                append(cycle_count, cycle_step, "[P3]", "[P3][P3]")
                continue
            else:
                raise DataValidationError(date, ps_name)
        # No Transition: Stable on P1, P2 or P1,P2
        elif flowing_current_tot(now):  # CURRENT STABLE ON
            if not abs(previous.current_tot - now.current_tot) <= current_change_threshold:
                append_error("Current is fluctuating")
                continue
            if (not flowing_outflow(now.outflow_level)) and (flowing_current_tot(previous_l4)):
                append_error("Pump(s) are running but no outflow, current started 4 samples ago")
                df.iloc[ix].outflow_level = calc_outflow(df.iloc[ix], pumping_station)
                continue
            if (flowing_current(now.current_1)) & (not flowing_current(now.current_2)):
                if not ((flowing_current(previous.current_1)) & (not flowing_current(previous.current_2))):
                    append_error("Unexpected change of selection of enabled pumps (1)")
                    continue
                if (now.outflow_level < (1 - outflow_tolerance ) * outflow_expected_single_p) & (flowing_current(previous_l4.current_1)):
                    append_error("Outflow level is too low (3)")
                    df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                if (now.outflow_level > (1 + outflow_tolerance ) * outflow_expected_single_p) & (flowing_current(previous_l4.current_1)):
                    append_error("Outflow level is too high (3)")
                    df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                append(cycle_count, cycle_step, "[P1]", None)
            elif (flowing_current(now.current_2)) & (not flowing_current(now.current_1)):
                if not ((flowing_current(now.current_2)) & (not flowing_current(now.current_1))):
                    append_error("Unexpected change of selection of enabled pumps (2)")
                    df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                if (now.outflow_level < (1 - outflow_tolerance ) * outflow_expected_single_p) & (flowing_current(previous_l4.current_2)):
                    append_error("Outflow level is too low (4)")
                    df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                if (now.outflow_level > (1 + outflow_tolerance ) * outflow_expected_single_p) & (flowing_current(previous_l4.current_2)):
                    append_error("Outflow level is too high (4)")
                    df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                append(cycle_count, cycle_step, "[P2]", None)
            elif (flowing_current(now.current_2)) & (flowing_current(now.current_1)):
                if not ((flowing_current(now.current_2)) & (flowing_current(now.current_1))):
                    append_error("Unexpected change of selection of enabled pumps (3)")
                    continue
                if (now.outflow_level < (1 - outflow_tolerance ) * outflow_expected_double_p) & (flowing_current_tot(previous_l4)):
                    append_error("Outflow level is too low (5)")
                    df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                if (now.outflow_level > (1 + outflow_tolerance ) * outflow_expected_double_p) & (flowing_current(previous_l4.current_tot)):
                    append_error("Outflow level is too high (5)")
                    df.loc[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue

                append(cycle_count, cycle_step, "[P1,P2]", None)
                continue
            else:
                raise DataValidationError(date, ps_name)
        # Transition:  [P1][], [P2][], [P1+P2][]
        elif not flowing_current_tot(now):
            # Transition:  [P1][], [P2][], [P1+P2][]
            if flowing_outflow(now.outflow_level):
                if not flowing_current_tot(previous_l4):
                    append_error("Pump is emitting water but operation stopped long time ago")
                    df.loc[date, 'outflow_level'] = 0
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
                    continue
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
