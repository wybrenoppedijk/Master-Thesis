import numpy as np
import log
import pandas as pd
import math
from src.model.PumpingStation import PumpingStation
from src.pumping_station_enum import PUMPING_STATION_ENUM as ps
pd.options.mode.chained_assignment = None

"""
What is the data validator?
- Validates each measurement in the dataframe
- Get new dataframe with the validated data and the validation results
- Adds cycle information for each measurement
- Repairs the dataframe if needed

What is not working (yet):
- The thresholds in 'validation_properties' are inaccurate for some pump stations
- Incorrect outflow is detected but not repaired

"""
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


def validate(df, pumping_station: PumpingStation, apply_data_corrections: bool):
    ps_name = pumping_station.name
    p = pumping_station
    # Custom rules for stations with 3 pumps
    p3 = True if ps_name == ps.PST240 else False

    # Modified dataframe optimized for validation
    dfc = df.copy()
    if p3:
        dfc["current_tot"] = dfc.apply(lambda row: row.current_1 + row.current_2 + row.current_3, axis=1)
    else:
        dfc["current_tot"] = dfc.apply(lambda row: row.current_1 + row.current_2, axis=1)

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
        return c > p.current_pump_on

    def flowing_current_tot(measurement):
        return True if (
                flowing_current(measurement.current_1) or
                flowing_current(measurement.current_2) or
                (p3 and flowing_current(measurement.current_3))
                ) else False

    def flowing_outflow(o):
        return o > p.outflow_pump_on

    def current_acceptable(c, pump_nr):
        if pump_nr == 1:
            return p.current_p1[0] < c < p.current_p1[1]
        elif pump_nr == 2:
            return p.current_p2[0] < c < p.current_p2[1]
        elif pump_nr == 3:
            return p.current_p3[0] < c < p.current_p3[1]
        else:
            raise Exception("Pump number not supported")

    def outflow_acceptable(o, pump_config):
        if pump_config == "[P1]":
            return p.outflow_p1[0] < o < p.outflow_p1[1]
        elif pump_config == "[P2]":
            return p.outflow_p2[0] < o < p.outflow_p2[1]
        elif pump_config == "[P1+P2]":
            return p.outflow_p1_and_p2[0] < o < p.outflow_p1_and_p2[1]
        elif pump_config == "[P3]":
            return p.outflow_p3[0] < o < p.outflow_p3[1]
        else:
            raise Exception("Pump configuration not supported")

    def current_and_outflow_incorrect(m):
        p1_on, p2_on = flowing_current(m.current_1), flowing_current(m.current_2)
        if p3:
            p3_on = flowing_current(m.current_3)
        else:
            p3_on = False

        if not p1_on and not p2_on and not p3_on:  # No pumps are on
            return False  # In transition, no error
        elif p1_on and not p2_on and not p3_on:  # [P1]
            return not current_acceptable(m.current_1, 1) and not outflow_acceptable(m.outflow_level, "[P1]")
        elif p2_on and not p1_on and not p3_on:  # [P2]
            return not current_acceptable(m.current_2, 1) and not outflow_acceptable(m.outflow_level, "[P2]")
        elif p1_on and p2_on and not p3_on:  # [P1+P2]
            p1_incorrect = not current_acceptable(m.current_1, 1) and not outflow_acceptable(m.outflow_level, "[P1+P2]")
            p2_incorrect = not current_acceptable(m.current_2, 1) and not outflow_acceptable(m.outflow_level, "[P1+P2]")
            return p1_incorrect or p2_incorrect
        elif p3_on and not p1_on and not p2_on:  # [P3]
            return not current_acceptable(m.current_3, 2) and not outflow_acceptable(m.outflow_level, "[P3]")
        elif p1_on and p2_on and p3_on:  # [P1+P2+P3]
            return False    # In transition, no error
        elif p2_on and p3_on:  # [P2+P3]
            return False    # In transition, no error
        elif p1_on and p3_on:  # [P1+P3]
            return False    # In transition, no error
        else:
            raise Exception(f"Current and outflow validation failed, unsupported configuration: "
                            f"p1={p1_on}, p2={p2_on}, p3={p3_on}")

    # Start validating:
    if apply_data_corrections:
        log.update("Starting validation and apply data correction... (this takes a long time)")
    else:
        log.update("Starting validation... (this takes a short time)")

    for ix, (date, now) in enumerate(dfc.iterrows()):
        # Maintain length
        assert len(cycle_nrs) == ix, f"New column length not maintained. Stopped at {ix}-'{date}'"
        assert len(df) == len(dfc), f"Length of validation set not maintained. Stopped at {ix}-'{date}'"
        # Skip first 4 rows and last 4 rows  => We cannot apply validation on them.
        if (ix < 4) | (ix >= len(dfc) - 4):
            append(np.nan, np.nan, None, None)
            continue

        previous, previous_l4 = dfc.iloc[ix-1], dfc.iloc[ix-4]
        next, next_l4 = dfc.iloc[ix + 1], dfc.iloc[ix + 4]

        # Ignore measurements with no current and outflow
        # ==================================================================================
        if (not flowing_current_tot(now)) and (not flowing_outflow(now.outflow_level)):
            if flowing_current_tot(previous) and flowing_outflow(previous.outflow_level) and \
                    flowing_current_tot(next) and flowing_outflow(next.outflow_level):
                append_error("No current and outflow, but previous and next have current and outflow")
                if apply_data_corrections:  # take average between previous and next
                    df.at[date, "current_1"] = (previous.current_1 + next.current_1) / 2
                    df.at[date, "current_2"] = (previous.current_2 + next.current_2) / 2
                    if p3:
                        df.at[date, "current_3"] = (previous.current_3 + next.current_3) / 2
                    df.at[date, "current_tot"] = (previous.current_tot + next.current_tot) / 2
                    df.at[date, "outflow_level"] = (previous.outflow_level + next.outflow_level) / 2
                continue
            append(np.nan, np.nan, None, None)
            continue

        # Check for simple errors
        # ==================================================================================
        # IF pump is turned on for just one sample
        elif (flowing_current_tot(now)) and (not flowing_current_tot(previous)) and (not flowing_current_tot(next)):
            append_error("Pump is turned on for just one sample")
            if apply_data_corrections:
                df.loc[date, ['current_1', 'current_2']] = 0
            continue
        # IF both pumps have current between 0.0 and 0.5
        elif (flowing_current(now.current_1) and (now.current_1 < 0.5)) and (flowing_current(now.current_2) and (now.current_2 < 0.5)):
            append_error("Leakage Current on both [P1] and [P2]") # Check both at same time to improve performance
            if apply_data_corrections:
                df.loc[date, ['current_1', 'current_2']] = 0
            continue
        # IF pump 1 has current between 0.0 and 0.5
        elif (flowing_current(now.current_1)) and (now.current_1 < 0.5):
            if apply_data_corrections:
                df.at[date, "current_1"] = 0
            append_error("Leakage Current on [P1]")
            continue
        # IF pump 2 has current between 0.0 and 0.5
        elif (flowing_current(now.current_2)) and (now.current_2 < 0.5):
            append_error("Leakage Current on [P2]")
            if apply_data_corrections:
                df.at[date, "current_2"] = 0
            continue
        # IF pump 1 has an unacceptable current
        elif (flowing_current(now.current_1)) and (not current_acceptable(now.current_1, 1)) and (not flowing_outflow(previous_l4.outflow_level)):
            # IF outflow is also incorrect:
            if current_and_outflow_incorrect(now):
                append_error("current_1 and outflow_level are not within boundaries")
                if apply_data_corrections:
                   df.loc[date, ['current_1', 'current_2', 'outflow_level']] = 0
                continue
            else:
                append_error("Current not within boundaries on [P1]")
                if apply_data_corrections:
                    df.at[date, "current_1"] = np.mean(p.current_p1)
            continue
        # IF pump 2 has an unacceptable current
        elif (flowing_current(now.current_2)) and (not current_acceptable(now.current_2, 1)) and (not flowing_outflow(previous_l4.outflow_level)):
            # IF outflow is also incorrect:
            if current_and_outflow_incorrect(now):
                append_error("current_2 and outflow_level are not within boundaries")
                if apply_data_corrections:
                   df.loc[date, ['current_1', 'current_2', 'outflow_level']] = 0
                continue
            else:
                append_error("Current not within boundaries on [P2]")
                if apply_data_corrections:
                    df.at[date, "current_1"] = np.mean(p.current_p1)
                continue
        elif p3 and (flowing_current(now.current_3)) and (not current_acceptable(now.current_3, 1)) and (not flowing_outflow(previous_l4.outflow_level)):
            append_error("Current not within boundaries on [P3]")
            if apply_data_corrections:
                df.at[date, "current_1"] = np.mean(p.current_p1)
            continue

        # Check for State Changes:
        # ==================================================================================
        # Transition:       [][P1], [][P2], [][P1+P2], [P1][P1+P2], [P2][P1+P2],  # CURRENT INCREASE
        # Transition P3:    [P1,P2][P3] or [P3][P3] (power increase)
        elif flowing_current_tot(now) and (now.current_tot - previous.current_tot > p.current_change_threshold):
            if not flowing_outflow(next_l4.outflow_level):
                append_error("Outflow does not start after 4 samples")
                for ix2, (date2, now2) in enumerate(df.iloc[ix:ix+4].iterrows()):
                    water_level_diff = now.water_level - df.iloc[ix + ix2 - 2].water_level
                    if apply_data_corrections and (water_level_diff < 0):
                        df.at[date2, "outflow_level"] = calc_outflow(now2, pumping_station)
                continue
            if (flowing_current(previous.current_1)) and (flowing_current(previous.current_2)):
                append_error("Current increased, but both pumps were already on")
                continue
            # Transition:  [][P1], [][P2], [][P1+P2], [][P3]
            if not (flowing_current_tot(previous)) or (not flowing_current_tot(previous_l4)):
                if flowing_outflow(now.outflow_level) and (not flowing_current_tot(previous)):
                    append_error("Expected delay in outflow not found")
                    continue
                if not (flowing_current(now.current_1) | flowing_current(now.current_2)):
                    append_error("Increase in current, but pumps are disabled")
                    continue
                # Transition:  [][P1]
                if (flowing_current(now.current_1)) and (not flowing_current(now.current_2)):
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1]", "[][P1]")
                    continue
                # Transition:  [][P2]
                elif (flowing_current(now.current_2)) and (not flowing_current(now.current_1)):
                    next_cycle()
                    append(cycle_count, cycle_step, "[P2]", "[][P2]")
                    continue
                # Transition:  [][P1+P2]
                elif (flowing_current(now.current_2)) and (flowing_current(now.current_1)):
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1,P2]", "[][P1,P2]")
                    continue
                elif p3 and (flowing_current(now.current_3)) and (not flowing_current(now.current_1)) and (not flowing_current(now.current_2)):
                    next_cycle()
                    append(cycle_count, cycle_step, "[P3]", "[][P3]")
                    continue
                else:
                    raise DataValidationError(date, ps_name)

            # Transition:  [P1][P1+P2], [P2][P1+P2]
            elif (flowing_current(now.current_1)) | (flowing_current(now.current_2)):
                both_draw_current = flowing_current(now.current_1) and flowing_current(previous.current_2)
                if not both_draw_current:
                    append_error("Current increased but a second pump is not activated")
                    continue
                if now.outflow_level - previous.outflow_level > p.outflow_change_threshold:
                    append_error("Expected delay in outflow not found")
                    continue
                # Transition:  [P1][P1+P2]
                if (flowing_current(previous.current_1)) and (not flowing_current(previous.current_2)):
                    if not outflow_acceptable(now.outflow_level, "[P1]"):
                        append_error("Outflow is not within boundaries [P1]")
                        if apply_data_corrections:
                            df.at[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                        continue
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1,P2]", "[P1][P1,P2]")
                    continue
                # Transition:  [P2][P1+P2]
                elif (flowing_current(previous.current_2)) and (not flowing_current(previous.current_1)):
                    if not outflow_acceptable(now.outflow_level, "[P2]"):
                        append_error("Outflow is not within boundaries [P2]")
                        if apply_data_corrections:
                            df.at[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                        continue
                    next_cycle()
                    append(cycle_count, cycle_step, "[P1,P2]", "[P2][P1,P2]")
                    continue
                else:
                    raise DataValidationError(date, ps_name)
            # Transition P3:    [P3][P3] (power increase)
            elif (flowing_current(now.current_3)) and (not flowing_current(now.current_1)) and (not flowing_current(now.current_2)):
                if not (flowing_outflow(now.outflow_level)):
                    append_error("Expected outflow not found")
                    continue
                next_cycle()
                append(cycle_count, cycle_step, "[P3]", "[P3][P3]")
                continue
            else:
                raise DataValidationError(date, ps_name)
        # Transition:  [P1+P2][P1], [P1+P2][P2]                     # CURRENT DECREASE
        elif (not p3 or not flowing_current(now.current_3)) and flowing_current_tot(now) and (previous.current_tot - now.current_tot > p.current_change_threshold) \
                and (flowing_current(previous_l4.current_1)) and (flowing_current(previous_l4.current_2)):
            if flowing_current(now.current_1) and flowing_current(now.current_2):
                append_error("Current decreased but both pumps still in operation")
                continue
            # Transition:  [P1+P2][P1]
            if (flowing_current(now.current_1)) and (not flowing_current(now.current_2)):
                next_cycle()
                append(cycle_count, cycle_step, "[P1]", "[P1,P2][P1]")
                continue
            # Transition:  [P1+P2][P1]
            elif (flowing_current(now.current_2)) and (not flowing_current(now.current_1)):
                next_cycle()
                append(cycle_count, cycle_step, "[P2]", "[P1,P2][P2]")
                continue
            else:
                raise DataValidationError(date, ps_name)
        # Transition: [P1,P2][P3] ,
        elif (p3 and (flowing_current(now.current_3) and (flowing_current(now.current_1) | flowing_current(now.current_2)))) or \
                (p3 and ((flowing_current(now.current_3) or flowing_current(previous.current_2)) and flowing_current(previous.current_2))) or \
                (p3 and ((flowing_current(now.current_3) or flowing_current(dfc.iloc[ix-2].current_2)) and flowing_current(dfc.iloc[ix-2].current_2)) )or \
                (p3 and ((flowing_current(now.current_3) or flowing_current(dfc.iloc[ix-3].current_2)) and flowing_current(dfc.iloc[ix-3].current_2))) or \
                (p3 and ((flowing_current(now.current_3) or flowing_current(previous_l4.current_2)) and flowing_current(previous_l4.current_2))):
            # Transition: [P1, P2][P3]  (First Part of transition)
            if flowing_current(now.current_3) and (flowing_current(now.current_1) | flowing_current(now.current_2)):
                if not (flowing_outflow(now.outflow_level) < flowing_outflow(previous_l4.outflow_level)):
                    append_error("Expected outflow to decrease during Transition [P1,P2][P3]")
                    continue
                if not flowing_outflow(now.outflow_level):
                    append_error("Expected outflow during Transition [P1,P2][P3]")
                    continue
                if flowing_current(previous_l4.current_3) and (flowing_current(previous_l4.current_1) | flowing_current(previous_l4.current_2)):
                    append_error("[P3] is running alongside another pump for a long time")
                    continue
                append(cycle_count, cycle_step, "[P1,P2,P3]", "[P1,P2][P3]")
                continue
            else:
                append(cycle_count, cycle_step, "[P3]", "[P1,P2][P3]")
                continue
        # Transition: [P3][P3] (power decrease), [P3][]
        elif (p3 and (flowing_current(now.current_3)) and flowing_current_tot(now) and (previous.current_tot - now.current_tot > p.current_change_threshold)) or \
                (p3 and (flowing_outflow(now.outflow_level) and (not flowing_current(now.current_3)) and (flowing_current(previous_l4.current_3)))) or \
                 (p3 and (flowing_outflow(now.outflow_level) and (flowing_current(now.current_3)) and (not flowing_current(next_l4.current_3)))):
            # Transition [P3][]
            if flowing_outflow(now.outflow_level) and (not flowing_current(now.current_3)) and (flowing_current(previous_l4.current_3)):
                append(cycle_count, cycle_step, "[]", "[P3][]")
                continue
            # Transition [P3][]
            elif flowing_outflow(now.outflow_level) and (flowing_current(now.current_3)) and (not flowing_current(next_l4.current_3)):
                append(cycle_count, cycle_step, "[P3]", "[P3][]")
                continue
            # Transition: [P3][P3] (power decrease)
            elif (flowing_current(now.current_3)) and flowing_current_tot(now) and (previous.current_tot - now.current_tot > p.current_change_threshold) :
                append(cycle_count, cycle_step, "[P3]", "[P3][P3]")
                continue
            else:
                raise DataValidationError(date, ps_name)
        # Transition:  [P1][], [P2][], [P1+P2][]
        #   -> No flowing current, or reduced current that is about to end.
        elif not flowing_current_tot(now) or \
                ((abs(previous_l4.current_tot - now.current_tot) > p.current_change_threshold) and not (flowing_current_tot(next_l4))):
            # Transition:  [P1][], [P2][], [P1+P2][]
            if flowing_outflow(now.outflow_level):
                if not flowing_current_tot(previous_l4):
                    append_error("Pump is emitting water but operation stopped long time ago")
                    if apply_data_corrections:
                        df.at[date, 'outflow_level'] = 0
                    continue
                # Transition:  [P1][]
                if (flowing_current(previous_l4.current_1)) and (not flowing_current(previous_l4.current_2)):
                    append(cycle_count, cycle_step, "[P1]", "[P1][]")
                    continue
                # Transition:  [P2][]
                elif (flowing_current(previous_l4.current_2)) and (not flowing_current(previous_l4.current_1)):
                    append(cycle_count, cycle_step, "[P2]", "[P2][]")
                    continue
                # Transition:  [P1+P2][]
                elif (flowing_current(previous_l4.current_1)) and (flowing_current(previous_l4.current_2)):
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
        # No Transition: Stable on P1, P2, [P1,P2] or P3
        elif flowing_current_tot(now):  # CURRENT STABLE ON
            if not abs(previous.current_tot - now.current_tot) <= p.current_change_threshold:
                append_error("Current is fluctuating")
                continue
            if (not flowing_outflow(now.outflow_level)) and (flowing_current_tot(previous_l4)):
                append_error("Pump(s) are running but no outflow, current started 4 samples ago")
                if apply_data_corrections:
                    df.iloc[ix].outflow_level = calc_outflow(df.iloc[ix], pumping_station)
                continue
            # Stable on [P1]
            if (flowing_current(now.current_1)) and (not flowing_current(now.current_2)):
                if not ((flowing_current(previous.current_1)) and (not flowing_current(previous.current_2))):
                    append_error("Unexpected change of selection of enabled pumps (1)")
                    continue
                if (not outflow_acceptable(now.outflow_level, "[P1]")) and (flowing_current(previous_l4.current_1)):
                    append_error("Outflow is not within boundaries [P1] (1)")
                    if apply_data_corrections:
                        df.at[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                append(cycle_count, cycle_step, "[P1]", None)
            # Stable on [P2]
            elif (flowing_current(now.current_2)) and (not flowing_current(now.current_1)):
                if not ((flowing_current(previous.current_2)) and (not flowing_current(previous.current_1))):
                    append_error("Unexpected change of selection of enabled pumps (2)")
                    if apply_data_corrections:
                        df.at[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                if (not outflow_acceptable(now.outflow_level, "[P2]")) and (flowing_current(previous_l4.current_2)):
                    append_error("Outflow is not within boundaries [P2] (2)")
                    if apply_data_corrections:
                        df.at[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                append(cycle_count, cycle_step, "[P2]", None)
            # Stable on [P1+P2]
            elif (flowing_current(now.current_1)) and (flowing_current(now.current_2)):
                if not ((flowing_current(previous.current_2)) and (flowing_current(previous.current_1))):
                    append_error("Unexpected change of selection of enabled pumps (3)")
                    continue
                if (not outflow_acceptable(now.outflow_level, "[P1+P2]")) and (flowing_current_tot(previous_l4)):
                    append_error("Outflow is not within boundaries [P1+P2] (3)")
                    if apply_data_corrections:
                        df.at[date, "outflow_level"] = calc_outflow(df.iloc[ix], pumping_station)
                    continue
                append(cycle_count, cycle_step, "[P1,P2]", None)
                continue
            elif (flowing_current(now.current_3)) and (not flowing_current(now.current_1)) and (not flowing_current(now.current_2)):
                if not ((flowing_current(previous.current_3)) and (not flowing_current(previous.current_1)) and (not flowing_current(previous.current_2))):
                    append_error("Unexpected change of selection of enabled pumps (4)")
                    continue
                append(cycle_count, cycle_step, "[P3]", None)
                continue
            else:
                raise DataValidationError(date, ps_name)
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
