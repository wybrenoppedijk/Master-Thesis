import numpy as np
from tqdm import tqdm
import log
from pumping_station_enum import PUMPING_STATION_ENUM as ps


class DataValidationError(Exception):
    def __init__(self, idx, name):
        super().__init__(f"Reached an unknown state in the DataValidator. See sample '{idx}' in '{name}'")


validation_supported = [
    ps.PST232
]


def assert_validation_supported(pumping_station):
    if pumping_station in validation_supported:
        pass
    else:
        raise Exception(f"Validation is not supported for '{pumping_station.name}'")


def validate(df, pumping_station: ps):
    assert_validation_supported(pumping_station)

    # Modified dataframe optimized for validation
    dfc = df.copy()
    dfc["current_tot"] = dfc.apply(lambda row: row.current_1 + row.current_2, axis=1)

    # Parameters
    current_tolerance = 0
    current_change_threshold = 1.5
    outflow_tolerance = 0

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

    def append_error(e):
        cycle_nrs.append(np.nan)
        cycle_steps.append(np.nan)
        cycle_states.append(None)
        cycle_transitions.append(None)
        errors.append(e)
        global cycle_step
        cycle_step += 1

    def flowing_current(c):
        return c > current_tolerance

    def flowing_outflow(o):
        return o > outflow_tolerance

    # Start validating:
    log.update("Starting validation... (this takes a long time)")
    for ix, (date, now) in enumerate(dfc.iterrows()):
        # Maintain length
        assert len(cycle_nrs) == ix, f"New column length not maintained. Stopped at '{date}'"

        # Skip first 2 rows and last row  => We cannot compare to previous rows
        if (ix < 2) | (ix >= len(dfc) - 3):
            append(np.nan, np.nan, "", "")
            continue
        previous, previous_l2, previous_l3 = dfc.iloc[ix - 1], dfc.iloc[ix - 2], dfc.iloc[ix - 3]
        next, next_l2, next_l3 = dfc.iloc[ix + 1], dfc.iloc[ix + 2], dfc.iloc[ix + 3]

        # No outflow and motors are disabled
        if (now.current_tot == 0) & (now.outflow_level == 0):
            append(np.nan, np.nan, "", "")

        # Check for State Changes:
        # ==================================================================================
        # Transition:  [][P1], [][P2], [][P1+P2], [P1][P1+P2], [P2][P1+P2],  # CURRENT INCREASE
        elif flowing_current(now.current_tot) & (now.current_tot - previous.current_tot > current_change_threshold):
            if flowing_outflow(now.outflow_level):
                append_error("Expected delay in outflow not found")
                continue
            if not flowing_outflow(next_l3.outflow_level):
                append_error("Outflow does not start after 3 samples")
                continue
            # Transition:  [][P1], [][P2], [][P1+P2]
            if not flowing_current(previous.current_tot):
                if not (flowing_current(now.current_1) | flowing_current(now.current_2)):
                    append_error("Increase in current, but pumps are disabled")
                    continue
                # Transition:  [][P1]
                if (flowing_current(now.current_1)) & (not flowing_current(now.current_2)):
                    append(cycle_count, cycle_step, "[P1]", "[][P1]")
                # Transition:  [][P2]
                elif (flowing_current(now.current_2)) & (not flowing_current(now.current_1)):
                    append(cycle_count, cycle_step, "[P2]", "[][P2]")
                # Transition:  [][P1+P2]
                elif (flowing_current(now.current_2)) & (flowing_current(now.current_1)):
                    append(cycle_count, cycle_step, "[P1,P2]", "[][P1,P2]")
                else:
                    raise DataValidationError(date, pumping_station.name)

            # Transition:  [P1][P1+P2], [P2][P1+P2]
            else:
                both_draw_current = flowing_current(now.current_1) & flowing_current(previous.current_2)
                if not both_draw_current:
                    append_error("Current increased but a second pump is not activated")
                    continue
                # Transition:  [P1][P1+P2]
                if (flowing_current(previous.current_1)) & (not flowing_current(previous.current_2)):
                    append(cycle_count, cycle_step, "[P1,P2]", "[P1][P1,P2]")
                # Transition:  [P2][P1+P2]
                elif (flowing_current(previous.current_2)) & (not flowing_current(previous.current_1)):
                    append(cycle_count, cycle_step, "[P1,P2]", "[P2][P1,P2]")
                else:
                    raise DataValidationError(date, pumping_station.name)
        # Transition:  [P1+P2][P1], [P1+P2][P2]                     # CURRENT DECREASE
        elif flowing_current(now.current_tot) & (previous.current_tot - now.current_tot > current_change_threshold):
            if flowing_current(now.current_1) & flowing_current(now.current_2):
                append_error("Current decreased but both pumps still in operation")
                continue
            # Transition:  [P1+P2][P1]
            if (flowing_current(now.current_1)) & (not flowing_current(now.current_2)):
                append(cycle_count, cycle_step, "[P1]", "[P1,P2][P1]")
            # Transition:  [P1+P2][P1]
            elif (flowing_current(now.current_2)) & (not flowing_current(now.current_1)):
                append(cycle_count, cycle_step, "[P2]", "[P1,P2][P2]")
            else:
                raise DataValidationError(date, pumping_station.name)
        # No Transition: Stable on P1, P2 or P1,P2
        elif flowing_current(now.current_tot):  # CURRENT STABLE ON
            if not abs(previous.current_tot - now.current_tot) <= current_change_threshold:
                append_error("Current is fluctuating")
                continue
            if not flowing_outflow(now.outflow_level):
                append_error("Pump(s) is running dry")
                continue
            if not next.water_level < now.water_level:
                append_error('Water level does not decrease while pumps are on for a while')
                continue
            if (flowing_current(now.current_1)) & (not flowing_current(now.current_2)):
                append(cycle_count, cycle_step, "[P1]", "")
            elif (flowing_current(now.current_2)) & (not flowing_current(now.current_1)):
                append(cycle_count, cycle_step, "[P2]", "")
            elif (flowing_current(now.current_2)) & (flowing_current(now.current_1)):
                append(cycle_count, cycle_step, "[P1,P2]", "")
            else:
                raise DataValidationError(date, pumping_station.name)
        # Transition:  [P1][], [P2][], [P1+P2][]
        elif not flowing_current(now.current_tot):
            # Transition:  [P1][], [P2][], [P1+P2][]
            if flowing_outflow(now.outflow_level):
                if not flowing_current(previous_l3.current_tot):
                    append_error("Pump is emitting water but operation stopped long time ago")
                    continue
                # Transition:  [P1][]
                if (flowing_current(previous_l3.current_1)) & (not flowing_current(previous_l3.current_2)):
                    append(cycle_count, cycle_step, "[P1]", "[P1][]")
                # Transition:  [P2][]
                elif (flowing_current(previous_l3.current_2)) & (not flowing_current(previous_l3.current_1)):
                    append(cycle_count, cycle_step, "[P2]", "[P2][]")
                # Transition:  [P1+P2][]
                elif (flowing_current(previous_l3.current_1)) & (flowing_current(previous_l3.current_2)):
                    append(cycle_count, cycle_step, "[P1,P2]", "[P1,P2][]")
                else:
                    raise DataValidationError(date, pumping_station.name)
            else:
                if flowing_outflow(previous.outflow_level):
                    cycle_count += 1
                    append(np.nan, np.nan, None, None)
                else:
                    if not next.water_level > now.water_level:
                        append_error("Water level should be rising")
                        continue
                    append(np.nan, np.nan, None, None)
        else:
            raise DataValidationError(date, pumping_station.name)
    log.update("Finished validation")

    # Set results
    df['cycle_nr'] = cycle_nrs
    df['cycle_step'] = cycle_steps
    df['cycle_state'] = cycle_states
    df['cycle_state_transition'] = cycle_transitions
    df['error'] = errors

    return df
