# Experiment instance building

import os
from collections import namedtuple

import instances.io as iio
from bottlenecks.utils import compute_longest_shift_overlap
from instances.problem_instance import ProblemInstance
from instances.problem_modifier import modify_instance

InstanceSetup = namedtuple("InstanceSetup", ("base_filename", "name", "gradual_level", "shifts", "due_dates",
                                             "tardiness_weights", "target_job", "scaledown_durations"))


MORNING = 1
AFTERNOON = 2
NIGHT = 4
SHIFT_INTERVALS = [
    [],                     # 0 =         |           |
    [( 6, 14)],             # 1 = Morning |           |
    [(14, 22)],             # 2 =         | Afternoon |
    [( 6, 22)],             # 3 = Morning | Afternoon |
    [( 0,  6), (22, 24)],   # 4 =         |           | Night
    [( 0, 14), (22, 24)],   # 5 = Morning |           | Night
    [( 0,  6), (14, 24)],   # 6 =         | Afternoon | Night
    [( 0, 24)],             # 7 = Morning | Afternoon | Night
]

experiment_instances: dict[str, InstanceSetup] = {
    # --- 30 jobs 4 resources MORNING | AFTERNOON shifts ---------------------------------------------------------------
    "instance01": InstanceSetup(
        base_filename="j3011_4.sm",
        name="instance01",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={23: 46, 28: 46, 29: 46, 30: 46, 32: 46},
        tardiness_weights={23: 1, 28: 1, 29: 3, 30: 1, 32: 1},
        target_job=29,
        scaledown_durations=False,
    ),
    "instance01_1": InstanceSetup(
        base_filename="j3011_2.sm",
        name="instance01_1",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={17: 46, 22: 46, 26: 46, 29: 46, 30: 46, 32: 46},
        tardiness_weights={17: 1, 22: 3, 26: 1, 29: 1, 30: 1, 32: 1},
        target_job=22,
        scaledown_durations=False,
    ),
    "instance01_2": InstanceSetup(
        base_filename="j3011_5.sm",
        name="instance01_2",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={8: 46, 25: 46, 26: 46, 28: 46, 29: 46, 30: 46, 32: 46},
        tardiness_weights={8: 1, 25: 1, 26: 3, 28: 1, 29: 1, 30: 1, 32: 1},
        target_job=26,
        scaledown_durations=False,
    ),
    "instance01_3": InstanceSetup(
        base_filename="j3011_6.sm",
        name="instance01_3",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={19: 46, 25: 46, 27: 46, 29: 46, 30: 46, 32: 46},
        tardiness_weights={19: 1, 25: 1, 27: 1, 29: 1, 30: 1, 32: 3},
        target_job=32,
        scaledown_durations=False,
    ),
    "instance01_4": InstanceSetup(
        base_filename="j3011_9.sm",
        name="instance01_4",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={18: 46, 28: 46, 29: 46, 30: 46, 32: 46},
        tardiness_weights={18: 1, 28: 3, 29: 1, 30: 1, 32: 1},
        target_job=28,
        scaledown_durations=False,
    ),
    # --- 30 jobs 2 resources ------------------------------------------------------------------------------------------
    "instance02": InstanceSetup(
        base_filename="j3010_2.sm",
        name="instance02",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON | NIGHT, "R2": MORNING | AFTERNOON},
        due_dates={26: 22, 27: 22, 29: 22, 30: 22, 32: 22},
        tardiness_weights={26: 1, 27: 1, 29: 1, 30: 1, 32: 1},
        target_job=26,
        scaledown_durations=False,
    ),
    "instance02_1": InstanceSetup(
        base_filename="j3010_4.sm",
        name="instance02_1",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON | NIGHT, "R2": MORNING | AFTERNOON},
        due_dates={23: 22, 28: 22, 29: 22, 30: 22, 32: 22},
        tardiness_weights={23: 1, 28: 1, 29: 1, 30: 1, 32: 1},
        target_job=26,
        scaledown_durations=False,
    ),
    "instance02_2": InstanceSetup(
        base_filename="j3010_5.sm",
        name="instance02_2",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON | NIGHT,"R2": MORNING | AFTERNOON},
        due_dates={19: 22, 22: 22, 23: 22, 26: 22, 27: 22, 29: 22, 30: 22, 32: 22},
        tardiness_weights={19: 1, 22: 1, 23: 1, 26: 1, 27: 1, 29: 1, 30: 1, 32: 1},
        target_job=26,
        scaledown_durations=False,
    ),
    "instance02_3": InstanceSetup(
        base_filename="j3010_7.sm",
        name="instance02_3",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON | NIGHT, "R2": MORNING | AFTERNOON},
        due_dates={17: 22, 25: 22, 28: 22, 29: 22, 30: 22, 32: 22},
        tardiness_weights={17: 1, 25: 1, 28: 1, 29: 1, 30: 1, 32: 1},
        target_job=26,
        scaledown_durations=False,
    ),
    "instance02_4": InstanceSetup(
        base_filename="j3010_8.sm",
        name="instance02_4",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON | NIGHT, "R2": MORNING | AFTERNOON},
        due_dates={24: 22, 27: 22, 28: 22, 29: 22, 30: 22, 32: 22},
        tardiness_weights={24: 1, 27: 1, 28: 1, 29: 1, 30: 1, 32: 1},
        target_job=26,
        scaledown_durations=False,
    ),
    # --- 60 jobs 1 resource Single tardy ------------------------------------------------------------------------------
    "instance03": InstanceSetup(
        base_filename="j6010_7.sm",
        name="instance03",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 94, 60: 94, 62: 94},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=False,
    ),
    "instance03_1": InstanceSetup(
        base_filename="j6010_8.sm",
        name="instance03_1",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 94, 60: 94, 62: 70},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=False,
    ),
    "instance03_2": InstanceSetup(
        base_filename="j6010_9.sm",
        name="instance03_2",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 94, 60: 94, 62: 94},
        tardiness_weights={59: 3, 60: 1, 62: 1},
        target_job=59,
        scaledown_durations=False,
    ),
    "instance03_3": InstanceSetup(
        base_filename="j6010_6.sm",
        name="instance03_3",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 94, 60: 94, 62: 46},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=False,
    ),
    "instance03_4": InstanceSetup(
        base_filename="j6010_2.sm",
        name="instance03_4",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 70, 60: 70, 62: 70},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=False,
    ),
    # --- 60 jobs 1 resource All tardy ---------------------------------------------------------------------------------
    "instance04": InstanceSetup(
        base_filename="j6010_7.sm",
        name="instance04",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 70, 60: 70, 62: 70},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=False,
    ),
    "instance04_1": InstanceSetup(
        base_filename="j6010_8.sm",
        name="instance04_1",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 46, 60: 46, 62: 46},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=False,
    ),
    "instance04_2": InstanceSetup(
        base_filename="j6010_9.sm",
        name="instance04_2",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 46, 60: 46, 62: 46},
        tardiness_weights={59: 3, 60: 1, 62: 1},
        target_job=59,
        scaledown_durations=False,
    ),
    "instance04_3": InstanceSetup(
        base_filename="j6010_6.sm",
        name="instance04_3",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 46, 60: 46, 62: 46},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=False,
    ),
    "instance04_4": InstanceSetup(
        base_filename="j6010_2.sm",
        name="instance04_4",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON},
        due_dates={59: 46, 60: 46, 62: 46},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=False,
    ),
    # --- 60 jobs 4 resources ------------------------------------------------------------------------------------------
    "instance05": InstanceSetup(
        base_filename="j6011_10.sm",
        name="instance05",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": AFTERNOON | NIGHT, "R4": MORNING | AFTERNOON | NIGHT},
        due_dates={59: 70, 60: 70, 62: 70},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=True,
    ),
    "instance05_1": InstanceSetup(
        base_filename="j6011_2.sm",
        name="instance05_1",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": AFTERNOON | NIGHT, "R4": MORNING | AFTERNOON | NIGHT},
        due_dates={59: 70, 60: 70, 62: 70},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=True,
    ),
    "instance05_2": InstanceSetup(
        base_filename="j6011_3.sm",
        name="instance05_2",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": AFTERNOON | NIGHT, "R4": MORNING | AFTERNOON | NIGHT},
        due_dates={59: 70, 60: 70, 62: 70},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=59,
        scaledown_durations=True,
    ),
    "instance05_3": InstanceSetup(
        base_filename="j6011_6.sm",
        name="instance05_3",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": AFTERNOON | NIGHT, "R4": MORNING | AFTERNOON | NIGHT},
        due_dates={59: 70, 60: 70, 62: 70},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=62,
        scaledown_durations=True,
    ),
    "instance05_4": InstanceSetup(
        base_filename="j6011_7.sm",
        name="instance05_4",
        gradual_level=1,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": AFTERNOON | NIGHT, "R4": MORNING | AFTERNOON | NIGHT},
        due_dates={59: 70, 60: 70, 62: 70},
        tardiness_weights={59: 1, 60: 1, 62: 3},
        target_job=59,
        scaledown_durations=True,
    ),
    # --- 60 jobs 4 resources ------------------------------------------------------------------------------------------
    "instance06": InstanceSetup(
        base_filename="j6013_6.sm",
        name="instance06",
        gradual_level=2,
        shifts={"R1": AFTERNOON, "R2": AFTERNOON, "R3": AFTERNOON, "R4": AFTERNOON},
        due_dates={32: 70, 54: 94, 57: 214, 58: 166, 59: 46, 60: 70, 62: 94},
        tardiness_weights={32: 1, 54: 1, 57: 1, 58: 1, 59: 1, 60: 3, 62: 1},
        target_job=60,
        scaledown_durations=True,
    ),
    "instance06_1": InstanceSetup(
        base_filename="j6013_2.sm",
        name="instance06_1",
        gradual_level=2,
        shifts={"R1": AFTERNOON, "R2": AFTERNOON, "R3": AFTERNOON, "R4": AFTERNOON},
        due_dates={6: 14, 50: 46, 51: 118, 53: 46, 54: 118, 58: 118, 59: 118, 60: 214, 62: 214},
        tardiness_weights={26: 1, 50: 1, 51: 3, 53: 1, 54: 1, 58: 1, 59: 1, 60: 1, 62: 1},
        target_job=51,
        scaledown_durations=True,
    ),
    "instance06_2": InstanceSetup(
        base_filename="j6013_3.sm",
        name="instance06_2",
        gradual_level=2,
        shifts={"R1": AFTERNOON, "R2": AFTERNOON, "R3": AFTERNOON, "R4": AFTERNOON},
        due_dates={52: 46, 54: 46, 55: 166, 56: 94, 57: 46, 59: 94, 60: 166, 62: 166},
        tardiness_weights={52: 1, 54: 1, 55: 1, 56: 1, 57: 1, 59: 3, 60: 1, 62: 1},
        target_job=59,
        scaledown_durations=True,
    ),
    "instance06_3": InstanceSetup(
        base_filename="j6013_5.sm",
        name="instance06_3",
        gradual_level=2,
        shifts={"R1": AFTERNOON, "R2": AFTERNOON, "R3": AFTERNOON, "R4": AFTERNOON},
        due_dates={38: 46, 39: 70, 56: 118, 57: 118, 58: 142, 59: 142, 60: 166, 62: 166},
        tardiness_weights={38: 1, 39: 1, 56: 1, 57: 1, 58: 1, 59: 1, 60: 3, 62: 1},
        target_job=60,
        scaledown_durations=True,
    ),
    "instance06_4": InstanceSetup(
        base_filename="j6013_10.sm",
        name="instance06_4",
        gradual_level=2,
        shifts={"R1": AFTERNOON, "R2": AFTERNOON, "R3": AFTERNOON, "R4": AFTERNOON},
        due_dates={26: 46, 50: 70, 53: 94, 54: 118, 58: 118, 59: 118, 60: 142, 62: 142},
        tardiness_weights={26: 1, 50: 1, 53: 1, 54: 1, 58: 3, 59: 1, 60: 1, 62: 1},
        target_job=58,
        scaledown_durations=True,
    ),
    # --- 120 jobs 4 resources -----------------------------------------------------------------------------------------
    "instance07": InstanceSetup(
        base_filename="j1201_1.sm",
        name="instance07",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={57: 46, 108: 46, 109: 46, 111: 70, 115: 94, 118: 118, 119: 142, 120: 142, 122: 166},
        tardiness_weights={57: 1, 108: 1, 109: 1, 111: 1, 115: 1, 118: 1, 119: 1, 120: 1, 122: 1},
        target_job=115,
        scaledown_durations=True,
    ),
    "instance07_1": InstanceSetup(
        base_filename="j1201_3.sm",
        name="instance07_1",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={47: 22, 114: 46, 116: 70, 117: 118, 118: 118, 119: 118, 120: 118, 122: 142},
        tardiness_weights={47: 1, 114: 1, 116: 1, 117: 1, 118: 1, 119: 1, 120: 1, 122: 1},
        target_job=117,
        scaledown_durations=True,
    ),
    "instance07_2": InstanceSetup(
        base_filename="j1201_6.sm",
        name="instance07_2",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={83: 46, 107: 46, 110: 46, 115: 70, 118: 94, 119: 94, 120: 70, 122: 94},
        tardiness_weights={83: 1, 107: 1, 110: 1, 115: 1, 118: 1, 119: 1, 120: 1, 122: 1},
        target_job=120,
        scaledown_durations=True,
    ),
    "instance07_3": InstanceSetup(
        base_filename="j1201_7.sm",
        name="instance07_3",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={73: 46, 82: 46, 112: 70, 113: 70, 115: 70, 118: 70, 119: 70, 120: 118, 122: 118},
        tardiness_weights={73: 1, 82: 1, 112: 1, 113: 1, 115: 1, 118: 1, 119: 1, 120: 1, 122: 1},
        target_job=122,
        scaledown_durations=True,
    ),
    "instance07_4": InstanceSetup(
        base_filename="j1201_10.sm",
        name="instance07_4",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON, "R3": MORNING | AFTERNOON, "R4": MORNING | AFTERNOON},
        due_dates={79: 46, 104: 46, 109: 46, 113: 46, 114: 94, 118: 94, 119: 118, 120: 118, 122: 118},
        tardiness_weights={79: 1, 104: 1, 109: 1, 113: 1, 114: 1, 118: 1, 119: 1, 120: 1, 122: 1},
        target_job=118,
        scaledown_durations=True,
    ),
    # ---- 120 jobs 2 resources ----------------------------------------------------------------------------------------
    "instance08": InstanceSetup(
        base_filename="j1205_1.sm",
        name="instance08",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON},
        due_dates={108: 22, 116: 70, 117: 70, 118: 70, 119: 46, 120: 118, 122: 94},
        tardiness_weights={108: 1, 116: 1, 117: 1, 118: 1, 119: 1, 120: 1, 122: 1},
        target_job=116,
        scaledown_durations=True,
    ),
    "instance08_1": InstanceSetup(
        base_filename="j1205_5.sm",
        name="instance08_1",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON},
        due_dates={101: 22, 112: 70, 113: 46, 116: 94, 117: 94, 119: 46, 120: 46, 122: 46},
        tardiness_weights={101: 1, 112: 1, 113: 1, 116: 1, 117: 1, 119: 1, 120: 1, 122: 1},
        target_job=112,
        scaledown_durations=True,
    ),
    "instance08_2": InstanceSetup(
        base_filename="j1205_6.sm",
        name="instance08_2",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON},
        due_dates={106: 22, 110: 70, 116: 46, 117: 70, 118: 134, 119: 70, 120: 46, 122: 70},
        tardiness_weights={106: 1, 110: 1, 116: 1, 117: 1, 118: 1, 119: 1, 120: 1, 122: 1},
        target_job=118,
        scaledown_durations=True,
    ),
    "instance08_3": InstanceSetup(
        base_filename="j1205_7.sm",
        name="instance08_3",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON},
        due_dates={101: 22, 110: 22, 114: 70, 115: 46, 116: 118, 117: 70, 119: 22, 120: 22, 122: 70},
        tardiness_weights={101: 1, 110: 1, 114: 1, 115: 1, 116: 1, 117: 1, 119: 1, 120: 1, 122: 1},
        target_job=117,
        scaledown_durations=True,
    ),
    "instance08_4": InstanceSetup(
        base_filename="j1205_9.sm",
        name="instance08_4",
        gradual_level=2,
        shifts={"R1": MORNING | AFTERNOON, "R2": MORNING | AFTERNOON},
        due_dates={57: 46, 90: 22, 110: 46, 116: 94, 118: 118, 119: 22, 120: 22, 122: 46},
        tardiness_weights={57: 1, 90: 1, 110: 1, 116: 1, 118: 1, 119: 1, 120: 1, 122: 1},
        target_job=116,
        scaledown_durations=True,
    ),
}


experiment_instances_info = {
    'instance01':   {"n": 32, },
    'instance01_1': {"n": 32, },
    'instance01_2': {"n": 32, },
    'instance01_3': {"n": 32, },
    'instance01_4': {"n": 32, },

    'instance02':   {"n": 32, },
    'instance02_1': {"n": 32, },
    'instance02_2': {"n": 32, },
    'instance02_3': {"n": 32, },
    'instance02_4': {"n": 32, },

    'instance03':   {"n": 62, },
    'instance03_1': {"n": 62, },
    'instance03_2': {"n": 62, },
    'instance03_3': {"n": 62, },
    'instance03_4': {"n": 62, },

    'instance04':   {"n": 62, },
    'instance04_1': {"n": 62, },
    'instance04_2': {"n": 62, },
    'instance04_3': {"n": 62, },
    'instance04_4': {"n": 62, },

    'instance05':   {"n": 62, },
    'instance05_1': {"n": 62, },
    'instance05_2': {"n": 62, },
    'instance05_3': {"n": 62, },
    'instance05_4': {"n": 62, },

    'instance06':   {"n": 62, },
    'instance06_1': {"n": 62, },
    'instance06_2': {"n": 62, },
    'instance06_3': {"n": 62, },
    'instance06_4': {"n": 62, },

    'instance07':   {"n": 122, },
    'instance07_1': {"n": 122, },
    'instance07_2': {"n": 122, },
    'instance07_3': {"n": 122, },
    'instance07_4': {"n": 122, },

    'instance08':   {"n": 122, },
    'instance08_1': {"n": 122, },
    'instance08_2': {"n": 122, },
    'instance08_3': {"n": 122, },
    'instance08_4': {"n": 122, },
}


def __build_shifts(shifts: dict[str, int]) -> dict[str, list[tuple[int, int]]]:
    return {r_key: SHIFT_INTERVALS[shift_flags] for r_key, shift_flags in shifts.items()}


def __parse_and_process(data_directory: str,
                        setup: InstanceSetup,
                        ) -> ProblemInstance:
    """
    Create a problem instance from a given setup. The base instance is parsed from a PSPLIB basefile and modified
    according to the setup.
    """

    shifts = __build_shifts(setup.shifts)

    # Parse
    instance = iio.parse_psplib(os.path.join(data_directory, setup.base_filename))

    # Get problem modifier
    instance_builder = modify_instance(instance)

    # Remove unused resources
    if len(shifts) < len(instance.resources):
        instance_builder.remove_resources(set(r.key for r in instance.resources) - set(shifts))

    # Component splitting, availabilities, due dates
    instance = (instance_builder
                .split_job_components(split="gradual", gradual_level=setup.gradual_level)
                .assign_resource_availabilities(availabilities=shifts)
                .assign_job_due_dates('uniform', interval=(0, 0))
                .assign_job_due_dates(due_dates=setup.due_dates, overwrite=True)
                .with_target_job(setup.target_job)
                .generate_modified_instance(setup.name)
                )

    # Scaling down long job durations
    if setup.scaledown_durations:
        longest_overlap = compute_longest_shift_overlap(instance)
        assert longest_overlap != 0, "There is no shift overlap"
        instance_builder.scaledown_job_durations(longest_overlap)

    # Tardiness weights
    for component in instance.components:
        component.weight = setup.tardiness_weights[component.id_root_job]

    return instance


def build_instance(instance_name: str,
                   base_instance_directory: str,
                   output_directory: str,
                   serialize: bool = True,
                   ) -> ProblemInstance:
    """
    Builds a problem instance based on the given parameters.

    Args:
        instance_name (str): The name of the experiment instance.
        base_instance_directory (str): The directory containing the base instance files.
        output_directory (str): The directory where the generated instance will be saved.
        serialize (bool, optional): Whether to serialize the instance as a JSON file. Defaults to True.

    Returns:
        ProblemInstance: The generated problem instance.

    Raises:
        ValueError: If the given instance name is not recognized.

    """
    if instance_name not in experiment_instances:
        raise ValueError(f'Unrecognized experiment instance "{instance_name}"')

    instance_setup = experiment_instances[instance_name]
    instance = __parse_and_process(base_instance_directory, instance_setup)

    if serialize:
        iio.serialize_json(instance, os.path.join(output_directory, instance_name+'.json'), is_extended=True)

    return instance
