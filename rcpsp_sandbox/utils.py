import sys
from collections import defaultdict
from typing import Iterable


def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def interval_step_function(intervals: Iterable[tuple[int, int, int]],
                           base_value: int = 0,
                           first_x: int = None,
                           ) -> list[tuple[int, int]]:
    value_diffs = defaultdict(int)
    for start, end, value in intervals:
        value_diffs[start] += value
        value_diffs[end] -= value
    xs = sorted(value_diffs)

    current = base_value
    steps: list[tuple[int, int]] = [(first_x, current)]
    for x in xs:
        current += value_diffs[x]
        steps.append((x, current))

    return steps


def interval_overlap_function(intervals: Iterable[tuple[int, int, int]],
                              base_value: int = 0,
                              first_x: int = None,
                              last_x: int = None,
                              ) -> list[tuple[int, int, int]]:
    steps = interval_step_function(intervals, base_value, first_x)

    intervals: list[tuple[int, int, int]] = []
    for (x, v), (x1, v1) in zip(steps, steps[1:]):
        intervals.append((x, x1, v))

    intervals.append((steps[-1][0], last_x, steps[-1][1]))

    return intervals
