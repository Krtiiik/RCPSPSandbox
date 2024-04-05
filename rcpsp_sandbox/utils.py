import sys
from collections import defaultdict
from typing import Iterable, Collection, TypeVar, Any

T_StepFunction = list[tuple[int, int, int]]


def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def interval_step_function(intervals: Iterable[tuple[int, int, int]],
                           base_value: int = 0,
                           first_x: int = None,
                           merge_same: bool = True,
                           ) -> list[tuple[int, int]]:
    value_diffs = defaultdict(int)
    for start, end, value in intervals:
        value_diffs[start] += value
        value_diffs[end] -= value
    xs = sorted(value_diffs)

    current = base_value
    steps: list[tuple[int, int]] = [(first_x, current)]
    for x in xs:
        if merge_same and value_diffs[x] == 0:
            continue
        current += value_diffs[x]
        steps.append((x, current))

    return steps


def interval_overlap_function(intervals: Iterable[tuple[int, int, int]],
                              base_value: int = 0,
                              first_x: int = None,
                              last_x: int = None,
                              merge_same: bool = True,
                              ) -> list[tuple[int, int, int]]:
    steps = interval_step_function(intervals, base_value, first_x, merge_same=merge_same)

    intervals: list[tuple[int, int, int]] = []
    for (x, v), (x1, v1) in zip(steps, steps[1:]):
        intervals.append((x, x1, v))

    intervals.append((steps[-1][0], last_x, steps[-1][1]))

    return intervals


def flatten(iterables: Iterable[Iterable]):
    import functools
    import operator
    return functools.reduce(operator.iconcat, iterables, [])


COLORS = [
    '#a6cee3',
    '#1f78b4',
    '#b2df8a',
    '#33a02c',
    '#fb9a99',
    '#e31a1c',
    '#fdbf6f',
    '#ff7f00',
    '#cab2d6',
    '#6a3d9a',

    # '#8dd3c7',
    # '#ffffb3',
    # '#bebada',
    # '#fb8072',
    # '#80b1d3',
    # '#fdb462',
    # '#b3de69',
    # '#fccde5',
    # '#d9d9d9',
    # '#bc80bd',

    # 'xkcd:green',
    # 'xkcd:orange',
    # 'xkcd:red',
    # 'xkcd:violet',
    # 'xkcd:blue',
    # 'xkcd:yellow green',
    # # 'xkcd:yellow',
    # 'xkcd:yellow orange',
    # # 'xkcd:red orange',
    # 'xkcd:red violet',
    # 'xkcd:blue violet',
    # 'xkcd:blue green'
]


class ColorMap:
    def __init__(self):
        self.colors = COLORS

    def __getitem__(self, item):
        assert isinstance(item, int)
        return self.colors[item % len(self.colors)]


T = TypeVar('T')


def index_groups(groups: Iterable[Collection[T]], keys: Collection[T]) -> dict[T, Collection[T]]:
    """
    Indexes a collection of groups by a set of keys.
    :param groups: An iterable of collections to be indexed.
    :param keys: A collection of keys to index the groups by.
    :return: A dictionary where each key is a key from the input collection and each value is the first group that contains that key.
    :raises KeyError: If a key is not found in any of the groups.
    """
    index: dict[T, Collection[T]] = dict()
    for group in groups:
        for key in keys:
            if key in group:
                index[key] = group
                break
        else:
            raise KeyError("Group does not contain a key")
    return index


def modify_tuple(old_tuple: tuple, index: int, new_value: Any) -> tuple:
    return old_tuple[0:index] + (new_value,) + old_tuple[index + 1:]
