import itertools
import sys
from collections import defaultdict
from typing import Iterable, Collection, TypeVar


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


def compute_component_jobs(problem_instance: "ProblemInstance") -> dict["Job", Collection["Job"]]:
    """
    Given a problem instance, returns a dictionary where each key is a root job of a component and the value is a
    collection of jobs that belong to that component.

    :param problem_instance: The problem instance to compute the component jobs for.
    :return: A dictionary where each key is a root job of a component and the value is a collection of jobs that belong
             to that component.
    """
    from instances.algorithms import traverse_instance_graph

    jobs_by_id = {j.id_job: j for j in problem_instance.jobs}
    jobs_components_grouped =\
        [[jobs_by_id[i[0]] for i in group]
         for _k, group in itertools.groupby(traverse_instance_graph(problem_instance, search="components topological generations",
                                                                    yield_state=True),
                                            key=lambda x: x[1])]  # we assume that the order in which jobs are returned is determined by the components, so we do not sort by component id
    component_jobs_by_root_job = index_groups(jobs_components_grouped,
                                              [jobs_by_id[c.id_root_job] for c in problem_instance.components])
    return component_jobs_by_root_job


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
    def __init__(self, color_count):
        self.range = color_count
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
