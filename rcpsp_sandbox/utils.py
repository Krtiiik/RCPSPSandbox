import itertools
import sys
from collections import defaultdict
from typing import Iterable, Collection, TypeVar, Any, Sequence


T = TypeVar('T')
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


def intervals_overlap(i1, i2):
    return i1[0] < i2[1] and i2[0] < i1[1]


def flatten(iterables: Iterable[Iterable[T]]) -> Iterable[T]:
    return itertools.chain.from_iterable(iterables)


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


def try_open_read(filename: str,
                  filework,
                  *args,
                  **kwargs) -> Any:
    try:
        with open(filename, "r") as file:
            return filework(file, *args, **kwargs)
    except FileNotFoundError:
        print_error(f"File not found: {filename}")
        raise
    except IOError as error:
        print_error(error)
        raise


def list_of(items: Iterable[T]) -> list[T]:
    return items if items is list else list(items)


def chunk(sequence: Sequence[T],
          chunk_size: int) -> Iterable[Iterable[T]]:
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i:(i + chunk_size)]


def str_or_default(x: Any):
    return str(x) if x is not None else ""


def group_evaluations_kpis_by_instance_type(evaluations_kpis,
                                            scale: bool = False
                                            ):
    from bottlenecks.evaluations import EvaluationKPIsLightweight
    from bottlenecks.evaluations import Evaluation

    def make_dfdict(): return defaultdict(list)
    alpha = 10

    grouped_evaluations_kpis = defaultdict(make_dfdict)
    for inst_name, inst_evaluations_kpis in evaluations_kpis.items():
        inst_group_name = inst_name[:10]
        for i in range(len(inst_evaluations_kpis)):
            if scale:
                improv_max = max(e_kpi.improvement for e_kpi in inst_evaluations_kpis[i])
                cost_max = max(e_kpi.cost for e_kpi in inst_evaluations_kpis[i])
                diff_max = max(e_kpi.schedule_difference for e_kpi in inst_evaluations_kpis[i])
                duration_max = max(e_kpi.evaluation.duration for e_kpi in inst_evaluations_kpis[i])
                improv_max = improv_max if improv_max != 0 else 1
                cost_max = cost_max if cost_max != 0 else 1
                diff_max = diff_max if diff_max != 0 else 1
                duration_max = duration_max if duration_max != 0 else 1
                # noinspection PyTypeChecker
                grouped_evaluations_kpis[inst_group_name][i] += [
                    EvaluationKPIsLightweight(Evaluation(e_kpis.evaluation.base_instance.name,
                                                         e_kpis.evaluation.base_solution.job_interval_solutions,
                                                         e_kpis.evaluation.modified_instance.name,
                                                         e_kpis.evaluation.solution.job_interval_solutions,
                                                         e_kpis.evaluation.by,
                                                         alpha * e_kpis.evaluation.duration / duration_max),
                                              alpha * e_kpis.cost / cost_max,
                                              alpha * e_kpis.improvement / improv_max,
                                              alpha * e_kpis.schedule_difference / diff_max)
                    for e_kpis in inst_evaluations_kpis[i]
                ]
            else:
                grouped_evaluations_kpis[inst_group_name][i] += inst_evaluations_kpis[i]

    return {group_name: [evals_kpis[key] for key in sorted(evals_kpis)]
            for group_name, evals_kpis in grouped_evaluations_kpis.items()}


def avg(it: Iterable):
    sm = 0
    count = 0
    for item in it:
        sm += item
        count += 1

    return sm / count
