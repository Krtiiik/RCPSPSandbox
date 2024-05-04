import itertools
import sys
from collections import defaultdict
from typing import Iterable, Collection, TypeVar, Any, Sequence

import numpy as np

T = TypeVar('T')
T_StepFunction = list[tuple[int, int, int]]


def print_error(*args, **kwargs):
    """
    Prints the given arguments to the standard error stream.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        None
    """
    print(*args, file=sys.stderr, **kwargs)


def interval_step_function(intervals: Iterable[tuple[int, int, int]],
                           base_value: int = 0,
                           first_x: int = None,
                           merge_same: bool = True,
                           ) -> list[tuple[int, int]]:
    """
    Computes an interval step function based on the given intervals.

    Args:
        intervals: An iterable of tuples representing the intervals. Each tuple should contain three elements:
                   start (int): The start point of the interval.
                   end (int): The end point of the interval.
                   value (int): The value associated with the interval.
        base_value: The base value for the step function. Defaults to 0.
        first_x: The x-coordinate of the first step. Defaults to None.
        merge_same: A flag indicating whether to merge steps with the same value. Defaults to True.

    Returns:
        A list of tuples representing the steps of the step function. Each tuple contains two elements:
        x (int): The x-coordinate of the step.
        y (int): The y-coordinate of the step.
    """
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
    """
    Compute the interval overlap function based for the given intervals.

    Args:
        intervals (Iterable[tuple[int, int, int]]): The input intervals.
        base_value (int, optional): The base value for the intervals. Defaults to 0.
        first_x (int, optional): The starting x-value for the intervals. Defaults to None.
        last_x (int, optional): The ending x-value for the intervals. Defaults to None.
        merge_same (bool, optional): Flag indicating whether to merge intervals with the same value. Defaults to True.

    Returns:
        list[tuple[int, int, int]]: The overlapping intervals.
    """
    steps = interval_step_function(intervals, base_value, first_x, merge_same=merge_same)

    intervals: list[tuple[int, int, int]] = []
    for (x, v), (x1, v1) in zip(steps, steps[1:]):
        intervals.append((x, x1, v))

    intervals.append((steps[-1][0], last_x, steps[-1][1]))

    return intervals


def intervals_overlap(i1, i2):
    """
    Check if two intervals overlap.

    Args:
        i1 (tuple): The first interval, represented as a tuple (start, end).
        i2 (tuple): The second interval, represented as a tuple (start, end).

    Returns:
        bool: True if the intervals overlap, False otherwise.
    """
    return i1[0] < i2[1] and i2[0] < i1[1]


def flatten(iterables: Iterable[Iterable[T]]) -> Iterable[T]:
    """
    Flattens a nested iterable into a single iterable.

    Args:
        iterables (Iterable[Iterable[T]]): The nested iterable to be flattened.

    Returns:
        Iterable[T]: The flattened iterable.
    """
    return itertools.chain.from_iterable(iterables)


# Colors used for plotting
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
    """
    A class representing a color map.
    """

    def __init__(self):
        self.colors = COLORS

    def __getitem__(self, item):
        """
        Returns the color at the specified index.

        Args:
            item (int): The index of the color to retrieve.

        Returns:
            str: The color at the specified index.
        """
        assert isinstance(item, int)
        return self.colors[item % len(self.colors)]


def index_groups(groups: Iterable[Collection[T]], keys: Collection[T]) -> dict[T, Collection[T]]:
    """
    Indexes a collection of groups by a set of keys.
    Args:
        groups: An iterable of collections to be indexed.
        keys: A collection of keys to index the groups by.
    Returns:
        A dictionary where each key is a key from the input collection and each value is the first group that contains that key.
    Raises:
        KeyError: If a key is not found in any of the groups.
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
    """
    Modifies a tuple by replacing the value at the specified index with a new value.

    Args:
        old_tuple (tuple): The original tuple.
        index (int): The index of the value to be replaced.
        new_value (Any): The new value to be inserted at the specified index.

    Returns:
        tuple: The modified tuple with the new value inserted at the specified index.
    """
    return old_tuple[0:index] + (new_value,) + old_tuple[index + 1:]


def try_open_read(filename: str,
                  filework,
                  *args,
                  **kwargs) -> Any:
    """
    Opens the specified file in read mode and performs the given filework function on it.

    Args:
        filename (str): The name of the file to open.
        filework (function): The function to perform on the opened file.
        *args: Variable length argument list to be passed to the filework function.
        **kwargs: Arbitrary keyword arguments to be passed to the filework function.

    Returns:
        Any: The result of the filework function.

    Raises:
        FileNotFoundError: If the specified file is not found.
        IOError: If there is an error while reading the file.
    """
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
    """
    Convert an iterable to a list.

    Args:
        items (Iterable[T]): The iterable to be converted.

    Returns:
        list[T]: The converted list.

    """
    return items if items is list else list(items)


def chunk(sequence: Sequence[T],
          chunk_size: int) -> Iterable[Iterable[T]]:
    """
    Splits a sequence into smaller chunks of a specified size.

    Args:
        sequence: The sequence to be chunked.
        chunk_size: The size of each chunk.

    Yields:
        An iterable of iterables, where each inner iterable represents a chunk of the original sequence.
    """
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i:(i + chunk_size)]


def str_or_default(x: Any):
    """
    Converts the given input to a string representation if it is not None,
    otherwise returns an empty string.

    Args:
        x (Any): The input value to convert to a string.

    Returns:
        str: The string representation of the input value, or an empty string if the input is None.
    """
    return str(x) if x is not None else ""


def group_evaluations_kpis_by_instance_type(evaluations_kpis,
                                            scale: bool = False
                                            ):
    """
    Groups evaluation KPIs by instance type.

    Args:
        evaluations_kpis (dict): A dictionary containing evaluation KPIs for each instance.
        scale (bool, optional): Indicates whether to scale the evaluation KPIs. Defaults to False.

    Returns:
        dict: A dictionary containing grouped evaluation KPIs by instance type.
    """
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
    """
    Calculates the average of the elements in the given iterable.

    Args:
        it (Iterable): An iterable containing numeric values.

    Returns:
        float: The average of the elements in the iterable.

    Raises:
        ZeroDivisionError: If the iterable is empty.

    """
    sm = 0
    count = 0
    for item in it:
        sm += item
        count += 1

    return sm / count


def __is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points.

    This function takes an array of costs and finds the pareto-efficient points.
    Pareto efficiency is a concept in multi-objective optimization, where a solution is considered
    pareto-efficient if there is no other solution that improves one objective without worsening
    any other objective.

    The function iteratively compares each point in the costs array with the remaining points
    to determine if it is pareto-efficient. It removes dominated points and returns the indices
    of the pareto-efficient points.

    Args:
        costs: An (n_points, n_costs) array
        return_mask: True to return a mask

    Returns:
        An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.

    Reference: https://stackoverflow.com/a/40239615
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def pareto_front_kpis(evaluations_kpis, x, y):
    """
    Filters a list of evaluation KPIs based on Pareto dominance.

    Args:
        evaluations_kpis (list): A list of evaluation KPIs.
        x (str): The x-axis KPI to consider for Pareto dominance.
        y (str): The y-axis KPI to consider for Pareto dominance.

    Returns:
        list: A list of evaluation KPIs that are Pareto efficient based on the given x and y KPIs.
    """
    pareto_extractors = {
        "cost": (lambda ev: ev.cost),
        "improvement": (lambda ev: -ev.improvement),
        "schedule difference": (lambda ev: ev.schedule_difference),
        "duration": (lambda ev: ev.evaluation.duration)
    }

    x_extractor, y_extractor = pareto_extractors[x], pareto_extractors[y]

    def build_kpis_array(_alg_evaluations_kpis):
        return np.array([(x_extractor(_e_kpis), y_extractor(_e_kpis)) for _e_kpis in _alg_evaluations_kpis])

    filtered_evaluations_kpis = [_e_kpis for _e_kpis in evaluations_kpis
                                 if (_e_kpis.improvement > 0
                                     or (_e_kpis.cost == 0 and _e_kpis.schedule_difference == 0))]
    return [filtered_evaluations_kpis[i] for i in __is_pareto_efficient(build_kpis_array(filtered_evaluations_kpis), return_mask=False)]
