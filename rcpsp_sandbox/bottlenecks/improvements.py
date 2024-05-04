import itertools
import math
from collections import defaultdict, namedtuple
from functools import partial
from queue import Queue
from typing import Iterable, Callable

import networkx as nx
import numpy as np
from intervaltree import IntervalTree

from bottlenecks.evaluations import EvaluationAlgorithm
from bottlenecks.metrics import average_uninterrupted_active_consumption, machine_resource_workload, \
    machine_resource_utilization_rate, evaluate_solution, T_MetricResult
from bottlenecks.utils import jobs_consuming_resource, compute_resource_shift_starts, \
    compute_resource_shift_ends, compute_resource_consumption, compute_capacity_surpluses, \
    group_consecutive_intervals
from instances.algorithms import build_instance_graph
from instances.problem_instance import ProblemInstance, compute_resource_periodical_availability, \
    CapacityMigration, CapacityChange, Resource
from instances.problem_modifier import modify_instance
from solver.model_builder import add_hot_start
from solver.solution import Solution
from utils import interval_overlap_function, intervals_overlap

ScheduleSuffixIntervalRelaxingAlgorithmSettings = namedtuple("ScheduleSuffixIntervalRelaxingAlgorithmSettings",
                                                             ("max_iterations", "relax_granularity", "max_improvement_intervals",
                                                              "interval_sort"))


class ScheduleSuffixIntervalRelaxingAlgorithm(EvaluationAlgorithm):
    """
    An algorithm for improving a solution, specifically the tardiness of a selected order.
    The algorithm works by relaxing solution suffixes, allowing jobs to start earlier than their original start times.
    In the relaxed solutions, improvement intervals are identified and the algorithm selects the best intervals to relax.

    The algorithm works as follows:
    1. For each iteration, the algorithm identifies the best intervals to relax.
    2. Capacity changes are applied based on the identified intervals.
    3. The modified instance is solved.
    4. Resource availability is reduced based on the new solution.
    """
    INTERVAL_SORTS = {
        "improvement": (lambda itv: -itv[3]),
        "time": (lambda itv: (-itv[0], -itv[1]))
    }

    def __init__(self):
        super().__init__()

    @property
    def settings_type(self) -> type:
        return ScheduleSuffixIntervalRelaxingAlgorithmSettings

    def _run(self,
             base_instance: ProblemInstance, base_solution: Solution,
             settings,
             ) -> tuple[ProblemInstance, Solution]:
        def modified_instance_name(): return (f'{modified_instance.name.split(EvaluationAlgorithm.ID_SEPARATOR)[0]}'
                                              f'{EvaluationAlgorithm.ID_SEPARATOR}{self.represent_short(settings)}'
                                              f'{EvaluationAlgorithm.ID_SUB_SEPARATOR}{i_iter}')
        if settings.interval_sort.lower() not in self.INTERVAL_SORTS:
            raise ValueError("Unrecognized interval sort")

        modified_instance = base_instance.copy()
        solution = base_solution
        reduce_capacity_changes(modified_instance, solution)  # Initial capacity reduction

        for i_iter in range(settings.max_iterations):  # Main loop
            intervals_to_relax = self.__find_intervals_to_relax(settings, modified_instance, solution)  # Find improvement intervals
            modified_instance = modify_instance_availability(modified_instance, {}, intervals_to_relax, modified_instance_name())  # Modify the resource availabilities

            model = self._build_standard_model(modified_instance)
            model = add_hot_start(model, solution)  # Warm-start the model
            solution = self._solver.solve(modified_instance, model)  # Solve the modified instance
            reduce_capacity_changes(modified_instance, solution)  # Reduce capacity changes based on the new solution, find migrations and additions

        return modified_instance, solution

    def __find_intervals_to_relax(self,
                                  settings: ScheduleSuffixIntervalRelaxingAlgorithmSettings,
                                  instance: ProblemInstance, solution: Solution,
                                  ) -> dict[str, list[CapacityChange]]:
        improvement_intervals = self.relaxed_interval_consumptions(instance, solution, settings, component=instance.target_job)

        # Choose the best intervals to relax
        improvement_intervals.sort(key=self.INTERVAL_SORTS[settings.interval_sort])
        best_intervals = improvement_intervals[:settings.max_improvement_intervals]

        # Construct the intervals to relax
        best_intervals_by_resource = defaultdict(list)
        for start, end, consumption, improvement in best_intervals:
            for resource in consumption.consumption_by_resource:
                if consumption[resource] > 0:
                    best_intervals_by_resource[resource.key].append(CapacityChange(start, end, consumption[resource]))

        return best_intervals_by_resource

    @staticmethod
    def time_relaxed_suffixes(instance: ProblemInstance, solution: Solution,
                              settings: ScheduleSuffixIntervalRelaxingAlgorithmSettings,
                              ):
        """
        Calculate the start times and first job indicators for each job at different time intervals.

        Given a solution to a given problem isntance, this function calculates the possible start times for each job
        at different time intervals under the assumption that the job can start at any time within the interval - ignoring
        capacity constraints.

        The first job indicator is a boolean value indicating whether the job is the first among its precedence predecessors
        to start in the given time interval under the assumption.

        Args:
            instance (ProblemInstance): The problem instance.
            solution (Solution): The solution containing job interval solutions.


        Returns:
            Tuple[Dict[int, Dict[int, int]], Dict[int, Dict[int, bool]]]: A tuple containing two dictionaries:
                - t_job_start: A dictionary mapping time intervals to job IDs and their corresponding start times.
                - t_job_first: A dictionary mapping time intervals to job IDs and their corresponding first job indicators.
        """

        start_times = {job.id_job: solution.job_interval_solutions[job.id_job].start for job in instance.jobs}
        durations = {job.id_job: job.duration for job in instance.jobs}
        predecessors = {job.id_job: [] for job in instance.jobs}
        for precedence in instance.precedences:
            predecessors[precedence.id_parent].append(precedence.id_child)

        t_job_start = {t: dict() for t in range(0, instance.horizon, settings.relax_granularity)}
        t_job_first = {t: dict() for t in range(0, instance.horizon, settings.relax_granularity)}

        graph = build_instance_graph(instance)
        jobs_topological = list(nx.topological_sort(graph))
        for t in range(0, instance.horizon, settings.relax_granularity):
            for job_id in jobs_topological:
                if start_times[job_id] <= t:
                    t_job_start[t][job_id] = start_times[job_id]
                    t_job_first[t][job_id] = True
                else:
                    earliest_bound = max((t_job_start[t][predecessor] + durations[predecessor] for predecessor, _ in graph.in_edges(job_id)), default=0)
                    t_job_start[t][job_id] = earliest_bound
                    t_job_first[t][job_id] = all(start_times[predecessor] <= t for predecessor, _ in graph.in_edges(job_id))

        return t_job_start, t_job_first

    def time_relaxed_suffix_consumptions(self,
                                         instance: ProblemInstance, solution: Solution,
                                         settings: ScheduleSuffixIntervalRelaxingAlgorithmSettings,
                                         component: int = None,
                                         ):
        """
        Computes consumptions of relaxed job intervals under the assumption that the job can start at any time within
        a time interval - ignoring capacity constraints.

        Args:
            instance (ProblemInstance): The problem instance.
            solution (Solution): The solution.

            component (int, optional): The component to consider. Defaults to None.

        Returns:
            dict: A dictionary containing the relaxed intervals. The keys of the dictionary are job IDs, and the values are
            lists of tuples representing the relaxed intervals for each job. Each tuple contains the start time and end time
            of a relaxed interval.
        """

        t_job_start, t_job_first = self.time_relaxed_suffixes(instance, solution, settings)
        durations = {j.id_job: j.duration for j in instance.jobs}
        consumptions = {j.id_job: j.resource_consumption for j in instance.jobs}
        start_times = {j.id_job: solution.job_interval_solutions[j.id_job].start for j in instance.jobs}
        component_closure = (
            left_closure(component, instance, solution) if component else set(j.id_job for j in instance.jobs))

        t_consumptions = defaultdict(list)
        included = set()
        for t, t1 in itertools.pairwise(sorted(t_job_start) + [instance.horizon + 1]):
            for job_id in t_job_start[t]:
                start = t_job_start[t][job_id]
                end = start + durations[job_id]
                if (t <= start < t1
                    and start < start_times[job_id]
                    and job_id not in included
                    and t_job_first[t][job_id]
                    and job_id in component_closure
                ):
                    t_consumptions[t].append((start, end, consumptions[job_id], start_times[job_id] - start))
                    included.add(job_id)

        return t_consumptions

    def relaxed_interval_consumptions(self,
                                      instance: ProblemInstance, solution: Solution,
                                      settings: ScheduleSuffixIntervalRelaxingAlgorithmSettings,
                                      component: int = None,
                                      ):
        """
        Calculate consumptions of jobs under a relaxed assumption that the job can start at any time within a time interval.

        Args:
            instance (ProblemInstance): The problem instance.
            solution (Solution): The solution.
            settings (ScheduleSuffixIntervalRelaxingAlgorithmSettings): The settings for the algorithm.
            component (int, optional): The component to consider. Defaults to None.

        Returns:
            resource_consumptions, relaxed_intervals: A tuple containing two dictionaries:
            - A dictionary containing the resource consumptions. The keys of the dictionary are resources, and the values
            are consumption step functions.
            - A dictionary containing the relaxed intervals. The keys of the dictionary are resources, and the values are
            lists of tuples representing the relaxed intervals for each resource. Each tuple contains the start time, end time,
            the resource consumption, and the start time improvement of the interval under this relaxation.
        """
        intervals_consumptions = self.time_relaxed_suffix_consumptions(instance, solution, settings, component)
        return list(itertools.chain.from_iterable(intervals_consumptions.values()))


def left_closure(id_job: int, instance: ProblemInstance, solution: Solution) -> Iterable[int]:
    """
    Computes the left closure of a job in the given instance and solution.

    Args:
        id_job (int): The ID of the job for which to compute the left closure.
        instance (ProblemInstance): The problem instance containing the jobs and resources.
        solution (Solution): The solution containing the job interval solutions.

    Returns:
        Iterable[int]: The set of job IDs in the left closure of the given job.
    """

    start_times = {j.id_job: solution.job_interval_solutions[j.id_job].start for j in instance.jobs}
    completion_times = {j.id_job: solution.job_interval_solutions[j.id_job].end for j in instance.jobs}
    durations = {j.id_job: j.duration for j in instance.jobs}
    graph = build_instance_graph(instance)
    resource_shift_starts = {r: set(ss) for r, ss in compute_resource_shift_starts(instance).items()}
    resource_shift_ends = {r: sorted(se) for r, se in compute_resource_shift_ends(instance).items()}
    resources_interval_trees = {r: IntervalTree.from_tuples((start_times[j.id_job], completion_times[j.id_job], j.id_job)
                                                            for j, c in jobs_consuming_resource(instance, r))
                                for r in instance.resources}

    completion_time_jobs = defaultdict(set)  # Mapping from time t to jobs with completion time t
    job_consumption = {}  # Mapping from job to resources the job consumes
    for job in instance.jobs:
        job_consumption[job.id_job] = set()
        completion_time_jobs[completion_times[job.id_job]].add(job.id_job)
        for r, c in job.resource_consumption.consumption_by_resource.items():
            if c > 0:
                job_consumption[job.id_job].add(r)

    queue = Queue()
    queue.put(id_job)

    closure = set()
    while not queue.empty():
        n = queue.get(block=False)
        start = start_times[n]

        if n in closure:
            continue

        # Process precedence predecessors
        for pred in graph.predecessors(n):
            if completion_times[pred] == start:
                queue.put(pred)

        if start in completion_time_jobs:
            # Process resource predecessors
            for pred in completion_time_jobs[start]:
                if job_consumption[n] & job_consumption[pred]:
                    queue.put(pred)

        # Process resource predecessors over resource pauses
        for r in job_consumption[n]:
            if start in resource_shift_starts[r]:
                prev_shift_end = max((s_end for s_end in resource_shift_ends[r] if s_end < start), default=0)
                low_bound = prev_shift_end - durations[n]
                for s, e, pred in resources_interval_trees[r].overlap(low_bound, prev_shift_end):
                    queue.put(pred)

        closure.add(n)

    return closure


def modify_instance_availability(instance: ProblemInstance,
                                 migrations: dict[str, list[CapacityMigration]],
                                 additions: dict[str, list[CapacityChange]],
                                 modified_instance_name: str,
                                 ):
    """
    Create a modified problem instance by modifying the resource availabilities based on the given migrations and additions.

    Args:
        instance (ProblemInstance): The original problem instance.
        migrations (dict[str, list[CapacityMigration]]): A dictionary containing migrations for each resource.
        additions (dict[str, list[CapacityChange]]): A dictionary containing additions for each resource.
        modified_instance_name (str): The name of the modified instance.

    Returns:
        ProblemInstance: The modified problem instance.
    """
    return modify_instance(instance).change_resource_availability(additions, migrations).generate_modified_instance(modified_instance_name)


def reduce_capacity_changes(instance: ProblemInstance, solution: Solution):
    """
    Reduce capacity changes in the given instance and solution.
    Find additions and migrations for each resource.

    Args:
        instance (ProblemInstance): The problem instance.
        solution (Solution): The solution.
    """

    horizon = instance.horizon

    # Compute required changes
    required_changes = dict()
    for resource in instance.resources:
        consumption = compute_resource_consumption(instance, solution, resource)
        periodical_availability = compute_resource_periodical_availability(resource, horizon)
        consumption_exceeding_capacity = interval_overlap_function(consumption + [(s, e, -c) for s, e, c in periodical_availability],
                                                                   first_x=0, last_x=horizon, merge_same=False)
        consumption_exceeding_capacity = [(s, e, c) for s, e, c in consumption_exceeding_capacity if c > 0]  # filter out consumption not reaching capacity
        required_changes[resource.key] = consumption_exceeding_capacity

    def min_surplus_in_range(_r_key, _b, _e):
        _overlapping_surpluses = (c for s, e, c in capacity_surpluses[_r_key] if intervals_overlap((_b, _e), (s, e)))
        return min(_overlapping_surpluses, default=0)

    def update_surplus(_r_key, _migrations):
        capacity_surpluses[_r_key] = interval_overlap_function(capacity_surpluses[_r_key]
                                                               + [(s, e, -c) for _r_from_key, s, e, c in _migrations],
                                                               first_x=0, last_x=horizon)

    def find_best_migrations(_r_to_key, _change_group):
        _to_find = sum(c for s, e, c in _change_group)
        _changes_to_find = [c for s, e, c in _change_group]
        _migrations = defaultdict(list)
        while _to_find > 0:
            _possible_migrations = dict()
            for _r_from_key in capacity_surpluses:
                if _r_from_key == _r_to_key:
                    continue
                _possible_migrations[_r_from_key] = [(chng[0], chng[1], min(_changes_to_find[i], min_surplus_in_range(_r_from_key, chng[0], chng[1])))
                                                     for i, chng in enumerate(_change_group)]

            # Find the resource to migrate from based on the most capacity that can be migrated
            _take_from = max(_possible_migrations.items(),
                             key=(lambda _r_key__migrations: sum(c for s, e, c in _r_key__migrations[1])),
                             default=(None, None))[0]

            _total_migration = 0 if _take_from is None else sum(c for s, e, c in _possible_migrations[_take_from])
            if _total_migration == 0:  # If no more migrations are possible...
                break

            _new_migrations = [CapacityMigration(_r_to_key, s, e, c) for s, e, c in _possible_migrations[_take_from] if c > 0]

            _migrations[_take_from] += _new_migrations
            update_surplus(_take_from, _new_migrations)
            _to_find -= _total_migration
            for _i, _mig in enumerate(_possible_migrations[_take_from]):
                _changes_to_find[_i] -= _mig[2]

        return _migrations

    # Compute migrations
    capacity_surpluses = compute_capacity_surpluses(solution, instance, ignore_changes=True)
    consecutive_required_changes = {r_key: group_consecutive_intervals(chngs) if chngs else [] for r_key, chngs in required_changes.items()}
    migrations = defaultdict(list)
    for r_key, change_groups in consecutive_required_changes.items():
        for change_group in change_groups:
            group_migrations = find_best_migrations(r_key, change_group)
            for r_from, migs in group_migrations.items():
                migrations[r_from] += migs

    # Compute additions
    migration_changes = defaultdict(list)
    for r_from, migs in migrations.items():
        for r_to, s, e, c in migs:
            migration_changes[r_from].append((s, e, -c))
            migration_changes[r_to].append((s, e, c))
    migration_changes = {r.key: sorted(migration_changes[r.key]) for r in instance.resources}

    additions = dict()
    for resource in instance.resources:
        consumption = compute_resource_consumption(instance, solution, resource)
        periodical_availability = compute_resource_periodical_availability(resource, horizon)
        adds = interval_overlap_function(consumption
                                         + [(s, e, -c) for s, e, c in periodical_availability]
                                         + [(s, e, -c) for s, e, c in migration_changes[resource.key]],
                                         first_x=0, last_x=horizon, merge_same=True)
        adds = [CapacityChange(s, e, c) for s, e, c in adds if c > 0]
        additions[resource.key] = adds

    for resource in instance.resources:
        resource.availability.additions = additions[resource.key]
        resource.availability.migrations = migrations[resource.key]


IdentificationIndicatorRelaxingAlgorithmSettings = namedtuple("IdentificationIndicatorRelaxingAlgorithmSettings",
                                                              ("metric", "granularity", "convolution_mask",
                                                               "max_iterations", "max_improvement_intervals", "capacity_addition"))


class IdentificationIndicatorRelaxingAlgorithm(EvaluationAlgorithm):
    """
    An algorithm for improving a solution by relaxing resource availabilities based on a given metric.
    The algorithm works by identifying bottleneck resource based on the metric and relaxing the resource.
    The relaxation is done by adding capacity to the resource in specific intervals identified by a convolution mask.

    The algorithm works as follows:
    1. For each iteration, the algorithm identifies the bottleneck resource based on the metric.
    2. The algorithm computes the period consumption of the resource and convolves it with a convolution mask.
    3. The algorithm identifies the best intervals to relax based on the convolution result.
    4. Capacity changes are applied based on the identified intervals.
    5. The modified instance is solved.
    6. Resource availability is reduced based on the new solution.
    """
    METRIC_MAPPING: dict[str, Callable[[Solution, ProblemInstance, Resource], T_MetricResult] | partial] = {
        'mrw': machine_resource_workload,
        'mrur': partial(machine_resource_utilization_rate, variable_capacity=True),
        'auau': partial(average_uninterrupted_active_consumption, average_over="consumption ratio"),
    }

    CONVOLUTION_MASKS: dict[str, list[int]] = {
        "pre1": [2, 4, 5, 2, 1],
        "around": [1, 3, 5, 3, 5, 3, 1],
        "post": [1, 1, 2, 3, 5, 4, 2],
    }

    @property
    def settings_type(self) -> type:
        return IdentificationIndicatorRelaxingAlgorithmSettings

    def _run(self,
             base_instance: ProblemInstance, base_solution: Solution,
             settings: IdentificationIndicatorRelaxingAlgorithmSettings,
             ) -> tuple[ProblemInstance, Solution]:
        def modified_instance_name(): return (f'{modified_instance.name.split(EvaluationAlgorithm.ID_SEPARATOR)[0]}'
                                              f'{EvaluationAlgorithm.ID_SEPARATOR}{self.represent_short(settings)}'
                                              f'{EvaluationAlgorithm.ID_SUB_SEPARATOR}{i_iter}')

        if settings.metric.lower() not in self.METRIC_MAPPING:
            raise ValueError("Unrecognized metric")
        if settings.convolution_mask.lower() not in self.CONVOLUTION_MASKS:
            raise ValueError("Unrecognized convolution mask")

        metric = self.METRIC_MAPPING[settings.metric.lower()]
        convolution_mask = self.CONVOLUTION_MASKS[settings.convolution_mask.lower()]

        modified_instance = base_instance.copy()
        solution = base_solution

        for i_iter in range(settings.max_iterations):
            metric_evaluation = evaluate_solution(solution, metric, modified_instance)  # Evaluate the solution with the identification indicator
            bottleneck_resource = argmax(metric_evaluation.evaluation)  # Identify the bottleneck resource

            period_consumption = self.__compute_period_consumption(solution, bottleneck_resource, settings.granularity)  # Compute the granular period consumption of the resource
            convolved_consumption = np.convolve(period_consumption, convolution_mask, mode='same')  # Convolve the granular consumption with the convolution mask
            priority_periods = np.argsort(convolved_consumption)[::-1]  # Sort the granular periods based on the convolution result

            relaxing_periods = priority_periods[:settings.max_improvement_intervals]  # Select the best periods to relax
            relaxing_intervals = [CapacityChange(period * settings.granularity, (period + 1) * settings.granularity, settings.capacity_addition)
                                  for period in relaxing_periods]
            modified_instance = modify_instance_availability(modified_instance, {}, {bottleneck_resource: relaxing_intervals}, modified_instance_name())

            model = self._build_standard_model(modified_instance)
            model = add_hot_start(model, solution)  # Warm-start the model
            solution = self._solver.solve(modified_instance, model)  # Solve the modified instance
            reduce_capacity_changes(modified_instance, solution)  # Reduce capacity changes based on the new solution, find migrations and additions

        return modified_instance, solution

    @staticmethod
    def __compute_period_consumption(solution: Solution, resource_key: str, granularity: int):
        """
        Compute the granular consumption of a resource.

        Args:
            solution (Solution): The solution object.
            resource_key (str): The key of the resource.
            granularity (int): The granular periods granularity.

        Returns:
            list: A list representing the granular consumption of the resource.
        """
        full_consumption = compute_resource_consumption(solution.instance, solution, solution.instance.resources_by_key[resource_key])

        period_consumption = [0] * math.ceil(solution.instance.horizon / granularity)
        for s, e, c in full_consumption:
            period_low = math.floor(s / granularity)
            period_high = math.ceil(e / granularity)
            if period_low + 1 == period_high:
                period_consumption[period_low] += (e-s) * c
            else:
                period_consumption[period_low] += (((period_low + 1) * granularity) - s) * c
                for period in range(period_low+1, period_high-1, 1):
                    period_consumption[period] += granularity * c

                period_consumption[period_high-1] += (e - ((period_high - 1) * granularity)) * c

        return period_consumption


def argmax(sequence):
    if isinstance(sequence, dict):
        return max(sequence, key=sequence.get)
    return max(enumerate(sequence), key=lambda ix: ix[1])[0]
