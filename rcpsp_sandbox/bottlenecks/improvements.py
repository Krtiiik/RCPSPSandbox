import itertools
import time
from collections import defaultdict, namedtuple
from queue import Queue
from typing import Iterable

import networkx as nx
from intervaltree import IntervalTree

import solver.model_builder
from bottlenecks.evaluations import EvaluationAlgorithm, Evaluation, ProblemSetup
from bottlenecks.utils import jobs_consuming_resource, compute_resource_shift_starts, \
    compute_resource_shift_ends, compute_capacity_migrations
from instances.algorithms import build_instance_graph
from instances.problem_instance import ProblemInstance, ResourceConsumption, Resource
from solver.solution import Solution
from solver.solver import Solver
from utils import interval_overlap_function


TimeVariableConstraintRelaxingAlgorithmSettings = namedtuple("TimeVariableConstraintRelaxingAlgorithmSettings",
                                                             ("max_iterations", "relax_granularity", "max_improvement_intervals"))


class TimeVariableConstraintRelaxingAlgorithm(EvaluationAlgorithm):
    def __init__(self):
        self._solver = Solver()

    def evaluate(self, problem: ProblemSetup, settings: TimeVariableConstraintRelaxingAlgorithmSettings
                 ) -> Evaluation:
        time_start = time.perf_counter()

        instance, target_job_id = problem
        model = self.__build_model(instance)
        solution = self._solver.solve(instance, model)

        instance_ = instance.copy()
        solution_ = solution
        for i_iter in range(settings.max_iterations):
            intervals_to_relax = self.__find_intervals_to_relax(settings, instance_, solution_, target_job_id)
            migrations, missing_capacity = compute_capacity_migrations(instance, solution, intervals_to_relax)
            instance_ = self.__modify_capacity(instance_)

            model =

        duration = time.perf_counter() - time_start
        return Evaluation(instance, instance_, solution_, duration)

    def __build_model(self, instance: ProblemInstance):
        return solver.model_builder.build_model(instance) \
               .with_precedences().with_resource_constraints() \
               .optimize_model()
    def __find_intervals_to_relax(self,
                                  settings: TimeVariableConstraintRelaxingAlgorithmSettings,
                                  instance: ProblemInstance, solution: Solution,
                                  target_job_id: int,
                                  ) -> dict[str, list[tuple[int, int, int]]]:
        _, improvement_intervals = relaxed_interval_consumptions(instance, solution, granularity=settings.relax_granularity, component=target_job_id)
        ints = [(resource_key, start, end, consumption, improvement)
                for resource_key, improvements in improvement_intervals.items()
                for start, end, consumption, improvement in improvements]
        ints.sort(key=lambda i: i[4], reverse=True)
        best_intervals = ints[:settings.max_improvement_intervals]

        best_intervals_by_resource = defaultdict(list)
        for resource_key, start, end, consumption, improvement in best_intervals:
            best_intervals_by_resource[resource_key].append((start, end, consumption))
        return best_intervals_by_resource

    def __modify_availability(self,
                            instance: ProblemInstance,
                            changes: dict[str, list[tuple[int, int, int]]],
                            migrations: dict[str, list[tuple[str, int, int, int]]],
                            ):
        if migrations:
            for r_from, migs in migrations.items():
                if r_from not in changes:
                    changes[r_from] = []
                for r_to, s, e, c in migs:
                    if r_to not in changes:
                        changes[r_to] = []
                    changes[r_from].append((s, e, -c))
                    changes[r_to].append((s, e, c))
        return modify_instance(inst).change_resource_availability(changes).generate_modified_instance()


def time_relaxed_suffixes(instance: ProblemInstance, solution: Solution,
                          granularity: int = 1,
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
        granularity (int, optional): The time interval granularity. Defaults to 1.

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

    t_job_start = {t: dict() for t in range(0, instance.horizon, granularity)}
    t_job_first = {t: dict() for t in range(0, instance.horizon, granularity)}

    graph = build_instance_graph(instance)
    jobs_topological = list(nx.topological_sort(graph))
    for t in range(0, instance.horizon, granularity):
        for job_id in jobs_topological:
            if start_times[job_id] <= t:
                t_job_start[t][job_id] = start_times[job_id]
                t_job_first[t][job_id] = True
            else:
                earliest_bound = max((t_job_start[t][predecessor] + durations[predecessor] for predecessor, _ in graph.in_edges(job_id)), default=0)
                t_job_start[t][job_id] = earliest_bound
                t_job_first[t][job_id] = all(start_times[predecessor] <= t for predecessor, _ in graph.in_edges(job_id))

    return t_job_start, t_job_first


def time_relaxed_suffix_consumptions(instance: ProblemInstance, solution: Solution,
                                     granularity: int = 1,
                                     component: int = None,
                                     ):
    """
    Computes consumptions of relaxed job intervals under the assumption that the job can start at any time within
    a time interval - ignoring capacity constraints.

    Args:
        instance (ProblemInstance): The problem instance.
        solution (Solution): The solution.
        granularity (int, optional): The granularity of the intervals. Defaults to 1.
        component (int, optional): The component to consider. Defaults to None.

    Returns:
        dict: A dictionary containing the relaxed intervals. The keys of the dictionary are job IDs, and the values are
        lists of tuples representing the relaxed intervals for each job. Each tuple contains the start time and end time
        of a relaxed interval.
    """

    t_job_start, t_job_first = time_relaxed_suffixes(instance, solution, granularity)
    durations = {j.id_job: j.duration for j in instance.jobs}
    consumptions = {j.id_job: j.resource_consumption for j in instance.jobs}
    start_times = {j.id_job: solution.job_interval_solutions[j.id_job].start for j in instance.jobs}
    component_closure = (left_closure(component, instance, solution) if component else set(j.id_job for j in instance.jobs))

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


def relaxed_interval_consumptions(instance: ProblemInstance, solution: Solution,
                                  granularity: int = 1,
                                  component: int = None,
                                  ):
    """
    Calculate consumptions of jobs under a relaxed assumption that the job can start at any time within a time interval.

    Args:
        instance (ProblemInstance): The problem instance.
        solution (Solution): The solution.
        granularity (int, optional): The granularity of the intervals. Defaults to 1.
        component (int, optional): The component to consider. Defaults to None.

    Returns:
        resource_consumptions, relaxed_intervals: A tuple containing two dictionaries:
        - A dictionary containing the resource consumptions. The keys of the dictionary are resources, and the values
        are consumption step functions.
        - A dictionary containing the relaxed intervals. The keys of the dictionary are resources, and the values are
        lists of tuples representing the relaxed intervals for each resource. Each tuple contains the start time, end time,
        the resource consumption, and the start time improvement of the interval under this relaxation.
    """

    interval_consumptions: dict[int, list[tuple[int, int, ResourceConsumption, int]]] = time_relaxed_suffix_consumptions(instance, solution, granularity, component)
    interval_consumptions_by_resource: dict[Resource, list[tuple[int, int, int]]] = defaultdict(list)
    improvements_by_resource: dict[Resource, list[int]] = defaultdict(list)

    for start, end, consumptions, improvement in itertools.chain(*interval_consumptions.values()):
        for resource in consumptions.consumption_by_resource:
            if consumptions.consumption_by_resource[resource] > 0:
                interval_consumptions_by_resource[resource].append((start, end, consumptions.consumption_by_resource[resource]))
                improvements_by_resource[resource].append(improvement)

    result = {r: interval_overlap_function(interval_consumptions_by_resource[r], first_x=0, last_x=instance.horizon) for r in interval_consumptions_by_resource}
    return result, {r: [(c[0], c[1], c[2], impr) for c, impr in zip(cons, improvements_by_resource[r])] for r, cons in interval_consumptions_by_resource.items()}


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
