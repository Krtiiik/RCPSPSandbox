import itertools
import math
from collections import defaultdict
from queue import Queue
from typing import Callable, Iterable, Literal

from intervaltree import IntervalTree
import networkx as nx
import numpy as np
import tabulate
from docplex.cp.solution import CpoIntervalVarSolution

from instances.algorithms import build_instance_graph
from instances.problem_instance import Resource, ProblemInstance, Job, ResourceConsumption
from solver.solution import Solution
from utils import print_error, interval_overlap_function

T_MetricResult = float
T_Evaluation = dict[Resource, T_MetricResult]

T_Period = list[tuple[CpoIntervalVarSolution, Job]]


class MetricEvaluation:
    name: str
    evaluation: T_Evaluation

    def __init__(self, name: str, evaluation: T_Evaluation):
        self.name = name
        self.evaluation = evaluation


def evaluate_solution(solution: Solution,
                      evaluation_metric: Callable[[Solution, ProblemInstance, Resource], T_MetricResult],
                      instance: ProblemInstance = None,
                      evaluation_name: str = None,
                      ) -> MetricEvaluation:
    if instance is None:
        instance = solution.instance

    resource_metrics = T_Evaluation()
    for resource in instance.resources:
        resource_metrics[resource] = evaluation_metric(solution, instance, resource)
    return MetricEvaluation(evaluation_name, resource_metrics)


def print_evaluation(instance: ProblemInstance, evaluations: Iterable[MetricEvaluation]):
    evaluation_data = []
    resources_sorted_by_key = sorted(instance.resources, key=lambda r: r.key)
    for resource in resources_sorted_by_key:
        resource_evaluations = [resource.key] + [me.evaluation[resource] for me in evaluations]
        resource_evaluations = tuple(resource_evaluations)
        evaluation_data.append(resource_evaluations)

    print("Evaluations")
    print("-----------")
    print(tabulate.tabulate(evaluation_data,
                            headers=["Resource"] + [me.name for me in evaluations]))


# ~~~~~~~ Metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def machine_workload(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    mw = sum(job.duration
             for job in __jobs_consuming_resource(instance, resource))
    return T_MetricResult(mw)


def machine_utilization_rate(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    mur = (machine_workload(solution, instance, resource) / __machine_worktime(solution, instance, resource))
    return T_MetricResult(mur)


def average_uninterrupted_active_duration(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    periods = __compute_active_periods(solution, instance, resource)

    auad = machine_workload(solution, instance, resource) / len(periods)
    return T_MetricResult(auad)


def machine_resource_workload(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    mrw = sum(job.duration * consumption
              for job, consumption in __jobs_consuming_resource(instance, resource, yield_consumption=True))
    return T_MetricResult(mrw)


def machine_resource_utilization_rate(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    mrur = (machine_resource_workload(solution, instance, resource)
            / (resource.capacity * __machine_worktime(solution, instance, resource)))
    return T_MetricResult(mrur)


def average_uninterrupted_active_consumption(solution: Solution, instance: ProblemInstance, resource: Resource,
                                             average_over: Literal["consumption", "consumption ratio", "averaged consumption"]) -> T_MetricResult:
    # TODO resource capacity does not account for variable capacities

    def period_consumption(period: T_Period) -> int: return sum(job.resource_consumption.consumption_by_resource[resource] for interval, job in period)
    def period_length(period: T_Period) -> int: return period[-1][0].end - period[0][0].start

    periods = __compute_active_periods(solution, instance, resource)

    auad = 0
    if average_over == "consumption":
        auad = machine_resource_workload(solution, instance, resource) / len(periods)
    elif average_over == "consumption ratio":
        auad = avg(period_consumption(period) / (resource.capacity * period_length(period))
                   for period in periods)
    elif average_over == "averaged consumption":
        auad = avg(period_consumption(period) / resource.capacity
                   for period in periods)
    else:
        print_error(f"Unrecognized average argument: {average_over}")

    return T_MetricResult(auad)


def cumulative_delay(solution: Solution, instance: ProblemInstance, resource: Resource,
                     earliest_completion_times: dict[Job, int]) -> T_MetricResult:
    value = 0
    for job, consumption in __jobs_consuming_resource(instance, resource, yield_consumption=True):
        delay = earliest_completion_times[job]
        value += job.duration * consumption * delay

    return T_MetricResult(value)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def capacity_surplus(solution: Solution, instance: ProblemInstance
                     ) -> dict[Resource, list[tuple[int, int, int]]]:
    surpluses = dict()
    for resource in instance.resources:
        capacity_f = compute_resource_availability(resource, instance.horizon)
        consumption_f = __compute_resource_consumption(instance, solution, resource)
        consumption_f = [(s, e, -c) for s, e, c in consumption_f]
        surplus_f = interval_overlap_function(capacity_f + consumption_f)
        surpluses[resource] = surplus_f
    return surpluses


def capacity_transfer_matrix(solution: Solution, instance: ProblemInstance) -> np.array:
    # capacity_surpluses =
    pass


def time_relaxed_suffixes(instance: ProblemInstance, solution: Solution,
                          granularity: int = 1,
                          ):
    start_times = {job.id_job: solution.job_interval_solutions[job.id_job].start for job in instance.jobs}
    completion_times = {job.id_job: solution.job_interval_solutions[job.id_job].end for job in instance.jobs}
    durations = {job.id_job: job.duration for job in instance.jobs}
    predecessors = {job.id_job: [] for job in instance.jobs}
    for precedence in instance.precedences:
        predecessors[precedence.id_parent].append(precedence.id_child)

    t_to_job_to_start = {t: dict() for t in range(0, instance.horizon, granularity)}
    t_job_first = {t: dict() for t in range(0, instance.horizon, granularity)}

    graph = build_instance_graph(instance)
    jobs_topological = list(nx.topological_sort(graph))
    for t in range(0, instance.horizon, granularity):
        for job_id in jobs_topological:
            if start_times[job_id] <= t:
                t_to_job_to_start[t][job_id] = start_times[job_id]
                t_job_first[t][job_id] = True
            else:
                earliest_bound = max((t_to_job_to_start[t][predecessor] + durations[predecessor] for predecessor, _ in graph.in_edges(job_id)), default=0)
                t_to_job_to_start[t][job_id] = earliest_bound
                t_job_first[t][job_id] = all(start_times[predecessor] <= t for predecessor, _ in graph.in_edges(job_id))

    return t_to_job_to_start, t_job_first


def relaxed_intervals(instance: ProblemInstance, solution: Solution,
                      granularity: int = 1,
                      component: int = None,
                      ):
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
                t_consumptions[t].append((start, end, consumptions[job_id]))
                included.add(job_id)

    return t_consumptions


def relaxed_interval_consumptions(instance: ProblemInstance, solution: Solution,
                                  granularity: int = 1,
                                  component: int = None,
                                  return_intervals: bool = False,
                                  ):
    interval_consumptions: dict[int, list[tuple[int, int, ResourceConsumption]]] = relaxed_intervals(instance, solution, granularity, component)
    interval_consumptions_by_resource: dict[Resource, list[tuple[int, int, int]]] = defaultdict(list)

    for start, end, consumptions in itertools.chain(*interval_consumptions.values()):
        for resource in consumptions.consumption_by_resource:
            if consumptions.consumption_by_resource[resource] > 0:
                interval_consumptions_by_resource[resource].append((start, end, consumptions.consumption_by_resource[resource]))

    result = {r: interval_overlap_function(interval_consumptions_by_resource[r], first_x=0, last_x=instance.horizon) for r in interval_consumptions_by_resource}
    return (result, interval_consumptions_by_resource) if return_intervals else result


def left_closure(id_job: int, instance: ProblemInstance, solution: Solution) -> Iterable[int]:
    start_times = {j.id_job: solution.job_interval_solutions[j.id_job].start for j in instance.jobs}
    completion_times = {j.id_job: solution.job_interval_solutions[j.id_job].end for j in instance.jobs}
    durations = {j.id_job: j.duration for j in instance.jobs}
    graph = build_instance_graph(instance)
    resource_shift_starts = {r: set(ss) for r, ss in compute_resource_shift_starts(instance).items()}
    resource_shift_ends = {r: sorted(se) for r, se in compute_resource_shift_ends(instance).items()}
    resources_interval_trees = {r: IntervalTree.from_tuples((start_times[j.id_job], completion_times[j.id_job], j.id_job)
                                                            for j, c in __jobs_consuming_resource(instance, r))
                                for r in instance.resources}

    completion_time_jobs = defaultdict(set)  # Mapping from time t to jobs with completion time t
    job_consumption = defaultdict(set)  # Mapping from job to resources the job consumes
    for job in instance.jobs:
        completion_time_jobs[completion_times[job.id_job]].add(job.id_job)
        for r, c in job.resource_consumption.consumption_by_resource.items():
            if c > 0:
                job_consumption[job].add(r)

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


def compute_resource_shift_starts(instance: ProblemInstance) -> dict[Resource, Iterable[int]]:
    shift_starts = defaultdict(list)
    for resource in instance.resources:
        resource_availability = compute_resource_availability(resource, instance.horizon)
        last_c = 0
        for s, e, c in resource_availability:
            if c > 0 and last_c == 0:
                shift_starts[resource].append(s)
            last_c = c

    return shift_starts


def compute_resource_shift_ends(instance: ProblemInstance) -> dict[Resource, list[int]]:
    shift_ends = defaultdict(list)
    for resource in instance.resources:
        resource_availability = compute_resource_availability(resource, instance.horizon)
        last_c = 0
        last_e = 0
        for s, e, c in resource_availability:
            if c == 0 and last_c > 0:
                shift_ends[resource].append(last_e)
            last_c = c
            last_e = e

    return shift_ends


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_resource_availability(resource: Resource, horizon: int) -> list[tuple[int, int, int]]:
    """
    Builds a step function representing the availability of a resource over time.

    Args:
        resource (Resource): The resource to build the availability function for.
        horizon (int): The total number of hours in the planning horizon.

    Returns:
        CpoStepFunction: A step function representing the availability of the resource.
    """
    days_count = math.ceil(horizon / 24)
    intervals = [(i_day * 24 + start, i_day * 24 + end, capacity)
                 for i_day in range(days_count)
                 for start, end, capacity in resource.availability.periodical_intervals]
    return interval_overlap_function(intervals + resource.availability.exception_intervals,
                                     first_x=0, last_x=days_count * 24)


def __compute_resource_consumption(instance, solution, resource):
    consumptions = []
    for job, consumption in __jobs_consuming_resource(instance, resource, yield_consumption=True):
        int_solution = solution.job_interval_solutions[job.id_job]
        consumptions.append((int_solution.start, int_solution.end, consumption))
    consumption_f = interval_overlap_function(consumptions, first_x=0, last_x=math.ceil(instance.horizon / 24))
    return consumption_f


def __machine_worktime(solution: Solution, instance: ProblemInstance, resource: Resource) -> int:
    def completion_time(j): return solution.job_interval_solutions[j.id_job].end
    def start_time(j): return solution.job_interval_solutions[j.id_job].start
    machine_worktime = max(completion_time(job) for job in instance.jobs) - min(start_time(job) for job in instance.jobs)
    return machine_worktime


def __compute_active_periods(solution: Solution, instance: ProblemInstance, resource: Resource) -> list[T_Period]:
    resource_intervals = sorted(((solution.job_interval_solutions[job.id_job], job)
                                 for job in instance.jobs
                                 if job.resource_consumption.consumption_by_resource[resource] > 0),
                                key=lambda i: (i[0].start, i[0].end))

    periods: list[T_Period] = []
    current_period = [resource_intervals[0]]
    for interval, job in resource_intervals[1:]:
        if interval.start > current_period[-1][0].end:
            periods.append(current_period)
            current_period = [(interval, job)]
        else:
            current_period.append((interval, job))
    periods.append(current_period)

    return periods


def __jobs_consuming_resource(instance: ProblemInstance, resource: Resource, yield_consumption: bool = False) -> Iterable[Job]:
    for job in instance.jobs:
        consumption = job.resource_consumption.consumption_by_resource[resource]
        if consumption > 0:
            yield job, consumption if yield_consumption else job


def avg(it: Iterable):
    sm = 0
    count = 0
    for item in it:
        sm += item
        count += 1

    return sm / count
