import itertools
from functools import partial
from typing import Callable, Iterable, Literal

import tabulate
from docplex.cp.solution import CpoIntervalVarSolution

from bottlenecks.utils import jobs_consuming_resource
from instances.problem_instance import Resource, ProblemInstance, Job, compute_resource_availability
from solver.solution import Solution
from utils import print_error, intervals_overlap, modify_tuple

T_MetricResult = float
T_Evaluation = dict[str, T_MetricResult]

T_Period = list[tuple[CpoIntervalVarSolution, Job]]


class MetricEvaluation:
    name: str
    evaluation: T_Evaluation

    def __init__(self, name: str, evaluation: T_Evaluation):
        self.name = name
        self.evaluation = evaluation


def evaluate_solution(solution: Solution,
                      evaluation_metric: Callable[[Solution, ProblemInstance, Resource], T_MetricResult] | partial,
                      instance: ProblemInstance = None,
                      evaluation_name: str = None,
                      ) -> MetricEvaluation:
    if instance is None:
        instance = solution.instance

    resource_metrics = T_Evaluation()
    for resource in instance.resources:
        resource_metrics[resource.key] = evaluation_metric(solution, instance, resource)
    return MetricEvaluation(evaluation_name, resource_metrics)


def print_evaluation(instance: ProblemInstance, evaluations: Iterable[MetricEvaluation]):
    evaluation_data = []
    resources_sorted_by_key = sorted(instance.resources, key=lambda r: r.key)
    for resource in resources_sorted_by_key:
        resource_evaluations = [resource.key] + [me.evaluation[resource.key] for me in evaluations]
        resource_evaluations = tuple(resource_evaluations)
        evaluation_data.append(resource_evaluations)

    print("Evaluations")
    print("-----------")
    print(tabulate.tabulate(evaluation_data,
                            headers=["Resource"] + [me.name for me in evaluations]))


# ~~~~~~~ Metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def machine_workload(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    mw = sum(job.duration
             for job in jobs_consuming_resource(instance, resource))
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
              for job, consumption in jobs_consuming_resource(instance, resource, yield_consumption=True))
    return T_MetricResult(mrw)


def machine_resource_utilization_rate(solution: Solution, instance: ProblemInstance, resource: Resource,
                                      variable_capacity: bool = False) -> T_MetricResult:
    denominator = (__machine_total_capacity(solution, instance, resource) if variable_capacity
                   else (resource.capacity * __machine_worktime(solution, instance, resource)))
    mrur = (machine_resource_workload(solution, instance, resource)
            / denominator)
    return T_MetricResult(mrur)


def average_uninterrupted_active_consumption(solution: Solution, instance: ProblemInstance, resource: Resource,
                                             average_over: Literal["consumption", "consumption ratio", "averaged consumption"],
                                             variable_capacity: bool = False) -> T_MetricResult:
    def period_consumption(period: T_Period) -> int: return sum(job.resource_consumption.consumption_by_resource[resource] for interval, job in period)
    def period_length(period: T_Period) -> int: return period[-1][0].end - period[0][0].start
    def period_availability(availability_period): return sum(_c for _s, _e, _c in availability_period)

    periods = __compute_active_periods(solution, instance, resource)
    availability_periods = __compute_availability_periods(periods, instance, resource)

    auad = 0
    if average_over == "consumption":
        auad = machine_resource_workload(solution, instance, resource) / len(periods)
    elif average_over == "consumption ratio":
        auad = avg(period_consumption(period) / (period_availability(availability_period))
                   for period, availability_period in zip(periods, availability_periods))
    elif average_over == "averaged consumption":
        auad = avg(period_consumption(period) / resource.capacity
                   for period in periods)
    else:
        print_error(f"Unrecognized average argument: {average_over}")

    return T_MetricResult(auad)


def cumulative_delay(solution: Solution, instance: ProblemInstance, resource: Resource,
                     earliest_completion_times: dict[Job, int]) -> T_MetricResult:
    value = 0
    for job, consumption in jobs_consuming_resource(instance, resource, yield_consumption=True):
        delay = earliest_completion_times[job]
        value += job.duration * consumption * delay

    return T_MetricResult(value)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def __machine_worktime(solution: Solution, instance: ProblemInstance, resource: Resource) -> int:
    def completion_time(j): return solution.job_interval_solutions[j.id_job].end
    def start_time(j): return solution.job_interval_solutions[j.id_job].start
    machine_worktime = max(completion_time(job) for job in instance.jobs) - min(start_time(job) for job in instance.jobs)
    return machine_worktime


def __machine_total_capacity(solution: Solution, instance: ProblemInstance, resource: Resource) -> int:
    def completion_time(j): return solution.job_interval_solutions[j.id_job].end
    def start_time(j): return solution.job_interval_solutions[j.id_job].start
    start, end = min(start_time(job) for job in instance.jobs), max(completion_time(job) for job in instance.jobs)
    availability = compute_resource_availability(resource, instance, end)
    return sum((e - s) * c for s, e, c in availability)


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


def __compute_availability_periods(periods: list[T_Period], instance: ProblemInstance, resource: Resource) -> list[list[tuple[int, int, int]]]:
    def period_availability_overlap(_begin, _end):
        _overlapping_availability = [(_s, _e, _c) for _s, _e, _c in resource_availability  # Given that periods are consuming the resource,
                                     if intervals_overlap((_s, _e), (_begin, _end))]       # the overlapping availability is non-zero
        _overlapping_availability.sort()  # should already be, but to make sure...
        _overlapping_availability[0] = modify_tuple(_overlapping_availability[0], 0, _begin)
        _overlapping_availability[-1] = modify_tuple(_overlapping_availability[-1], 1, _end)
        return _overlapping_availability

    resource_availability = compute_resource_availability(resource, instance, instance.horizon)

    availability_periods = [period_availability_overlap(period[0][0].start, period[-1][0].end)
                            for period in periods]
    return availability_periods


def avg(it: Iterable):
    sm = 0
    count = 0
    for item in it:
        sm += item
        count += 1

    return sm / count
