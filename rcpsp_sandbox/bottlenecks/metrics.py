import math
from typing import Callable, Iterable, Literal

import numpy as np
import tabulate
from docplex.cp.solution import CpoIntervalVarSolution

from instances.problem_instance import Resource, ProblemInstance, Job
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
        capacity_f = __compute_resource_availability(resource, instance.horizon)
        consumption_f = __compute_resource_consumption(instance, solution, resource)
        consumption_f = [(s, e, -c) for s, e, c in consumption_f]
        surplus_f = interval_overlap_function(capacity_f + consumption_f)
        surpluses[resource] = surplus_f
    return surpluses


def capacity_transfer_matrix(solution: Solution, instance: ProblemInstance) -> np.array:
    capacity_surpluses = 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def __compute_resource_availability(resource: Resource, horizon: int) -> list[tuple[int, int, int]]:
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
