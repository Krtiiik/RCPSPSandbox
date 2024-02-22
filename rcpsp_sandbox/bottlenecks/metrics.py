from typing import Callable

from instances.problem_instance import Resource, ProblemInstance
from solver.solution import Solution


T_MetricResult = float


def evaluate_solution(solution: Solution,
                      instance: ProblemInstance,
                      evaluation_metric: Callable[[Solution, ProblemInstance, Resource], T_MetricResult]
                      ) -> dict[Resource, T_MetricResult]:
    resource_metrics = dict()
    for resource in instance.resources:
        resource_metrics[resource] = evaluation_metric(solution, instance, resource)
    return resource_metrics


# ~~~~~~~ Metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def machine_workload(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    mw = sum(job.duration
             for job in instance.jobs
             if job.resource_consumption.consumption_by_resource[resource] > 0)
    return T_MetricResult(mw)


def machine_utilization_rate(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    def completion_time(j): return solution.job_interval_solutions[j].end

    mur = (machine_workload(solution, instance, resource) /
           (max(completion_time(job) for job in instance.jobs) - min(completion_time(job) - job.duration for job in instance.jobs))
           )
    return mur


def average_uninterrupted_active_duration(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    resource_intervals = sorted((solution.job_interval_solutions[job.id_job]
                                 for job in instance.jobs
                                 if job.resource_consumption.consumption_by_resource[resource] > 0),
                                key=lambda i: (i.start, i.end))

    periods = []
    current_period = [resource_intervals[0]]
    for interval in resource_intervals[1:]:
        assert interval.start >= current_period[-1].end
        if interval.start > current_period[-1].end:
            periods.append(current_period)
            current_period = [interval]
        else:
            current_period.append(interval)

    auad = machine_workload(solution, instance, resource) / len(periods)
    return auad
