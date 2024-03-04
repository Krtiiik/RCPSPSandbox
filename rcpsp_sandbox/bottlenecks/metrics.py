from typing import Callable, Iterable

import tabulate

from instances.problem_instance import Resource, ProblemInstance
from solver.solution import Solution


T_MetricResult = float
T_Evaluation = dict[Resource, T_MetricResult]


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
             for job in instance.jobs
             if job.resource_consumption.consumption_by_resource[resource] > 0)
    return T_MetricResult(mw)


def machine_utilization_rate(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    def completion_time(j): return solution.job_interval_solutions[j.id_job].end

    mur = (machine_workload(solution, instance, resource) /
           (max(completion_time(job) for job in instance.jobs) - min(completion_time(job) - job.duration for job in instance.jobs))
           )
    return T_MetricResult(mur)


def average_uninterrupted_active_duration(solution: Solution, instance: ProblemInstance, resource: Resource) -> T_MetricResult:
    resource_intervals = sorted((solution.job_interval_solutions[job.id_job]
                                 for job in instance.jobs
                                 if job.resource_consumption.consumption_by_resource[resource] > 0),
                                key=lambda i: (i.start, i.end))

    periods = []
    current_period = [resource_intervals[0]]
    for interval in resource_intervals[1:]:
        if interval.start > current_period[-1].end:
            periods.append(current_period)
            current_period = [interval]
        else:
            current_period.append(interval)
    periods.append(current_period)

    auad = machine_workload(solution, instance, resource) / len(periods)
    return T_MetricResult(auad)
