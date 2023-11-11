import itertools
import math
from collections import defaultdict
from typing import Collection, Iterable, Literal, Tuple

from docplex.cp import modeler
from docplex.cp.expression import interval_var, CpoExpr
from docplex.cp.function import CpoStepFunction
from docplex.cp.model import CpoModel

from instances.algorithms import traverse_instance_graph
from instances.problem_instance import ProblemInstance, Resource, Job
from utils import index_groups


def build_model(problem_instance: ProblemInstance,
                opt: Literal["None", "Tardiness all", "Tardiness selected"] = "None",
                selected: Collection[Job] = None) -> CpoModel:
    """
    Builds a CpoModel for the given problem instance and optimization options.

    Args:
        problem_instance (ProblemInstance): The problem instance to build the model for.
        opt (Literal["None", "Tardiness all", "Tardiness selected"], optional): The optimization option to use. Defaults to "None".
        selected (Collection[Job], optional): The collection of selected jobs to use for the "Tardiness selected" optimization option. Defaults to None.

    Raises:
        ValueError: If an unrecognized optimization option is provided or if the "Tardiness selected" option is used without providing a non-empty collection of selected jobs.

    Returns:
        CpoModel: The built CpoModel.
    """
    model, job_intervals = __base_model(problem_instance)

    if opt not in ["None", "Tardiness all", "Tardiness selected"]:
        raise ValueError(f"Unrecognized optimization option: {opt}")

    if opt == "None":
        pass
    elif opt in ["Tardiness all", "Tardiness selected"]:
        component_jobs_by_root_job = __compute_component_jobs(problem_instance)
        weights_by_id_root_job = {c.id_root_job: c.weight for c in problem_instance.components}

        if opt == "Tardiness all":
            required_component_jobs = component_jobs_by_root_job.items()
        elif opt == "Tardiness selected":
            if selected is None:
                raise ValueError("Tardiness selected optimization option requires a non-empty collection of selected jobs.")
            selected_set = set(selected)
            required_component_jobs = [(root_job, jobs)
                                        for root_job, jobs in component_jobs_by_root_job.items()
                                        if (selected_set & set(jobs))]  # If the intersection of the selected jobs and the jobs in the component is non-empty, the component is required.

        weighted_tardiness = (problem_instance.projects[0].tardiness_cost  # assuming only a single project
                            * __criteria_component_tardiness_sum(required_component_jobs,
                                                                job_intervals,
                                                                weights_by_id_root_job))
        optimization_goal = modeler.minimize(weighted_tardiness)

        model.add(optimization_goal)

    return model


def __base_model(problem_instance: ProblemInstance) -> CpoModel:
    """
    Builds a base model for the given problem instance.

    The base model contains the following constraints:
    - Precedence constraints between jobs.
    - Resource capacity constraints.
    - Job execution availability constraints.

    Args:
        problem_instance (ProblemInstance): The problem instance to build the model for.

    Returns:
        Tuple[CpoModel, Dict[int, IntervalVar]]: A tuple containing the built model and a dictionary of job intervals.
    """
    resource_availabilities = {resource: __build_resource_availability(resource, problem_instance.horizon)
                               for resource in problem_instance.resources}
    job_intervals = {job.id_job: interval_var(name=f"Job {job.id_job}",
                                              size=job.duration,
                                              intensity=__build_job_execution_availability(job,
                                                                                           resource_availabilities))
                     for job in problem_instance.jobs}
    precedence_constraints = [modeler.end_before_start(job_intervals[precedence.id_child], job_intervals[precedence.id_parent])
                              for precedence in problem_instance.precedences]
    jobs_consuming_resource = {resource: [job for job in problem_instance.jobs if job.resource_consumption[resource] > 0]
                               for resource in problem_instance.resources}
    resource_capacity_constraints = [__build_resource_capacity_constraint(resource, jobs_consuming_resource, job_intervals)
                                     for resource in problem_instance.resources]

    model = CpoModel(problem_instance.name)
    model.add(job_intervals.values())
    model.add(precedence_constraints)
    model.add(resource_capacity_constraints)
    return model, job_intervals



def __build_resource_availability(resource: Resource, horizon: int) -> CpoStepFunction:
    """
    Builds a step function representing the availability of a resource over time.

    Args:
        resource (Resource): The resource to build the availability function for.
        horizon (int): The total number of hours in the planning horizon.

    Returns:
        CpoStepFunction: A step function representing the availability of the resource.
    """
    day_operating_hours = resource.availability if resource.availability is not None else [(0, 24)]
    days_count = math.ceil(horizon / 24)
    step_values = dict()
    for i_day in range(days_count):
        day_offset = i_day * 24
        for start, end in day_operating_hours:
            step_values[day_offset + start] = 1
            step_values[day_offset + end] = 0

    steps = sorted(step_values.items())
    return CpoStepFunction(steps)


def __build_resource_capacity_constraint(resource: Resource, jobs_consuming_resource: dict[Resource, list[Job]], job_intervals: dict[int, interval_var],) -> CpoExpr:
    """
    Builds a constraint that ensures the capacity of a given resource is not exceeded by the jobs consuming it.

    The constraint is built as follows: for each job that consumes the resource, a pulse function is created that
    represents the resource consumption of the job. The sum of all pulse functions is then constrained to be less than
    or equal to the resource's capacity. This ensures that the capacity of the resource is not exceeded by the jobs
    consuming it.

    Args:
        resource (Resource): The resource to check the capacity for.
        jobs_consuming_resource (dict[Resource, list[Job]]): A dictionary mapping resources to the jobs that consume them.
        job_intervals (dict[int, interval_var]): A dictionary mapping job IDs to their corresponding interval variables.

    Returns:
        CpoExpr: The constraint expression.
    """
    return (modeler.sum(modeler.pulse(job_intervals[job.id_job], job.resource_consumption[resource])
                        for job in jobs_consuming_resource[resource])
            <= resource.capacity)


def __build_job_execution_availability(job: Job,
                                       resource_availabilities: dict[Resource, CpoStepFunction]) -> CpoStepFunction:
    """
    Builds a CpoStepFunction representing the availability of resources required by a job.

    Args:
        job (Job): The job for which to build the availability function.
        resource_availabilities (dict[Resource, CpoStepFunction]): A dictionary mapping resources to their availability
            functions.

    Returns:
        CpoStepFunction: A CpoStepFunction representing the aggregate availability of resources required by the job.
    """
    used_resources = [resource for resource, consumption in job.resource_consumption.consumption_by_resource.items()
                      if consumption > 0]
    # Time to availability mapping. If a resource is not available at a given time, the value is false; otherwise, it is true.
    step_values = defaultdict(lambda: True)
    for resource in used_resources:
        for step in resource_availabilities[resource].get_step_list():
            time, is_available = step[0], (step[1] == 1)
            step_values[time] &= is_available  # If `resource` is not available at `time`, the job cannot be executed at `time`.

    steps = sorted((step[0], 100 if step[1] else 0)
                   for step in step_values.items())
    return CpoStepFunction(steps)


def __criteria_most_tardy_job_tardiness(jobs: Collection[Job], job_intervals: dict[int, interval_var]) -> CpoExpr:
    """
    Calculates the maximum tardiness of all jobs in the given collection, based on their due dates and completion times.

    The tardiness of a job is defined as the difference between its completion time and its due date, if the job is
    completed after its due date. If no job is completed after its due date, the maximum tardiness is 0.

    Args:
        jobs: A collection of Job objects.
        job_intervals: A dictionary mapping job IDs to their corresponding interval variables.

    Returns:
        A CpoExpr representing the maximum tardiness of all jobs in the collection.
    """
    tardiness_of_jobs = (modeler.end_of(job_intervals[job.id_job]) - job.due_date for job in jobs)
    return modeler.max(modeler.max(tardiness_of_jobs), 0)


def __criteria_component_tardiness_sum(component_jobs: Iterable[Tuple[Job, Collection[Job]]],
                                       job_intervals: dict[int, interval_var],
                                       weights_by_id_root_job: dict[int, int]) -> CpoExpr:
    """
    Computes the sum of the weighted tardiness of the most tardy job in each component.

    Args:
        component_jobs: An iterable of tuples, where each tuple contains a root job and a collection of jobs that belong
            to the same component as the root job.
        job_intervals: A dictionary that maps job IDs to their corresponding interval variables.
        weights_by_id_root_job: A dictionary that maps the IDs of root jobs to their corresponding weights.

    Returns:
        A CpoExpr object representing the sum of the weighted tardiness of the most tardy job in each component.
    """
    return modeler.sum(weights_by_id_root_job[root_job.id_job]  # each component has its weight in the sum
                       * __criteria_most_tardy_job_tardiness(jobs, job_intervals)  # each component contributes only through the tardiest job
                       for root_job, jobs in component_jobs)


def __compute_component_jobs(problem_instance: ProblemInstance) -> dict[Job, Collection[Job]]:
    """
    Given a problem instance, returns a dictionary where each key is a root job of a component and the value is a
    collection of jobs that belong to that component.

    :param problem_instance: The problem instance to compute the component jobs for.
    :return: A dictionary where each key is a root job of a component and the value is a collection of jobs that belong
             to that component.
    """
    jobs_by_id = {j.id_job: j for j in problem_instance.jobs}
    jobs_components_grouped = [[jobs_by_id[i[0]] for i in group]
                               for _k, group in itertools.groupby(traverse_instance_graph(problem_instance, search="components topological generations", yield_state=True),
                                                                  key=lambda x: x[1])]  # we assume that the order in which jobs are returned is determined by the components, so we do not sort by component id
    component_jobs_by_root_job = index_groups(jobs_components_grouped,
                                              [jobs_by_id[c.id_root_job] for c in problem_instance.components])
    return component_jobs_by_root_job
