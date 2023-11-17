import itertools
import math
from collections import defaultdict
from typing import Collection, Iterable, Literal, Tuple, Self

from docplex.cp import modeler
from docplex.cp.expression import interval_var, CpoExpr, CpoIntervalVar
from docplex.cp.function import CpoStepFunction
from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoModelSolution, CpoIntervalVarSolution

from instances.algorithms import traverse_instance_graph
from instances.problem_instance import ProblemInstance, Resource, Job
from utils import index_groups


class ModelBuilder:
    def __init__(self, instance: ProblemInstance, model: CpoModel, job_intervals: dict[int, interval_var]):
        self.instance = instance
        self.model = model
        self.job_intervals = job_intervals

    @staticmethod
    def build_model(instance: ProblemInstance) -> 'ModelBuilder':
        model, job_intervals = ModelBuilder.__base_model(instance)
        return ModelBuilder(instance, model, job_intervals)

    def optimize_model(self,
                       opt: Literal["Tardiness all", "Tardiness selected"] = "Tardiness all",
                       priority_jobs: Collection[Job] = None) -> Self:
        """
        Optimizes the model by adding an optimization goal. The optimization can minimize the total tardiness of all
        job components, tardiness of selected priority job components.

        Args:
            opt (Literal["None", "Tardiness all", "Tardiness selected"], optional): The optimization option to use. Defaults to "None".
            priority_jobs (Collection[Job], optional): The collection of selected jobs to use for the "Tardiness selected" optimization option. Defaults to None.

        Raises:
            ValueError: If an unrecognized optimization option is provided or if the "Tardiness selected" option is used without providing a non-empty collection of selected jobs.
        """
        if opt not in ["Tardiness all", "Tardiness selected"]:
            raise ValueError(f"Unrecognized optimization option: {opt}")

        if opt in ["Tardiness all", "Tardiness selected"]:
            component_jobs_by_root_job = ModelBuilder.__compute_component_jobs(self.instance)
            weights_by_id_root_job = {c.id_root_job: c.weight for c in self.instance.components}

            if opt == "Tardiness all":
                required_component_jobs = component_jobs_by_root_job.items()
            else:  # opt == "Tardiness selected"
                if priority_jobs is None:
                    raise ValueError(
                        "Tardiness selected optimization option requires a non-empty collection of priority jobs.")
                selected_set = set(priority_jobs)
                required_component_jobs = [(root_job, jobs)
                                           for root_job, jobs in component_jobs_by_root_job.items()
                                           if (selected_set & set(jobs))]  # If the intersection of the selected jobs and the jobs in the component is non-empty, the component is required.

            weighted_tardiness = (self.instance.projects[0].tardiness_cost  # assuming only a single project
                                  * ModelBuilder.__criteria_component_tardiness_sum(required_component_jobs,
                                                                                    self.job_intervals,
                                                                                    weights_by_id_root_job))
            optimization_goal = modeler.minimize(weighted_tardiness)

            self.model.add(optimization_goal)

        return self

    def restrain_model_based_on_solution(self,
                                         solution: CpoModelSolution,
                                         eps: float = 1.) -> Self:
        """
        Restrains the given model based on the given solution.

        Job interval variables of the model are restrained to start in an epsilon-neighbourhood of the
        corresponding job intervals in the solution.

        Args:
            solution (CpoModelSolution): The solution to use for restraining the model.
            eps (float, optional): The tolerance for the size of the job intervals. Defaults to 1.
        """
        model_job_intervals, solution_job_interval_solutions = ModelBuilder.__get_model_solution_job_intervals(self.model, solution)

        constraints = []
        for id_job in model_job_intervals.keys():
            interval, interval_solution = model_job_intervals[id_job], solution_job_interval_solutions[id_job]
            lo, hi = interval_solution.get_size() - eps, interval_solution.get_size() + eps
            constraints.append(lo <= modeler.start_of(interval) <= hi)

        self.model.add(constraints)

        return self

    def minimize_model_solution_difference(self, solution: CpoModelSolution) -> Self:
        """
        Minimizes the difference between the start times of the jobs in the given model and solution.

        Args:
            solution (CpoModelSolution): The solution to minimize the difference for.
        """
        model_job_intervals, solution_job_interval_solutions = ModelBuilder.__get_model_solution_job_intervals(self.model, solution)

        criteria = modeler.sum(modeler.abs(modeler.start_of(model_job_intervals[id_job])
                                           - solution_job_interval_solutions[id_job].get_start())
                               for id_job in solution_job_interval_solutions.keys())
        optimization_goal = modeler.minimize(criteria)

        # TODO
        obj_old = self.model.objective
        self.model.remove(obj_old)
        self.model.add(modeler.minimize(obj_old.children[0] + criteria))

        return self

    def get_model(self):
        return self.model

    @staticmethod
    def __base_model(problem_instance: ProblemInstance) -> Tuple[CpoModel, dict[int, interval_var]]:
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
        resource_availabilities = {resource: ModelBuilder.__build_resource_availability(resource, problem_instance.horizon)
                                   for resource in problem_instance.resources}
        job_intervals = {job.id_job: interval_var(name=f"Job {job.id_job}",
                                                  size=job.duration,
                                                  intensity=ModelBuilder.__build_job_execution_availability(job, resource_availabilities))
                         for job in problem_instance.jobs}
        precedence_constraints = [
            modeler.end_before_start(job_intervals[precedence.id_child], job_intervals[precedence.id_parent])
            for precedence in problem_instance.precedences]
        jobs_consuming_resource = {
            resource: [job for job in problem_instance.jobs if job.resource_consumption[resource] > 0]
            for resource in problem_instance.resources}
        resource_capacity_constraints = [
            ModelBuilder.__build_resource_capacity_constraint(resource, jobs_consuming_resource, job_intervals)
            for resource in problem_instance.resources]

        model = CpoModel(problem_instance.name)
        model.add(job_intervals.values())
        model.add(precedence_constraints)
        model.add(resource_capacity_constraints)
        return model, job_intervals

    @staticmethod
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

    @staticmethod
    def __build_resource_capacity_constraint(resource: Resource,
                                             jobs_consuming_resource: dict[Resource, list[Job]],
                                             job_intervals: dict[int, interval_var], ) -> CpoExpr:
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

    @staticmethod
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

    @staticmethod
    def __criteria_most_tardy_job_tardiness(jobs: Collection[Job],
                                            job_intervals: dict[int, interval_var]) -> CpoExpr:
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

    @staticmethod
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
                           * ModelBuilder.__criteria_most_tardy_job_tardiness(jobs, job_intervals)
                           # each component contributes only through the tardiest job
                           for root_job, jobs in component_jobs)

    @staticmethod
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
                                   for _k, group in itertools.groupby(
                traverse_instance_graph(problem_instance, search="components topological generations",
                                        yield_state=True),
                key=lambda x: x[1])]  # we assume that the order in which jobs are returned is determined by the components, so we do not sort by component id
        component_jobs_by_root_job = index_groups(jobs_components_grouped,
                                                  [jobs_by_id[c.id_root_job] for c in problem_instance.components])
        return component_jobs_by_root_job

    @staticmethod
    def __get_model_solution_job_intervals(model: CpoModel, solution: CpoModelSolution) -> Tuple[dict[int, interval_var], dict[int, CpoIntervalVarSolution]]:
        solution_job_interval_solutions: dict[int, CpoIntervalVarSolution] = \
            {int(var_solution.get_name()[4:]): var_solution
             for var_solution in solution.get_all_var_solutions()
             if var_solution is CpoIntervalVarSolution and var_solution.get_name().startswith("Job")}
        model_job_intervals: dict[int, CpoIntervalVar] = \
            {int(var.get_name()[4:]): var
             for var in model.get_all_variables()
             if var is CpoIntervalVar and var.get_name().startswith("Job")}

        return model_job_intervals, solution_job_interval_solutions


def build_model(problem_instance: ProblemInstance) -> ModelBuilder:
    """
    Builds a ModelBuilder which build the optimization model for the given problem instance.

    Args:
        problem_instance (ProblemInstance): The problem instance to build the builder for.

    Returns:
        ModelBuilder: The model builder instance with built base model.
    """
    return ModelBuilder.build_model(problem_instance)
