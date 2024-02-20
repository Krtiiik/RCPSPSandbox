import math
from collections import defaultdict
from typing import Collection, Iterable, Literal, Tuple, Self

from docplex.cp import modeler
from docplex.cp.catalog import Oper_end_before_start, Oper_less_or_equal
from docplex.cp.expression import interval_var, CpoExpr, CpoFunctionCall
from docplex.cp.function import CpoStepFunction
from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoIntervalVarSolution

from instances.problem_instance import ProblemInstance, Resource, Job, Component
from solver.solution import Solution
from solver.utils import compute_component_jobs, get_model_job_intervals


class ModelBuilder:
    instance: ProblemInstance
    model: CpoModel

    _job_intervals: dict[int, interval_var]
    _precedence_constraints: list[CpoExpr]
    _resource_capacity_constraints: list[CpoExpr]
    _resource_capacity_changes_intervals: dict[Resource, list[Tuple[int, int, interval_var]]]

    def __init__(self, instance: ProblemInstance, model: CpoModel):
        self.instance = instance
        self.model = model

        self._job_intervals = dict()
        self._precedence_constraints = []
        self._resource_capacity_constraints = []
        self._resource_capacity_changes_intervals = dict()

    @staticmethod
    def build_model(instance: ProblemInstance) -> 'ModelBuilder':
        if instance.components is None or instance.components == []:
            instance.components = [Component(1, 1)]
        return ModelBuilder(instance, CpoModel(instance.name)).__base_model(instance)

    def optimize_model(self,
                       opt: Literal["Tardiness all", "Tardiness selected"] = "Tardiness all",
                       priority_jobs: Iterable[Job] = None) -> Self:
        """
        Optimizes the model by adding an optimization goal. The optimization can minimize the total tardiness of all
        job components, tardiness of selected priority job components.

        Args:
            opt (Literal["None", "Tardiness all", "Tardiness selected"], optional): The optimization option to use. Defaults to "None".
            priority_jobs (Iterable[Job], optional): The collection of selected jobs to use for the "Tardiness selected" optimization option. Defaults to None.

        Raises:
            ValueError: If an unrecognized optimization option is provided or if the "Tardiness selected" option is used without providing a non-empty collection of selected jobs.
        """
        if opt not in ["Tardiness all", "Tardiness selected"]:
            raise ValueError(f"Unrecognized optimization option: {opt}")

        if opt in ["Tardiness all", "Tardiness selected"]:
            component_jobs_by_root_job = compute_component_jobs(self.instance)
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
                                                                                    self._job_intervals,
                                                                                    weights_by_id_root_job))
            optimization_goal = modeler.minimize(weighted_tardiness)

            self.model.add(optimization_goal)

        return self

    def restrain_model_based_on_solution(self,
                                         solution: Solution,
                                         exclude: Iterable[Job] = None,
                                         eps: float = 1.) -> Self:
        """
        Restrains the given model based on the given solution.

        Job interval variables of the model are restrained to start in an epsilon-neighbourhood of the
        corresponding job intervals in the solution.

        Args:
            solution (Solution): The solution to use for restraining the model.
            exclude (Iterable[Job], optional): The collection of jobs to exclude from the restraint. Defaults to None.
            eps (float, optional): The tolerance for the size of the job intervals. Defaults to 1.
        """
        model_job_intervals, solution_job_interval_solutions = ModelBuilder.__get_model_solution_job_intervals(self.model, solution)
        excluded = set(j.id_job for j in exclude) if exclude is not None else {}

        constraints = []
        to_restrain = (j for j in model_job_intervals.keys() if j not in excluded)
        for id_job in to_restrain:
            interval, interval_solution = model_job_intervals[id_job], solution_job_interval_solutions[id_job]
            lo, hi = interval_solution.get_start() - eps, interval_solution.get_start() + eps
            constraints.append(modeler.start_of(interval) >= lo)
            constraints.append(modeler.start_of(interval) <= hi)

        self.model.add(constraints)

        return self

    def minimize_model_solution_difference(self, solution: Solution, exclude: Iterable[Job] = None, alpha: float = 1.) -> Self:
        """
        Minimizes the difference between the start times of the jobs in the given model and solution.

        Args:
            solution (Solution): The solution to minimize the difference for.
            exclude (Iterable[Job], optional): The collection of jobs to exclude from the minimization. Defaults to None.
            alpha (float, optional): The weight of the difference in the minimization sum. Defaults to 1.
        """
        excluded = set(j.id_job for j in exclude) if exclude is not None else {}
        model_job_intervals, solution_job_interval_solutions = ModelBuilder.__get_model_solution_job_intervals(self.model, solution)

        criteria = modeler.sum(modeler.abs(modeler.start_of(model_job_intervals[id_job])
                                           - solution_job_interval_solutions[id_job].get_start())
                               for id_job in solution_job_interval_solutions.keys()
                               if id_job not in excluded)

        # !!! This is a nasty piece of code !!!
        #
        # We assume that the optimization of the current model is a minimization of a value.
        # We then extract the value, sum it with the new criteria and minimize that new sum.
        # Should new optimization, especially not a minimization, get added, this will need
        # a rework.
        #
        # It is also a dangerous implementation-dependent operation as the original
        # optimization value is extracted from inside the CpoFunctionCall class.
        obj_old = self.model.objective
        obj_old_expr = obj_old.children[0]
        self.model.remove(obj_old)
        self.model.add(modeler.minimize(obj_old_expr + (alpha * criteria)))

        return self

    def change_resource_capacities(self, changes: dict[Resource, Iterable[Tuple[int, int, int]]]) -> Self:
        """
        Changes the capacities of the given resources in the model.

        Args:
            changes (dict[Resource, Tuple[int, int, int]]): A dictionary mapping resources to their new capacities. The
                new capacities are represented as tuples (start, end, capacity), where start and end are the times when
                the capacity change occurs.

        Returns:
            ModelBuilder: The modified model builder.
        """

        jobs_consuming_resource = {
            resource: [job for job in self.instance.jobs if job.resource_consumption[resource] > 0]
            for resource in self.instance.resources}
        variable_resource_capacity_constraints = [
            self.__build_resource_capacity_constraint(resource, jobs_consuming_resource, self._job_intervals, changes[resource] if resource in changes else None)
            for resource in self.instance.resources]

        self.model.remove(self._resource_capacity_constraints)
        self.model.add(variable_resource_capacity_constraints)

        self._resource_capacity_constraints = variable_resource_capacity_constraints

        for resource in changes:
            if resource not in self._resource_capacity_changes_intervals:
                self._resource_capacity_changes_intervals[resource] = []
            self._resource_capacity_changes_intervals[resource] += changes[resource]

        return self

    def get_model(self):
        return self.model

    def __base_model(self, problem_instance: ProblemInstance) -> Self:
        """
        Builds a base model for the given problem instance.

        The base model contains the following constraints:
        - Precedence constraints between jobs.
        - Resource capacity constraints.
        - Job execution availability constraints.

        Args:
            problem_instance (ProblemInstance): The problem instance to build the model for.
        """
        resource_availabilities = {
            resource: ModelBuilder.__build_resource_availability(resource, problem_instance.horizon)
            for resource in problem_instance.resources}

        job_intervals = {
            job.id_job: interval_var(name=f"Job {job.id_job}",
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
            self.__build_resource_capacity_constraint(resource, jobs_consuming_resource, job_intervals, None)
            for resource in problem_instance.resources]

        model = CpoModel(problem_instance.name)
        model.add(job_intervals.values())
        model.add(precedence_constraints)
        model.add(resource_capacity_constraints)

        self.model = model
        self._job_intervals = job_intervals
        self._precedence_constraints = precedence_constraints
        self._resource_capacity_constraints = resource_capacity_constraints

        return self

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
        day_operating_hours = resource.availability if resource.availability is not None else [(0, 24, resource.capacity)]
        days_count = math.ceil(horizon / 24)
        step_values = dict()
        for i_day in range(days_count):
            day_offset = i_day * 24
            for start, end, capacity in day_operating_hours:
                step_values[day_offset + start] = capacity
                step_values[day_offset + end] = 0

        steps = sorted(step_values.items())
        return CpoStepFunction(steps)

    def __build_resource_capacity_constraint(self,
                                             resource: Resource,
                                             jobs_consuming_resource: dict[Resource, list[Job]],
                                             job_intervals: dict[int, interval_var],
                                             capacity_changes: Iterable[Tuple[int,int,int]]) -> CpoExpr:
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
            capacity_changes (Iterable[Tuple[int,int,int]], optional): An iterable of tuples representing capacity changes

        Returns:
            CpoExpr: The constraint expression.
        """
        max_capacity = max(resource.capacity, max(change[2] for change in capacity_changes) if capacity_changes else 0)
        consumption_pulses = [modeler.pulse(job_intervals[job.id_job], job.resource_consumption[resource])
                              for job in jobs_consuming_resource[resource]]

        blocking_pulses = []
        if capacity_changes is not None:
            capacity_changes = sorted(capacity_changes)
            increases = [change for change in capacity_changes if change[2] > resource.capacity]
            decreases = [change for change in capacity_changes if change[2] < resource.capacity]

            # Capacity increases
            # Constructs a blocking pulse function spanning (possibly some) decreases
            last_end = 0
            last_capacity = resource.capacity
            for start, end, capacity in increases:
                if start > last_end:
                    blocking_pulses.append(modeler.pulse((last_end, start), max_capacity - last_capacity))
                blocking_pulses.append(modeler.pulse((start, end), max_capacity - capacity))

                last_end = end
                last_capacity = capacity

            if last_end < self.instance.horizon:
                blocking_pulses.append(modeler.pulse((last_end, self.instance.horizon), last_capacity - resource.capacity))

            # Capacity decreases
            for start, end, capacity in decreases:
                blocking_pulses.append(modeler.pulse((start, end), resource.capacity - capacity))

        return (modeler.sum(consumption_pulses + blocking_pulses)
                <= max_capacity)

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
                time, is_available = step[0], (step[1] > 0)
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
    def __get_model_solution_job_intervals(model: CpoModel, solution: Solution) -> Tuple[dict[int, interval_var], dict[int, CpoIntervalVarSolution]]:
        model_job_intervals = get_model_job_intervals(model)
        solution_job_interval_solutions = solution.job_interval_solutions()

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


def edit_model(model: CpoModel, problem_instance: ProblemInstance) -> ModelBuilder:
    """
    Edits the given model.

    Args:
        model (CpoModel): The model to edit.
        problem_instance (ProblemInstance): The problem instance of the model.
    """
    builder = ModelBuilder(problem_instance, model)
    builder._job_intervals = get_model_job_intervals(model)
    builder._precedence_constraints = [expr for expr, _loc in model.get_all_expressions() if isinstance(expr, CpoFunctionCall) and expr.operation == Oper_end_before_start]
    builder._resource_capacity_constraints = [expr for expr, _loc in model.get_all_expressions() if isinstance(expr, CpoFunctionCall) and expr.operation == Oper_less_or_equal]
    return builder
