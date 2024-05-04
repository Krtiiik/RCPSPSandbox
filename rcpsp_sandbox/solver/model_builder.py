import math
from collections import defaultdict
from typing import Iterable, Literal, Tuple, Self

from docplex.cp import modeler
from docplex.cp.catalog import Oper_end_before_start, Oper_less_or_equal
from docplex.cp.expression import interval_var, CpoExpr, CpoFunctionCall, CpoIntervalVar
from docplex.cp.function import CpoStepFunction
from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoIntervalVarSolution

from instances.problem_instance import ProblemInstance, Resource, Job, Component, compute_resource_availability
from solver.solution import Solution


class ModelBuilder:
    """
    A class that builds a model for the given problem instance.
    """

    instance: ProblemInstance
    model: CpoModel

    _job_intervals: dict[int, interval_var]
    _precedence_constraints: list[CpoExpr]
    _resource_capacity_constraints: list[CpoExpr]

    def __init__(self, instance: ProblemInstance, model: CpoModel):
        """
        Initializes a new instance of the ModelBuilder class.

        Args:
            instance (ProblemInstance): The problem instance.
            model (CpoModel): The model being built.
        """
        self.instance: ProblemInstance = instance
        self.model: CpoModel = model

        self._job_intervals: dict[int, interval_var] = dict()
        self._precedence_constraints: list[CpoExpr] = []
        self._resource_capacity_constraints: list[CpoExpr] = []

    @staticmethod
    def build_model(instance: ProblemInstance) -> 'ModelBuilder':
        """
        Builds a model for the given problem instance.

        Args:
            instance (ProblemInstance): The problem instance.

        Returns:
            ModelBuilder: The ModelBuilder instance.
        """
        return ModelBuilder(instance, CpoModel(instance.name)).__base_model()

    def with_precedences(self) -> Self:
        """
        Adds precedence constraints to the model.

        Returns:
            Self: The ModelBuilder instance.
        """
        precedence_constraints = [
            modeler.end_before_start(self._job_intervals[precedence.id_child],
                                     self._job_intervals[precedence.id_parent])
            for precedence in self.instance.precedences]

        self.model.add(precedence_constraints)

        self._precedence_constraints = precedence_constraints

        return self

    def with_resource_constraints(self) -> Self:
        """
        Adds resource capacity constraints to the model.

        Returns:
            Self: The ModelBuilder instance.
        """
        resource_capacity_constraints = [
            self.__build_resource_capacity_constraint(resource, self.instance, self._job_intervals, self.instance.horizon)
            for resource in self.instance.resources]

        self.model.add(resource_capacity_constraints)

        self._resource_capacity_constraints = resource_capacity_constraints

        return self

    def optimize_model(self,
                       opt: Literal["tardiness"] = "tardiness",
                       selected: Iterable[Component] = None,
                       ) -> Self:
        """
        Sets the optimization goal for the model.

        Args:
            opt (Literal["tardiness"], optional): The optimization option. Defaults to "tardiness".
            selected (Iterable[Component], optional): The selected components. Defaults to None.

        Returns:
            Self: The ModelBuilder instance.
        """
        if opt not in ["tardiness"]:
            raise ValueError(f"Unrecognized optimization option: {opt}")

        components = list(selected) if selected is not None else self.instance.components

        def tardiness(c: Component):
            value = modeler.end_of(self._job_intervals[c.id_root_job]) - self.instance.jobs_by_id[c.id_root_job].due_date
            return modeler.max(0, value)

        weighted_tardiness = modeler.sum(component.weight * tardiness(component) for component in components)
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

    def add_hot_start(self,
                      solution: Solution = None,
                      job_interval_solutions: dict[int, tuple[int, int]] = None,
                      ) -> Self:
        """
        Adds a hot start to the model.

        Args:
            solution (Solution, optional): The initial solution to start the model from. Defaults to None.
            job_interval_solutions (dict[int, tuple[int, int]], optional): A dictionary mapping job IDs to tuples
                representing the start and end intervals for each job. Defaults to None.

        Returns:
            Self: The current instance of the ModelBuilder.
        """
        add_hot_start(self.model, solution=solution, job_interval_solutions=job_interval_solutions)
        return self

    def minimize_model_solution_difference(self, solution: Solution, excluded: Iterable[Job] = None, alpha: float = 1.) -> Self:
        """
        Minimizes the difference between the start times of the jobs in the given model and solution.

        Args:
            solution (Solution): The solution to minimize the difference for.
            excluded (Iterable[Job], optional): A collection of jobs to exclude from the minimization. Defaults to None.
            alpha (float, optional): The weight of the difference in the minimization sum. Defaults to 1.

        Returns:
            Self: The current instance of the model builder.

        Notes:
            This method minimizes the difference between the start times of the jobs in the given model and a provided solution.
            It calculates the difference for each job in the model, excluding any jobs specified in the `excluded` parameter.
            The difference is weighted by the `alpha` parameter and summed up to form the criteria for minimization.
            The original optimization value of the model is extracted and combined with the new criteria for minimization.
            This method assumes that the optimization of the current model is a minimization of a value.
            If new optimizations, especially not a minimization, are added, this method will need to be reworked.
            Please use this method with caution as it performs an implementation-dependent operation.
        """
        job_ids = set(j.id_job for j in self.instance.jobs) - (set(excluded) if excluded else set())

        def difference_for(id_job): return modeler.abs(modeler.start_of(self._job_intervals[id_job])
                                                       - solution.job_interval_solutions[id_job].get_start())
        criteria = modeler.sum(difference_for(id_job) for id_job in job_ids)

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

    def get_model(self):
        """
        Returns the built model.
        """
        return self.model

    def __base_model(self) -> Self:
        """
        Builds a base model for the given problem instance.
        """
        job_intervals = {job.id_job: interval_var(name=f"Job {job.id_job}", end=(0, self.instance.horizon) ,size=job.duration, length=job.duration)
                         for job in self.instance.jobs}

        model = CpoModel(self.instance.name)
        model.add(job_intervals.values())

        self.model = model
        self._job_intervals = job_intervals

        return self

    def __build_resource_capacity_constraint(self,
                                             resource: Resource,
                                             instance: ProblemInstance,
                                             job_intervals: dict[int, interval_var],
                                             horizon: int,
                                             ) -> CpoExpr:
        """
        Builds a constraint that ensures the capacity of a given resource is not exceeded by the jobs consuming it.

        The constraint is built as follows: for each job that consumes the resource, a pulse function is created that
        represents the resource consumption of the job. The sum of all pulse functions is then constrained to be less than
        or equal to the resource's capacity. This ensures that the capacity of the resource is not exceeded by the jobs
        consuming it.

        Args:
            resource (Resource): The resource to check the capacity for.
            instance (ProblemInstance): The problem instance.
            job_intervals (dict[int, interval_var]): A dictionary mapping job IDs to their corresponding interval variables.
            horizon (int): The horizon of the problem instance.

        Returns:
            CpoExpr: The constraint expression.
        """
        jobs_consuming_resource = [job for job in self.instance.jobs if job.resource_consumption[resource] > 0]
        capacity_intervals = compute_resource_availability(resource, instance, horizon)
        max_capacity = max(resource.capacity, max(i[2] for i in capacity_intervals))
        consumption_pulses = [modeler.pulse(job_intervals[job.id_job], job.resource_consumption[resource])
                              for job in jobs_consuming_resource]

        blocking_pulses = [modeler.pulse((start, end), max_capacity - capacity)
                           for start, end, capacity in capacity_intervals
                           if capacity < max_capacity]

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
    def __get_model_solution_job_intervals(model: CpoModel, solution: Solution) -> Tuple[dict[int, interval_var], dict[int, CpoIntervalVarSolution]]:
        model_job_intervals = get_model_job_intervals(model)
        solution_job_interval_solutions = solution.job_interval_solutions

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

    Returns:
        ModelBuilder: The model builder instance with the given model.
    """
    builder = ModelBuilder(problem_instance, model)
    builder._job_intervals = get_model_job_intervals(model)
    builder._precedence_constraints = [expr for expr, _loc in model.get_all_expressions() if isinstance(expr, CpoFunctionCall) and expr.operation == Oper_end_before_start]
    builder._resource_capacity_constraints = [expr for expr, _loc in model.get_all_expressions() if isinstance(expr, CpoFunctionCall) and expr.operation == Oper_less_or_equal]
    return builder


def add_hot_start(model: CpoModel,
                  solution: Solution = None,
                  job_interval_solutions: dict[int, tuple[int, int]] = None,
                  ) -> CpoModel:
    """
    Adds a hot start to the given CpoModel.

    Args:
        model (CpoModel): The CpoModel to add the hot start to.
        solution (Solution, optional): The solution object containing the hot start values. Defaults to None.
        job_interval_solutions (dict[int, tuple[int, int]], optional): A dictionary mapping job IDs to their interval solutions. 
            Each interval solution is represented as a tuple of start and end times. Defaults to None.

    Returns:
        CpoModel: The CpoModel with the hot start added.
    """
    if job_interval_solutions is None:
        if solution is None:
            raise ValueError("No hot start values given")
        job_interval_solutions = {job_id: (interval_solution.start, interval_solution.end)
                                  for job_id, interval_solution in solution.job_interval_solutions.items()}

    job_intervals = get_model_job_intervals(model)
    hot_start = model.create_empty_solution()
    for job_id, (start, end) in job_interval_solutions.items():
        hot_start.add_interval_var_solution(job_intervals[job_id], presence=True, start=start, end=end)

    model.set_starting_point(hot_start)

    return model


def get_model_job_intervals(model: CpoModel) -> dict[int, interval_var]:
    """
    Returns a dictionary mapping job IDs to their corresponding interval variables in the given model.

    Args:
        model (CpoModel): The model containing the interval variables.

    Returns:
        dict[int, interval_var]: A dictionary mapping job IDs (extracted from the variable names) to their corresponding interval variables.
    """
    return {int(var.get_name()[4:]): var
            for var in model.get_all_variables()
            if isinstance(var, CpoIntervalVar) and var.get_name().startswith("Job")}
