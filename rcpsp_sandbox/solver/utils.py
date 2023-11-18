import itertools
from typing import Iterable, TypeVar, Collection

from docplex.cp.expression import interval_var, CpoIntervalVar
from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoModelSolution, CpoIntervalVarSolution

from rcpsp_sandbox.instances.algorithms import traverse_instance_graph
from rcpsp_sandbox.instances.problem_instance import ProblemInstance, Job


T = TypeVar('T')


def index_groups(groups: Iterable[Collection[T]], keys: Collection[T]) -> dict[T, Collection[T]]:
    """
    Indexes a collection of groups by a set of keys.
    :param groups: An iterable of collections to be indexed.
    :param keys: A collection of keys to index the groups by.
    :return: A dictionary where each key is a key from the input collection and each value is the first group that contains that key.
    :raises KeyError: If a key is not found in any of the groups.
    """
    index: dict[T, Collection[T]] = dict()
    for group in groups:
        for key in keys:
            if key in group:
                index[key] = group
                break
        else:
            raise KeyError("Group does not contain a key")
    return index


def compute_component_jobs(problem_instance: ProblemInstance) -> dict[Job, Collection[Job]]:
    """
    Given a problem instance, returns a dictionary where each key is a root job of a component and the value is a
    collection of jobs that belong to that component.

    :param problem_instance: The problem instance to compute the component jobs for.
    :return: A dictionary where each key is a root job of a component and the value is a collection of jobs that belong
             to that component.
    """
    jobs_by_id = {j.id_job: j for j in problem_instance.jobs}
    jobs_components_grouped =\
        [[jobs_by_id[i[0]] for i in group]
         for _k, group in itertools.groupby(traverse_instance_graph(problem_instance, search="components topological generations",
                                                                    yield_state=True),
                                            key=lambda x: x[1])]  # we assume that the order in which jobs are returned is determined by the components, so we do not sort by component id
    component_jobs_by_root_job = index_groups(jobs_components_grouped,
                                              [jobs_by_id[c.id_root_job] for c in problem_instance.components])
    return component_jobs_by_root_job


def get_solution_job_interval_solutions(solution: CpoModelSolution) -> dict[int, CpoIntervalVarSolution]:
    return {int(var_solution.get_name()[4:]): var_solution
            for var_solution in solution.get_all_var_solutions()
            if isinstance(var_solution, CpoIntervalVarSolution) and var_solution.expr.get_name().startswith("Job")}


def get_model_job_intervals(model: CpoModel) -> dict[int, interval_var]:
    return {int(var.get_name()[4:]): var
            for var in model.get_all_variables()
            if isinstance(var, CpoIntervalVar) and var.get_name().startswith("Job")}