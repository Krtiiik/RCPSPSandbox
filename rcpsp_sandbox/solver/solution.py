from ast import Tuple
from typing import Iterable

from docplex.cp.solution import CpoSolveResult, CpoModelSolution, CpoIntervalVarSolution

from instances.problem_instance import Job

class Solution:
    _solve_result: CpoSolveResult

    def __init__(self, solve_result: CpoSolveResult):
        self._solve_result = solve_result


    def get_job_interval_solutions(self) -> dict[int, CpoIntervalVarSolution]:
        if self._solve_result is None:
            return None

        return {int(var_solution.get_name()[4:]): var_solution
                for var_solution in self._solve_result.get_all_var_solutions()
                if isinstance(var_solution, CpoIntervalVarSolution) and var_solution.expr.get_name().startswith("Job")}

    def difference_to(self, other: 'Solution', selected_jobs: Iterable[Job] = None) -> Tuple[int, dict[int, int]]:
        return solution_difference(self, other, selected_jobs)


def solution_difference(a: Solution,
                        b: Solution,
                        selected_jobs: Iterable[Job] = None) -> Tuple[int, dict[int, int]]:
    a_interval_solutions = a.get_job_interval_solutions()
    b_interval_solutions = b.get_job_interval_solutions()
    assert a_interval_solutions.keys() == b_interval_solutions.keys()

    job_ids = ((j.id_job for j in selected_jobs) if selected_jobs is not None
               else a_interval_solutions.keys())
    differences = {job_id: a_interval_solutions[job_id].get_end() - b_interval_solutions[job_id].get_end()
                   for job_id in job_ids}
    difference = sum(abs(diff) for diff in differences.values())
    return difference, differences

