from typing import Iterable, Tuple

from docplex.cp.solution import CpoSolveResult, CpoIntervalVarSolution

from instances.problem_instance import Job, ProblemInstance, compute_component_jobs


class Solution:
    _solve_result: CpoSolveResult
    _instance: ProblemInstance

    _cached_job_interval_solutions: dict[int, CpoIntervalVarSolution] = None
    _cached_tardiness: dict[int, int] = None
    _cached_weighted_tardiness: dict[int, int] = None

    def __init__(self, solve_result: CpoSolveResult, instance: ProblemInstance):
        if solve_result is None or not solve_result.is_solution():
            raise ValueError("Cannot wrap a non-solution result")

        self._solve_result = solve_result
        self._instance = instance

    def difference_to(self, other: 'Solution', selected_jobs: Iterable[Job] = None) -> Tuple[int, dict[int, int]]:
        return solution_difference(self, other, selected_jobs)

    def plot(self, *args, **kwargs):
        from solver.drawing import plot_solution
        plot_solution(self.instance, self, *args, **kwargs)

    @property
    def job_interval_solutions(self) -> dict[int, CpoIntervalVarSolution]:
        if self._cached_job_interval_solutions is None:
            self._cached_job_interval_solutions = \
                {int(var_solution.get_name()[4:]): var_solution
                 for var_solution in self._solve_result.get_all_var_solutions()
                 if isinstance(var_solution, CpoIntervalVarSolution) and var_solution.expr.get_name().startswith("Job")}

        return self._cached_job_interval_solutions

    @property
    def solve_result(self) -> CpoSolveResult:
        return self._solve_result

    @property
    def instance(self) -> ProblemInstance:
        return self._instance

    def tardiness(self, job_id=None) -> int | dict[int, int]:
        if self._cached_tardiness is None:
            self._cached_tardiness = compute_job_tardiness(self)

        return self._cached_tardiness if job_id is None else self._cached_tardiness[job_id]

    def weighted_tardiness(self, job_id=None) -> int | dict[int, int]:
        if self._cached_weighted_tardiness is None:
            self._cached_weighted_tardiness = compute_job_weighted_tardiness(self)

        return self._cached_weighted_tardiness if job_id is None else self._cached_weighted_tardiness[job_id]


def solution_difference(a: Solution,
                        b: Solution,
                        selected_jobs: Iterable[Job] = None) -> Tuple[int, dict[int, int]]:
    """
    Computes the difference between two solutions. The difference is the sum of the absolute values of the differences
    between the end times of the jobs in the two solutions.

    :param a: The first solution.
    :param b: The second solution.
    :param selected_jobs: The jobs to compute the difference for. If None, all jobs are used.\

    :return: A tuple where the first element is the difference and the second element is a dictionary where each key is
    the id of a job and each value is the difference between the end times of that job in the two solutions.
    """
    a_interval_solutions = a.job_interval_solutions
    b_interval_solutions = b.job_interval_solutions
    assert a_interval_solutions.keys() == b_interval_solutions.keys()

    job_ids = ((j.id_job for j in selected_jobs) if selected_jobs is not None
               else a_interval_solutions.keys())
    differences = {job_id: a_interval_solutions[job_id].get_end() - b_interval_solutions[job_id].get_end()
                   for job_id in job_ids}
    difference = sum(abs(diff) for diff in differences.values())
    return difference, differences


def compute_job_tardiness(solution: Solution,
                          selected_jobs: Iterable[Job] = None,
                          ) -> dict[int, int]:
    interval_solutions = solution.job_interval_solutions
    job_ids = ((j.id_job for j in selected_jobs) if selected_jobs is not None
               else interval_solutions.keys())
    jobs_by_id = {j.id_job: j for j in solution.instance.jobs}

    job_tardiness = dict()
    for job_id in job_ids:
        due_date = jobs_by_id[job_id].due_date
        completion_time = interval_solutions[job_id].get_end()
        job_tardiness[job_id] = max(0, completion_time - due_date)

    return job_tardiness

def compute_job_weighted_tardiness(solution: Solution,
                                   job_tardiness: dict[int, int] = None,
                                   selected_jobs: Iterable[Job] = None,
                                   ) -> dict[int, int]:
    job_ids = (j.id_job for j in (selected_jobs if selected_jobs is not None else solution.instance.jobs))
    if job_tardiness is None:
        job_tardiness = compute_job_tardiness(solution, selected_jobs)

    components_by_id_root_job = {c.id_root_job: c for c in solution.instance.components}
    job_tardiness_weight = {j.id_job: components_by_id_root_job[root_job.id_job].weight
                            for root_job, jobs in compute_component_jobs(solution.instance).items()
                            for j in jobs}

    job_weighted_tardiness = {j_id: job_tardiness_weight[j_id] * job_tardiness[j_id] for j_id in job_ids}
    return job_weighted_tardiness
