from collections import namedtuple
from typing import Iterable, Tuple, Self

from docplex.cp.solution import CpoSolveResult, CpoIntervalVarSolution

from instances.problem_instance import Job, ProblemInstance, compute_component_jobs


IntervalSolution = namedtuple("IntervalSolution", ("start", "end"))


class Solution:
    """
    Represents a solution to the problem instance in the RCPSPSandbox solver.

    Attributes:
        instance (ProblemInstance): The problem instance associated with the solution.
        job_interval_solutions (dict[int, IntervalSolution]): A dictionary mapping job IDs to interval solutions.
    """

    def __init__(self, instance: ProblemInstance, job_interval_solutions: dict[int, IntervalSolution]):
        """
        Initializes a new instance of the Solution class.

        Args:
            instance (ProblemInstance): The problem instance associated with the solution.
            job_interval_solutions (dict[int, IntervalSolution]): A dictionary mapping job IDs to interval solutions.
        """
        self._instance: ProblemInstance = instance
        self._interval_solutions: dict[int, IntervalSolution] = job_interval_solutions

        self._cached_tardiness: dict[int, int] | None = None
        self._cached_weighted_tardiness: dict[int, int] | None = None

    def difference_to(self, other: Self, selected_jobs: Iterable[Job] = None) -> Tuple[int, dict[int, int]]:
        """
        Calculates the difference between this solution and another solution.

        Args:
            other (Solution): The other solution to compare with.
            selected_jobs (Iterable[Job], optional): The selected jobs to consider when calculating the difference.
                Defaults to None.

        Returns:
            Tuple[int, dict[int, int]]: A tuple containing the total difference and a dictionary mapping job IDs to
                their individual differences.
        """
        return solution_difference(self, other, selected_jobs)

    def plot(self, *args, **kwargs):
        """
        Plots the solution.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        from solver.drawing import plot_solution
        plot_solution(self.instance, self, *args, **kwargs)

    @property
    def job_interval_solutions(self) -> dict[int, IntervalSolution]:
        """
        Gets the dictionary of job interval solutions.

        Returns:
            dict[int, IntervalSolution]: A dictionary mapping job IDs to interval solutions.
        """
        return self._interval_solutions

    @property
    def instance(self) -> ProblemInstance:
        """
        Gets the problem instance associated with the solution.

        Returns:
            ProblemInstance: The problem instance.
        """
        return self._instance

    def tardiness(self, job_id=None) -> int | dict[int, int]:
        """
        Calculates the tardiness of the solution.

        Args:
            job_id (int, optional): The ID of the job to calculate the tardiness for.
                If not specified, returns the tardiness for all jobs. Defaults to None.

        Returns:
            int | dict[int, int]: The tardiness of the solution. If job_id is specified, returns the tardiness
                for that particular job.
        """
        if self._cached_tardiness is None:
            self._cached_tardiness = compute_job_tardiness(self)

        return self._cached_tardiness if job_id is None else self._cached_tardiness[job_id]

    def weighted_tardiness(self, job_id=None) -> int | dict[int, int]:
        """
        Calculates the weighted tardiness of the solution.

        Args:
            job_id (int, optional): The ID of the job to calculate the weighted tardiness for.
                If not specified, returns the weighted tardiness for all jobs. Defaults to None.

        Returns:
            int | dict[int, int]: The weighted tardiness of the solution. If job_id is specified, returns the
                weighted tardiness for that particular job.
        """
        if self._cached_weighted_tardiness is None:
            self._cached_weighted_tardiness = compute_job_weighted_tardiness(self)

        return self._cached_weighted_tardiness if job_id is None else self._cached_weighted_tardiness[job_id]


class ModelSolution(Solution):
    """
    Represents a model solution for a given problem instance.

    Args:
        instance (ProblemInstance): The problem instance.
        solve_result (CpoSolveResult): The solve result containing the solution.

    Raises:
        ValueError: If the solve result is None or not a valid solution.

    Attributes:
        solve_result (CpoSolveResult): The solve result associated with the solution.
    """

    def __init__(self, instance: ProblemInstance, solve_result: CpoSolveResult):
        if solve_result is None or not solve_result.is_solution():
            raise ValueError("Cannot wrap a non-solution result")

        super().__init__(
            instance,
            {int(var_solution.get_name()[4:]): var_solution
             for var_solution in solve_result.get_all_var_solutions()
             if isinstance(var_solution, CpoIntervalVarSolution) and var_solution.expr.get_name().startswith("Job")
             })

        self._solve_result = solve_result

    @property
    def solve_result(self) -> CpoSolveResult:
        """
        Get the solve result associated with the solution.

        Returns:
            CpoSolveResult: The solve result.
        """
        return self._solve_result


class ExplicitSolution(Solution):
    """
    Represents an explicit solution for a problem instance.
    """

    def __init__(self, instance: ProblemInstance, job_interval_solutions: dict[int, IntervalSolution]):
        """
        Initializes a Solution object.

        Args:
            instance (ProblemInstance): The problem instance.
            job_interval_solutions (dict[int, IntervalSolution]): A dictionary mapping job IDs to interval solutions.
        """
        super().__init__(instance, job_interval_solutions)


def solution_difference(a: Solution,
                        b: Solution,
                        selected_jobs: Iterable[Job] = None) -> Tuple[int, dict[int, int]]:
    """
    Calculates the difference between two solutions.

    Args:
        a (Solution): The first solution.
        b (Solution): The second solution.
        selected_jobs (Iterable[Job], optional): A collection of selected jobs to consider. Defaults to None.

    Returns:
        Tuple[int, dict[int, int]]: A tuple containing the total difference and a dictionary of differences for each job.
    """
    a_interval_solutions = a.job_interval_solutions
    b_interval_solutions = b.job_interval_solutions
    assert a_interval_solutions.keys() == b_interval_solutions.keys()

    job_ids = ((j.id_job for j in selected_jobs) if selected_jobs is not None
               else a_interval_solutions.keys())
    differences = {job_id: a_interval_solutions[job_id].end - b_interval_solutions[job_id].end
                   for job_id in job_ids}
    difference = sum(abs(diff) for diff in differences.values())
    return difference, differences


def compute_job_tardiness(solution: Solution,
                          selected_jobs: Iterable[Job] = None,
                          ) -> dict[int, int]:
    """
    Computes the tardiness of each job in the solution.

    Args:
        solution (Solution): The solution object containing the job interval solutions.
        selected_jobs (Iterable[Job], optional): The selected jobs to compute tardiness for. 
            If None, computes tardiness for all jobs in the solution. Defaults to None.

    Returns:
        dict[int, int]: A dictionary mapping job IDs to their tardiness values.
    """
    interval_solutions = solution.job_interval_solutions
    job_ids = ((j.id_job for j in selected_jobs) if selected_jobs is not None
               else interval_solutions.keys())
    jobs_by_id = {j.id_job: j for j in solution.instance.jobs}

    job_tardiness = dict()
    for job_id in job_ids:
        due_date = jobs_by_id[job_id].due_date
        completion_time = interval_solutions[job_id].end
        job_tardiness[job_id] = max(0, completion_time - due_date)

    return job_tardiness

def compute_job_weighted_tardiness(solution: Solution,
                                   job_tardiness: dict[int, int] = None,
                                   selected_jobs: Iterable[Job] = None,
                                   ) -> dict[int, int]:
    """
    Computes the weighted tardiness for each job in the solution.

    Args:
        solution (Solution): The solution for which to compute the weighted tardiness.
        job_tardiness (dict[int, int], optional): A dictionary mapping job IDs to their tardiness values.
            If not provided, the tardiness values will be computed using the `compute_job_tardiness` function.
        selected_jobs (Iterable[Job], optional): An iterable of selected jobs for which to compute the weighted tardiness.
            If not provided, all jobs in the solution instance will be considered.

    Returns:
        dict[int, int]: A dictionary mapping job IDs to their weighted tardiness values.
    """
    job_ids = (j.id_job for j in (selected_jobs if selected_jobs is not None else solution.instance.jobs))
    if job_tardiness is None:
        job_tardiness = compute_job_tardiness(solution, selected_jobs)

    components_by_id_root_job = {c.id_root_job: c for c in solution.instance.components}
    job_tardiness_weight = {j.id_job: components_by_id_root_job[root_job.id_job].weight
                            for root_job, jobs in compute_component_jobs(solution.instance).items()
                            for j in jobs}

    job_weighted_tardiness = {j_id: job_tardiness_weight[j_id] * job_tardiness[j_id] for j_id in job_ids}
    return job_weighted_tardiness
