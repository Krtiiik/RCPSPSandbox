from typing import Iterable, Tuple

from docplex.cp.solution import CpoModelSolution

from rcpsp_sandbox.instances.problem_instance import ProblemInstance, Job
from rcpsp_sandbox.solver.utils import get_solution_job_interval_solutions, compute_component_jobs


def solution_tardiness_value(solution: CpoModelSolution,
                             instance: ProblemInstance,
                             selected_jobs: Iterable[Job] = None):
    interval_solutions = get_solution_job_interval_solutions(solution)
    job_ids = ((j.id_job for j in selected_jobs) if selected_jobs is not None
               else interval_solutions.keys())
    component_jobs = compute_component_jobs(instance)

    components_by_id_root_job = {c.id_root_job: c for c in instance.components}
    job_component_id_root_job = {j: root_job.id_job
                                 for root_job, jobs in component_jobs.items()
                                 for j in jobs}
    jobs_by_id = {j.id_job: j for j in instance.jobs}

    value = sum(max(0, (interval_solutions[job_id].get_end() - jobs_by_id[job_id].due_date)
                    * components_by_id_root_job[job_component_id_root_job[jobs_by_id[job_id]]].weight)
                for job_id in job_ids)

    return value


def solution_difference(a: CpoModelSolution,
                        b: CpoModelSolution,
                        selected_jobs: Iterable[Job] = None) -> Tuple[int, dict[int, int]]:
    a_interval_solutions = get_solution_job_interval_solutions(a)
    b_interval_solutions = get_solution_job_interval_solutions(b)
    assert a_interval_solutions.keys() == b_interval_solutions.keys()

    job_ids = ((j.id_job for j in selected_jobs) if selected_jobs is not None
               else a_interval_solutions.keys())
    differences = {job_id: a_interval_solutions[job_id].get_end() - b[job_id].get_end()
                   for job_id in job_ids}
    difference = sum(abs(diff) for diff in differences.values())
    return difference, differences

