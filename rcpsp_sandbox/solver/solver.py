import random
from typing import Iterable, Tuple

from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoSolveResult, CpoModelSolution

from rcpsp_sandbox.instances.drawing import draw_instance_graph
from rcpsp_sandbox.instances.problem_instance import ProblemInstance, Job
from rcpsp_sandbox.solver.drawing import plot_solution
from rcpsp_sandbox.instances.problem_modifier import modify_instance
from rcpsp_sandbox.solver.model_builder import build_model
from rcpsp_sandbox.solver.utils import get_solution_job_interval_solutions, compute_component_jobs


class SolveResult:
    result: CpoSolveResult

    def __init__(self, cpo_solve_result):
        self.result = cpo_solve_result

    def tardiness_value(self, instance: ProblemInstance, selected_jobs: Iterable[Job] = None):
        interval_solutions = get_solution_job_interval_solutions(self.result.get_solution())
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

    def difference_from(self, other_solution: CpoModelSolution, selected_jobs: Iterable[Job] = None) -> Tuple[int, dict[int, int]]:
        # TODO compute difference metric
        assert self.result.is_solution()
        self_solution = self.result.get_solution()
        self_interval_solutions, other_interval_solutions = get_solution_job_interval_solutions(
            self_solution), get_solution_job_interval_solutions(other_solution)
        assert self_interval_solutions.keys() == other_interval_solutions.keys()

        job_ids = ((j.id_job for j in selected_jobs) if selected_jobs is not None
                   else self_interval_solutions.keys())
        differences = {job_id: self_interval_solutions[job_id].get_end() - other_interval_solutions[job_id].get_end()
                       for job_id in job_ids}
        difference = sum(abs(diff) for diff in differences.values())
        return difference, differences


class Solver:
    def solve(self,
              problem_instance: ProblemInstance or None = None,
              model: CpoModel or None = None) -> SolveResult:
        if model is not None:
            return SolveResult(self.__solve_model(model))
        elif problem_instance is not None:
            model = build_model(problem_instance) \
                .optimize_model(opt="Tardiness all") \
                .get_model()
            return SolveResult(self.__solve_model(model))
        else:
            raise TypeError("No problem instance nor model was specified to solve")

    @staticmethod
    def __solve_model(model: CpoModel) -> CpoSolveResult:
        return model.solve()


if __name__ == "__main__":
    import rcpsp_sandbox.instances.io as ioo
    random.seed(42)

    # inst = ioo.parse_psplib("../../../Data/RCPSP/extended/instance_11.rp", is_extended=True)
    inst = ioo.parse_psplib("../../../Data/RCPSP/j30/j301_2.sm")
    draw_instance_graph(inst)

    inst = (modify_instance(inst)
            .split_job_components(split="paths")
            .assign_job_due_dates(choice="uniform", interval=(0, 200))
            # .assign_job_due_dates(choice="gradual", gradual_base=10, gradual_interval=(0, 10))
            .generate_modified_instance())
    draw_instance_graph(inst)

    s = Solver()
    solve_result = s.solve(inst)
    if solve_result.is_solution():
        solution = solve_result.get_solution()
        # solution.print_solution()
        plot_solution(inst, solution)

        alt_inst = inst.copy()

        # ~~~~~ Modifications ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        alt_inst.resources[2].capacity += 10

        alt_model = build_model(alt_inst) \
            .optimize_model(opt="Tardiness selected", priority_jobs=random.choices(inst.jobs, k=3)) \
            .restrain_model_based_on_solution(solution) \
            .minimize_model_solution_difference(solution) \
            .get_model()
        alt_solve_result = s.solve(model=alt_model)
        if alt_solve_result.is_solution():
            alt_solution = alt_solve_result.get_solution()
            # alt_solution.print_solution()
            plot_solution(alt_inst, alt_solution)
