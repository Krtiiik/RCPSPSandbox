import random

from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoModelSolution, CpoIntervalVarSolution

import instances.problem_instance
from instances.drawing import draw_instance_graph
from instances.problem_instance import Component, ProblemInstance, Job
from instances.problem_modifier import modify_instance
from solver.drawing import plot_solution, print_difference
from solver.model_builder import build_model
from solver.solution import Solution


class Solver:
    def solve(self,
              problem_instance: ProblemInstance or None = None,
              model: CpoModel or None = None) -> Solution:
        if model is not None:
            return self.__solve_model(model, problem_instance)
        elif problem_instance is not None:
            model = build_model(problem_instance) \
                .optimize_model(opt="Tardiness all") \
                .get_model()
            return self.__solve_model(model, problem_instance)
        else:
            raise TypeError("No problem instance nor model was specified to solve")

    @staticmethod
    def __solve_model(model: CpoModel, instance: ProblemInstance) -> Solution:
        return Solution(model.solve(), instance)


if __name__ == "__main__":
    import rcpsp_sandbox.instances.io as ioo

    random.seed(42)

    inst = ioo.parse_psplib("../../../Data/RCPSP/extended/instance_11.rp", is_extended=True)
    # inst = ioo.parse_psplib("../../../Data/RCPSP/j30/j301_4.sm")
    # inst = ioo.parse_psplib("../../../Data/RCPSP/j30/j303_1.sm")
    # draw_instance_graph(inst)
    inst.components = [rcpsp_sandbox.instances.problem_instance.Component(c.id_root_job + 1, c.weight) for c in inst.components]

    # inst = (modify_instance(inst)
    #         .split_job_components(split="paths")
    #         # .split_job_components(split="random roots")
    #         # .assign_job_due_dates(choice="uniform", interval=(0, 200))
    #         .assign_job_due_dates(choice="gradual", gradual_base=-5, gradual_interval=(0, 10))
    #         .generate_modified_instance())
    draw_instance_graph(inst)

    # ~~~~~ Modifications ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    optimize_for_orig: list[Job] = [inst.jobs_by_id[28]]
    optimize_for_orig[0].due_date = 0

    s = Solver()
    solve_result = s.solve(inst)
    if solve_result.is_solution():
        solution:CpoModelSolution = solve_result.get_solution()
        # solution.print_solution()

        alt_inst = inst.copy()

        # ~~~~~ Modifications ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # for r in alt_inst.resources:
        #     r.capacity += 10
        # optimize_for = random.choices(inst.jobs, k=3)
        optimize_for: list[Job] = [alt_inst.jobs_by_id[28]]
        optimize_for[0].due_date = 0

        alt_model = build_model(alt_inst) \
            .optimize_model(opt="Tardiness selected", priority_jobs=optimize_for) \
            .restrain_model_based_on_solution(solution, exclude=optimize_for, eps=20.) \
            .minimize_model_solution_difference(solution, exclude=optimize_for, alpha=1) \
            .get_model()
        alt_solve_result = s.solve(model=alt_model)
        if alt_solve_result.is_solution():
            alt_solution: CpoModelSolution = alt_solve_result.get_solution()
            # alt_solution.print_solution()

            max_end = max(max(i.end for i in solution.get_all_var_solutions() if isinstance(i, CpoIntervalVarSolution)),
                          max(i.end for i in alt_solution.get_all_var_solutions() if isinstance(i, CpoIntervalVarSolution)))
            print_difference(solution, inst, alt_solution, alt_inst)
            plot_solution(inst, solution, split_resource_consumption=True, fit_to_width=max_end)
            plot_solution(inst, solution, split_components=True, fit_to_width=max_end)
            plot_solution(alt_inst, alt_solution, split_resource_consumption=True, fit_to_width=max_end)
            plot_solution(alt_inst, alt_solution, split_components=True, fit_to_width=max_end)


