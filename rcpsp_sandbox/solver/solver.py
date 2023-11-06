import random
from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoSolveResult

from instances.drawing import draw_instance_graph
from instances.problem_instance import ProblemInstance
from drawing import plot_solution
from instances.problem_modifier import modify_instance
from model_builder import build_model


class Solver:
    def solve(self,
              problem_instance: ProblemInstance or None = None,
              model: CpoModel or None = None) -> CpoSolveResult:
        if model is not None:
            return self.__solve_model(model)
        elif problem_instance is not None:
            model = build_model(problem_instance)
            return self.__solve_model(model)
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
        solution.print_solution()
        plot_solution(inst, solution)
