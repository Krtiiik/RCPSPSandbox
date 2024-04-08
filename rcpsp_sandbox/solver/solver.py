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

import solver.cpo_config
solver.cpo_config.efficient(False)


class Solver:
    def solve(self,
              problem_instance: ProblemInstance or None = None,
              model: CpoModel or None = None) -> Solution:
        if model is not None:
            return self.__solve_model(model, problem_instance)
        elif problem_instance is not None:
            model = build_model(problem_instance) \
                    .with_precedences().with_resource_constraints() \
                    .optimize_model() \
                    .get_model()
            return self.__solve_model(model, problem_instance)
        else:
            raise TypeError("No problem instance nor model was specified to solve")

    @staticmethod
    def __solve_model(model: CpoModel, instance: ProblemInstance) -> Solution:
        return Solution(model.solve(), instance)
