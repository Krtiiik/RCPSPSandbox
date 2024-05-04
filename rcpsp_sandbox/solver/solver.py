from docplex.cp.model import CpoModel

from instances.problem_instance import ProblemInstance
from solver.model_builder import build_model
from solver.solution import Solution, ModelSolution

import solver.cpo_config
solver.cpo_config.efficient(False)


class Solver:
    """
    A solver class for solving problem instances.
    """

    def solve(self,
              problem_instance: ProblemInstance | None = None,
              model: CpoModel | None = None) -> Solution:
        """
        Solves either a given problem instance or a provided model.

        Args:
            problem_instance (ProblemInstance or None): The problem instance to solve.
            model (CpoModel or None): The model to use for solving the problem instance.

        Returns:
            Solution: The solution to the problem instance.

        Raises:
            TypeError: If neither problem instance nor model is specified.
        """
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
        """
        Solves the problem instance using the provided model.

        Args:
            model (CpoModel): The model to use for solving the problem instance.
            instance (ProblemInstance): The problem instance to solve.

        Returns:
            Solution: The solution to the problem instance.
        """
        return ModelSolution(instance, model.solve(TimeLimit=10))
