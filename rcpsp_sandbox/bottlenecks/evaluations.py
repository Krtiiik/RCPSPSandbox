import abc
from collections import namedtuple

from bottlenecks.drawing import plot_solution
from instances.problem_instance import ProblemInstance
from solver.solution import Solution


ProblemSetup = namedtuple("ProblemSetup", ("instance", "target_job"))


class Evaluation:
    _base_instance: ProblemInstance
    _base_solution: Solution
    _modified_instance: ProblemInstance
    _target_job: int
    _solution: Solution
    _by: str
    _duration: float

    def __init__(self,
                 base_instance: ProblemInstance, base_solution: Solution, target_job: int,
                 modified_instance: ProblemInstance, solution: Solution,
                 by: str, duration: float,
                 ):
        self._base_instance = base_instance
        self._base_solution = base_solution
        self._target_job = target_job
        self._modified_instance = modified_instance
        self._solution = solution
        self._by = by
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def tardiness_improvement(self):
        original_tardiness = self._base_solution.tardiness(self._target_job)
        updated_tardiness = self._solution.tardiness(self._target_job)
        original_weighted_tardiness = self._base_solution.weighted_tardiness(self._target_job)
        updated_weighted_tardiness = self._solution.weighted_tardiness(self._target_job)
        return original_tardiness - updated_tardiness, original_weighted_tardiness - updated_weighted_tardiness

    def plot(self, block: bool = True, save_as: list[str] = None, dimensions: list[tuple[int, int]] = [(8, 11), (8, 11)]):
        def get(iterable, i): return None if iterable is None else iterable[i]
        plot_solution(self._base_solution, block=block, save_as=get(save_as, 0), dimensions=get(dimensions, 0), component_legends=legends)
        plot_solution(self._solution, block=block, save_as=get(save_as, 1), dimensions=get(dimensions, 1), component_legends=legends)

    def print(self):
        print(str(self))

    def __str__(self):
        tardiness_improvement, weighted_tardiness_improvement = self.tardiness_improvement()
        return '\n\t'.join([
            f'Evaluation result',
            f'Base instance         : {self._base_instance.name}',
            f'Modified instance     : {self._modified_instance.name}',
            f'Tardiness improvement : {weighted_tardiness_improvement} ({tardiness_improvement} h)',
            f'Evaluated by          : {self._by}',
            f'Computed in           : {self._duration:.2f} s',
        ])


class EvaluationAlgorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, problem: ProblemSetup, settings) -> Evaluation:
        """Evaluates the given instance."""

    @abc.abstractmethod
    def represent(self, settings) -> str:
        """Constructs a representation of the algorithm using its settings."""
