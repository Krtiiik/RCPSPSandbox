import abc
import functools
import itertools
import time
from collections import namedtuple
from multiprocessing.pool import Pool
from typing import Iterable, Any

from bottlenecks.drawing import plot_solution
from instances.problem_instance import ProblemInstance
from solver.model_builder import build_model
from solver.solution import Solution
from solver.solver import Solver


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
    def base_instance(self) -> ProblemInstance:
        return self._base_instance

    @property
    def base_solution(self) -> Solution:
        return self._base_solution

    @property
    def target_job(self) -> int:
        return self._target_job

    @property
    def modified_instance(self) -> ProblemInstance:
        return self._modified_instance

    @property
    def solution(self) -> Solution:
        return self._solution

    @property
    def by(self) -> str:
        return self._by

    @property
    def duration(self) -> float:
        return self._duration

    def tardiness_improvement(self):
        original_tardiness = self._base_solution.tardiness(self._target_job)
        updated_tardiness = self._solution.tardiness(self._target_job)
        original_weighted_tardiness = self._base_solution.weighted_tardiness(self._target_job)
        updated_weighted_tardiness = self._solution.weighted_tardiness(self._target_job)
        return original_tardiness - updated_tardiness, original_weighted_tardiness - updated_weighted_tardiness

    def plot(self, block: bool = True, save_as: list[str] = None, dimensions: list[tuple[int, int]] = ((8, 11), (8, 11))):
        def get(iterable, i): return None if iterable is None else iterable[i]
        horizon = max(max(int_sol.end for int_sol in self._base_solution.job_interval_solutions.values()),
                      max(int_sol.end for int_sol in self._solution.job_interval_solutions.values()))
        plot_solution(self._base_solution, block=block, save_as=get(save_as, 0), dimensions=get(dimensions, 0), horizon=horizon)
        plot_solution(self._solution, block=block, save_as=get(save_as, 1), dimensions=get(dimensions, 1), horizon=horizon)

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
    _solver: Solver

    def __init__(self):
        self._solver = Solver()

    @property
    @abc.abstractmethod
    def settings_type(self) -> type:
        """Gets the type of the algorithm settings type"""

    @abc.abstractmethod
    def represent(self, settings) -> str:
        """Constructs a representation of the algorithm using its settings."""

    @abc.abstractmethod
    def _run(self,
             base_instance: ProblemInstance, base_solution: Solution, target_job_id: int,
             settings,
             ) -> tuple[ProblemInstance, Solution]:
        """Runs the algorithm."""

    def evaluate(self, problem: ProblemSetup, settings) -> Evaluation:
        """Evaluates the given instance."""
        time_start = time.thread_time()

        base_instance, target_job_id = problem
        model = self._build_standard_model(base_instance)
        base_solution = self._solver.solve(base_instance, model)

        modified_instance, solution = self._run(base_instance, base_solution, target_job_id, settings)

        duration = time.thread_time() - time_start
        return Evaluation(base_instance, base_solution, target_job_id, modified_instance, solution, self.represent(settings), duration)

    @staticmethod
    def _build_standard_model(instance: ProblemInstance):
        return build_model(instance) \
               .with_precedences().with_resource_constraints() \
               .optimize_model() \
               .get_model()


def evaluate_algorithms(problem: ProblemSetup,
                        algorithms_settings: Iterable[EvaluationAlgorithm | tuple[EvaluationAlgorithm, dict | Iterable[Any]]],
                        ) -> list[list[Evaluation]]:
    algorithms, alg_settings = __construct_settings(algorithms_settings)

    evaluations = []
    for algorithm, settings in zip(algorithms, alg_settings):
        algorithm_evaluations = []
        for setting in settings:
            result = algorithm.evaluate(problem, setting)
            algorithm_evaluations.append(result)
        evaluations.append(algorithm_evaluations)

    return evaluations


def __construct_settings(algorithms_settings: Iterable[EvaluationAlgorithm | tuple[EvaluationAlgorithm, dict | Iterable[Any]]]
                         ) -> tuple[list[EvaluationAlgorithm], list[Any]]:
    def error(): raise ValueError("Unexpected algorithm argument")

    algorithms = []
    settings = []
    for alg_maybe in algorithms_settings:
        if isinstance(alg_maybe, Evaluation):
            algorithms.append(alg_maybe)
            settings.append(None)
        elif isinstance(alg_maybe, tuple):
            assert len(alg_maybe) == 2
            algorithm, setting_iter_maybe = alg_maybe
            algorithms.append(algorithm)

            if isinstance(setting_iter_maybe, dict):
                setting_dict = setting_iter_maybe
                setting_params = list(setting_dict)
                settings.append([algorithm.settings_type(**{k: v for k, v in zip(setting_params, setting_values)})
                                 for setting_values in itertools.product(*setting_dict.values())])
            else:
                try: iter(setting_iter_maybe)
                except TypeError: error()

                settings.append([])
                for setting_maybe in setting_iter_maybe:
                    if not isinstance(setting_maybe, algorithm.settings_type):
                        error()
                    settings[-1].append(setting_maybe)
        else:
            error()

    return algorithms, settings
