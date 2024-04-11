import abc
import itertools
import math
import time
from collections import namedtuple
from typing import Iterable, Any

from instances.problem_instance import ProblemInstance
from solver.model_builder import build_model
from solver.solution import Solution, ExplicitSolution
from solver.solver import Solver
from collections import namedtuple


ProblemSetup = namedtuple("ProblemSetup", ("instance", "target_job"))
IntervalSolutionLightweight = namedtuple("IntervalSolutionLightweight", ("start", "end"))
SolutionLightweight = dict[int, IntervalSolutionLightweight]


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

    def tardiness_improvement(self) -> tuple[int, int]:
        original_tardiness = self._base_solution.tardiness(self._target_job)
        updated_tardiness = self._solution.tardiness(self._target_job)
        original_weighted_tardiness = self._base_solution.weighted_tardiness(self._target_job)
        updated_weighted_tardiness = self._solution.weighted_tardiness(self._target_job)

        tardiness_change = original_tardiness - updated_tardiness
        weighted_tardiness_change = original_weighted_tardiness - updated_weighted_tardiness
        return tardiness_change, weighted_tardiness_change

    def total_capacity_changes(self) -> tuple[int, int]:
        def get_additions(inst): return ((r.key, *addition)
                                         for r in inst.resources
                                         for addition in r.availability.additions)

        def get_migrations(inst): return ((r.key, *migration)
                                          for r in inst.resources
                                          for migration in r.availability.migrations)

        base_additions = set(get_additions(self._base_instance))
        base_migrations = set(get_migrations(self._base_instance))
        modified_additions = set(get_additions(self._modified_instance))
        modified_migrations = set(get_migrations(self._modified_instance))

        addition_changes = base_additions ^ modified_additions
        migration_changes = base_migrations ^ modified_migrations

        addition_changes_sum = sum(addition[3] for addition in addition_changes)
        migration_changes_sum = sum(migration[4] for migration in migration_changes)

        return addition_changes_sum, migration_changes_sum

    def schedule_difference(self) -> tuple[int, dict[int, int]]:
        return self._base_solution.difference_to(self._solution)

    def print(self):
        tardiness_improvement, weighted_tardiness_improvement = self.tardiness_improvement()
        additions, migrations = self.total_capacity_changes()
        solution_difference, _ = self.schedule_difference()

        string = '\n\t'.join([
            f'Evaluation result by  : {self._by}',
            f'Computed in           : {self._duration:.2f} s',
            f'----------------------|'
            f'Base instance         : {self._base_instance.name}',
            f'Modified instance     : {self._modified_instance.name}',
            f'----------------------|'
            f'Tardiness improvement : {weighted_tardiness_improvement} ({tardiness_improvement} h)',
            f'Capacity changes      : {additions} additions, {migrations} migrations',
            f'Solution difference   : {solution_difference} h',
            f'----------------------|'
        ])
        print(string)

    @property
    def alg_string(self):
        return evaluation_alg_string(self.by)

    @property
    def settings_string(self):
        return evaluation_settings_string(self.by)


class EvaluationLightweight:
    _base_instance: str
    _base_solution: SolutionLightweight
    _modified_instance: str
    _target_job: int
    _solution: SolutionLightweight
    _by: str
    _duration: float

    def __init__(self,
                 base_instance: str, base_solution: SolutionLightweight, target_job: int,
                 modified_instance: str, solution: SolutionLightweight,
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
    def base_instance(self) -> str:
        return self._base_instance

    @property
    def base_solution(self) -> SolutionLightweight:
        return self._base_solution

    @property
    def target_job(self) -> int:
        return self._target_job

    @property
    def modified_instance(self) -> str:
        return self._modified_instance

    @property
    def solution(self) -> SolutionLightweight:
        return self._solution

    @property
    def by(self) -> str:
        return self._by

    @property
    def duration(self) -> float:
        return self._duration

    def build_full_evaluation(self,
                              base_instance: ProblemInstance,
                              modified_instance: ProblemInstance
                              ) -> Evaluation:
        base_solution = ExplicitSolution(base_instance, self._base_solution)
        modified_solution = ExplicitSolution(modified_instance, self._solution)

        return Evaluation(base_instance=base_instance, base_solution=base_solution, target_job=self._target_job,
                          modified_instance=modified_instance, solution=modified_solution,
                          by=self._by, duration=self._duration)


class EvaluationKPIs:
    _evaluation: Evaluation
    _cost: int
    _improvement: int
    _schedule_difference: int

    def __init__(self, evaluation: Evaluation, cost: int, improvement: int, schedule_difference: int):
        self._evaluation = evaluation
        self._cost = cost
        self._improvement = improvement
        self._schedule_difference = schedule_difference

    @property
    def evaluation(self) -> Evaluation:
        return self._evaluation

    @property
    def cost(self) -> int:
        return self._cost

    @property
    def improvement(self) -> int:
        return self._improvement

    @property
    def schedule_difference(self) -> int:
        return self._schedule_difference


class EvaluationKPIsLightweight:
    _evaluation: EvaluationLightweight
    _cost: int
    _improvement: int
    _schedule_difference: int

    def __init__(self, evaluation: EvaluationLightweight, cost: int, improvement: int, schedule_difference: int):
        self._evaluation = evaluation
        self._cost = cost
        self._improvement = improvement
        self._schedule_difference = schedule_difference

    @property
    def evaluation(self):
        return self._evaluation

    @property
    def cost(self):
        return self._cost

    @property
    def improvement(self):
        return self._improvement

    @property
    def schedule_difference(self) -> int:
        return self._schedule_difference

    def build_full_evaluation_kpis(self,
                                   base_instance: ProblemInstance,
                                   modified_instance: ProblemInstance
                                   ) -> EvaluationKPIs:
        return EvaluationKPIs(self._evaluation.build_full_evaluation(base_instance, modified_instance),
                              self._cost, self._improvement, self._schedule_difference)


class EvaluationAlgorithm(metaclass=abc.ABCMeta):
    ID_SEPARATOR = '--'
    ID_SETTINGS_SEPARATOR = '-'
    ID_SUB_SEPARATOR = '#'

    _solver: Solver

    def __init__(self):
        self._solver = Solver()

    @property
    @abc.abstractmethod
    def settings_type(self) -> type:
        """Gets the type of the algorithm settings type"""

    @abc.abstractmethod
    def _run(self,
             base_instance: ProblemInstance, base_solution: Solution, target_job_id: int,
             settings,
             ) -> tuple[ProblemInstance, Solution]:
        """Runs the algorithm."""

    @property
    def shortname(self) -> str:
        """Gets a short name of the algorithm."""
        return ''.join(filter(str.isupper, type(self).__name__))

    def represent(self, settings) -> str:
        """Constructs a representation of the algorithm using its settings."""
        settings_str = self.ID_SETTINGS_SEPARATOR.join(map(str, settings))
        alg_str = type(self).__name__
        return f'{alg_str}{self.ID_SEPARATOR}{settings_str}'

    def represent_short(self, settings) -> str:
        """Constructs a representation of the algorithm using its settings."""
        settings_str = self.ID_SETTINGS_SEPARATOR.join(map(str, settings))
        alg_str = self.shortname
        return f'{alg_str}{self.ID_SEPARATOR}{settings_str}'

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

    def timeit():
        delta = time.time() - start_time
        return f'{delta//60:0>2.0f}\'{delta % 60:0>2.0f}"'

    print_n_algs, print_n_algs_digits = len(algorithms), int(math.log10(len(algorithms))) + 1
    start_time = time.time()
    print(f"Evaluating {print_n_algs} algorithms...")

    evaluations = []
    for i_alg, (algorithm, settings) in enumerate(zip(algorithms, alg_settings)):
        print_n_settings, print_n_settings_digits = len(settings), int(math.log10(len(settings))) + 1
        print_prefix = f"\t{1+i_alg:>{print_n_algs_digits}d}/{print_n_algs}"
        print(f"\r{print_prefix}: ({0:>{print_n_settings_digits}d}/{print_n_settings}) >> {timeit()}", end='')

        algorithm_evaluations = []
        for i_setting, setting in enumerate(settings):
            result = algorithm.evaluate(problem, setting)
            algorithm_evaluations.append(result)

            print(f"\r{print_prefix}: ({1+i_setting:>{print_n_settings_digits}d}/{print_n_settings}) >> {timeit()}", end='')

        evaluations.append(algorithm_evaluations)

    print(f"\rEvaluation completed in {timeit()}")

    return evaluations


def compute_evaluation_kpis(evaluations: list[Evaluation | list[Evaluation]],
                            addition_price: int, migration_price: int,
                            ) -> list[EvaluationKPIs | list[EvaluationKPIs]]:
    def cost(_evaluation):
        _addition, _migration = _evaluation.total_capacity_changes()
        return addition_price * _addition + migration_price * _migration

    def improvement(_evaluation):
        return _evaluation.tardiness_improvement()[1]

    def schedule_difference(_evaluation):
        return _evaluation.schedule_difference()[0]

    if not isinstance(evaluations[0], list):
        return [EvaluationKPIs(evaluation, cost(evaluation), improvement(evaluation), schedule_difference(evaluation))
                for evaluation in evaluations]
    else:
        return [[EvaluationKPIs(evaluation, cost(evaluation), improvement(evaluation), schedule_difference(evaluation))
                 for evaluation in evaluation_iter]
                for evaluation_iter in evaluations]


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
                try:
                    iter(setting_iter_maybe)
                except TypeError:
                    error()

                settings.append([])
                for setting_maybe in setting_iter_maybe:
                    if not isinstance(setting_maybe, algorithm.settings_type):
                        error()
                    settings[-1].append(setting_maybe)
        else:
            error()

    return algorithms, settings


def evaluation_alg_string(by_str: str):
    return by_str.split(EvaluationAlgorithm.ID_SEPARATOR)[0]


def evaluation_settings_string(by_str: str):
    return by_str.split(EvaluationAlgorithm.ID_SEPARATOR)[1]
