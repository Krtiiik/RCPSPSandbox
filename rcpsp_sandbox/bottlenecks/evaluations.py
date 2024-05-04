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


IntervalSolutionLightweight = namedtuple("IntervalSolutionLightweight", ("start", "end"))
SolutionLightweight = dict[int, IntervalSolutionLightweight]


class Evaluation:
    """
    Represents an evaluation of a problem instance and its solution.
    """

    def __init__(self,
                 base_instance: ProblemInstance, base_solution: Solution,
                 modified_instance: ProblemInstance, solution: Solution,
                 by: str, duration: float,
                 ):
        """
        Initializes a new instance of the Evaluation class.

        Args:
            base_instance (ProblemInstance): The base problem instance.
            base_solution (Solution): The base solution.
            modified_instance (ProblemInstance): The modified problem instance.
            solution (Solution): The modified solution.
            by (str): The evaluator of the evaluation.
            duration (float): The duration of the evaluation in seconds.
        """
        self._base_instance = base_instance
        self._base_solution = base_solution
        self._modified_instance = modified_instance
        self._solution = solution
        self._by = by
        self._duration = duration

    @property
    def base_instance(self) -> ProblemInstance:
        """
        Gets the base problem instance.

        Returns:
            ProblemInstance: The base problem instance.
        """
        return self._base_instance

    @property
    def base_solution(self) -> Solution:
        """
        Gets the base solution.

        Returns:
            Solution: The base solution.
        """
        return self._base_solution

    @property
    def modified_instance(self) -> ProblemInstance:
        """
        Gets the modified problem instance.

        Returns:
            ProblemInstance: The modified problem instance.
        """
        return self._modified_instance

    @property
    def solution(self) -> Solution:
        """
        Gets the modified solution.

        Returns:
            Solution: The modified solution.
        """
        return self._solution

    @property
    def by(self) -> str:
        """
        Gets the evaluator of the evaluation.

        Returns:
            str: The evaluator of the evaluation.
        """
        return self._by

    @property
    def duration(self) -> float:
        """
        Gets the duration of the evaluation in seconds.

        Returns:
            float: The duration of the evaluation in seconds.
        """
        return self._duration

    def tardiness_improvement(self) -> tuple[int, int]:
        """
        Calculates the improvement in tardiness between the base solution and the modified solution.

        Returns:
            tuple[int, int]: A tuple containing the improvement in tardiness and weighted tardiness.
        """
        target_job = self._base_instance.target_job
        original_tardiness = self._base_solution.tardiness(target_job)
        updated_tardiness = self._solution.tardiness(target_job)
        original_weighted_tardiness = self._base_solution.weighted_tardiness(target_job)
        updated_weighted_tardiness = self._solution.weighted_tardiness(target_job)

        tardiness_change = original_tardiness - updated_tardiness
        weighted_tardiness_change = original_weighted_tardiness - updated_weighted_tardiness
        return tardiness_change, weighted_tardiness_change

    def total_capacity_changes(self) -> tuple[int, int]:
        """
        Calculates the total capacity changes between the base instance and the modified instance.

        Returns:
            tuple[int, int]: A tuple containing the sum of additions and migrations.
        """
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
        """
        Calculates the difference in schedule between the base solution and the modified solution.

        Returns:
            tuple[int, dict[int, int]]: A tuple containing the difference in schedule and a dictionary of job differences.
        """
        return self._base_solution.difference_to(self._solution)

    def print(self):
        """
        Prints the evaluation result.
        """
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
        """
        Gets the algorithm string.

        Returns:
            str: The algorithm string.
        """
        return evaluation_alg_string(self.by)

    @property
    def settings_string(self):
        """
        Gets the settings string.

        Returns:
            str: The settings string.
        """
        return evaluation_settings_string(self.by)


class EvaluationLightweight:
    """
    Represents a lightweight evaluation of a solution for a modified problem instance
    compared to a base problem instance.
    """

    _base_instance: str
    _base_solution: SolutionLightweight
    _modified_instance: str
    _solution: SolutionLightweight
    _by: str
    _duration: float

    def __init__(self,
                 base_instance: str, base_solution: SolutionLightweight,
                 modified_instance: str, solution: SolutionLightweight,
                 by: str, duration: float,
                 ):
        """
        Initializes a new instance of the EvaluationLightweight class.

        Args:
            base_instance (str): The base problem instance.
            base_solution (SolutionLightweight): The solution for the base problem instance.
            modified_instance (str): The modified problem instance.
            solution (SolutionLightweight): The solution for the modified problem instance.
            by (str): The method used to modify the problem instance.
            duration (float): The duration of the evaluation in seconds.
        """
        self._base_instance = base_instance
        self._base_solution = base_solution
        self._modified_instance = modified_instance
        self._solution = solution
        self._by = by
        self._duration = duration

    @property
    def base_instance(self) -> str:
        """
        Gets the base problem instance.

        Returns:
            str: The base problem instance.
        """
        return self._base_instance

    @property
    def base_solution(self) -> SolutionLightweight:
        """
        Gets the solution for the base problem instance.

        Returns:
            SolutionLightweight: The solution for the base problem instance.
        """
        return self._base_solution

    @property
    def modified_instance(self) -> str:
        """
        Gets the modified problem instance.

        Returns:
            str: The modified problem instance.
        """
        return self._modified_instance

    @property
    def solution(self) -> SolutionLightweight:
        """
        Gets the solution for the modified problem instance.

        Returns:
            SolutionLightweight: The solution for the modified problem instance.
        """
        return self._solution

    @property
    def by(self) -> str:
        """
        Gets the method used to modify the problem instance.

        Returns:
            str: The method used to modify the problem instance.
        """
        return self._by

    @property
    def duration(self) -> float:
        """
        Gets the duration of the evaluation in seconds.

        Returns:
            float: The duration of the evaluation in seconds.
        """
        return self._duration

    def build_full_evaluation(self,
                              base_instance: ProblemInstance,
                              modified_instance: ProblemInstance
                              ) -> Evaluation:
        """
        Builds a full evaluation object based on the lightweight evaluation.

        Args:
            base_instance (ProblemInstance): The base problem instance.
            modified_instance (ProblemInstance): The modified problem instance.

        Returns:
            Evaluation: The full evaluation object.
        """
        base_solution = ExplicitSolution(base_instance, self._base_solution)
        modified_solution = ExplicitSolution(modified_instance, self._solution)

        return Evaluation(base_instance=base_instance, base_solution=base_solution,
                          modified_instance=modified_instance, solution=modified_solution,
                          by=self._by, duration=self._duration)


class EvaluationKPIs:
    """
    Represents evaluation KPIs.
    """

    _evaluation: Evaluation
    _cost: int
    _improvement: int
    _schedule_difference: int

    def __init__(self, evaluation: Evaluation, cost: int, improvement: int, schedule_difference: int):
        """
        Initializes a new instance of the EvaluationKPIs class.

        Args:
            evaluation (Evaluation): The evaluation object.
            cost (int): The cost of the evaluation.
            improvement (int): The improvement of the evaluation.
            schedule_difference (int): The schedule difference of the evaluation.
        """
        self._evaluation = evaluation
        self._cost = cost
        self._improvement = improvement
        self._schedule_difference = schedule_difference

    @property
    def evaluation(self) -> Evaluation:
        """
        Gets the evaluation object.

        Returns:
            Evaluation: The evaluation object.
        """
        return self._evaluation

    @property
    def cost(self) -> int:
        """
        Gets the cost of the evaluation.

        Returns:
            int: The cost of the evaluation.
        """
        return self._cost

    @property
    def improvement(self) -> int:
        """
        Gets the improvement of the evaluation.

        Returns:
            int: The improvement of the evaluation.
        """
        return self._improvement

    @property
    def schedule_difference(self) -> int:
        """
        Gets the schedule difference of the evaluation.

        Returns:
            int: The schedule difference of the evaluation.
        """
        return self._schedule_difference


class EvaluationKPIsLightweight:
    """
    Represents the lightweight version of evaluation KPIs.
    """

    _evaluation: EvaluationLightweight
    _cost: int
    _improvement: int
    _schedule_difference: int

    def __init__(self, evaluation: EvaluationLightweight, cost: int, improvement: int, schedule_difference: int):
        """
        Initializes a new instance of the EvaluationKPIsLightweight class.

        Args:
            evaluation (EvaluationLightweight): The lightweight evaluation object.
            cost (int): The cost of the evaluation.
            improvement (int): The improvement of the evaluation.
            schedule_difference (int): The schedule difference of the evaluation.
        """
        self._evaluation = evaluation
        self._cost = cost
        self._improvement = improvement
        self._schedule_difference = schedule_difference

    @property
    def evaluation(self):
        """
        Gets the lightweight evaluation object.

        Returns:
            EvaluationLightweight: The lightweight evaluation object.
        """
        return self._evaluation

    @property
    def cost(self):
        """
        Gets the cost of the evaluation.

        Returns:
            int: The cost of the evaluation.
        """
        return self._cost

    @property
    def improvement(self):
        """
        Gets the improvement of the evaluation.

        Returns:
            int: The improvement of the evaluation.
        """
        return self._improvement

    @property
    def schedule_difference(self) -> int:
        """
        Gets the schedule difference of the evaluation.

        Returns:
            int: The schedule difference of the evaluation.
        """
        return self._schedule_difference

    def build_full_evaluation_kpis(self,
                                   base_instance: ProblemInstance,
                                   modified_instance: ProblemInstance
                                   ) -> EvaluationKPIs:
        """
        Builds the full evaluation key performance indicators (KPIs) object.

        Args:
            base_instance (ProblemInstance): The base problem instance.
            modified_instance (ProblemInstance): The modified problem instance.

        Returns:
            EvaluationKPIs: The full evaluation KPIs object.
        """
        return EvaluationKPIs(self._evaluation.build_full_evaluation(base_instance, modified_instance),
                              self._cost, self._improvement, self._schedule_difference)


class EvaluationAlgorithm(metaclass=abc.ABCMeta):
    """
    Represents an algorithm for finding improvement in problem instances and their solutions.
    """
    ID_SEPARATOR = '--'
    ID_SETTINGS_SEPARATOR = '-'
    ID_SUB_SEPARATOR = '#'

    _solver: Solver

    def __init__(self):
        """
        Initializes a new instance of the EvaluationAlgorithm class.
        """
        self._solver = Solver()

    @property
    @abc.abstractmethod
    def settings_type(self) -> type:
        """Gets the type of the algorithm settings type"""

    @abc.abstractmethod
    def _run(self,
             base_instance: ProblemInstance, base_solution: Solution,
             settings,
             ) -> tuple[ProblemInstance, Solution]:
        """Runs the algorithm."""

    @property
    def name(self) -> str:
        """Gets the name of the algorithm."""
        return type(self).__name__

    @property
    def shortname(self) -> str:
        """Gets a short name of the algorithm."""
        return ''.join(filter(str.isupper, type(self).__name__))

    def represent(self, settings) -> str:
        """Constructs a representation of the algorithm using its settings."""
        settings_str = self.ID_SETTINGS_SEPARATOR.join(map(str, settings))
        alg_str = self.name
        return f'{alg_str}{self.ID_SEPARATOR}{settings_str}'

    def represent_short(self, settings) -> str:
        """Constructs a representation of the algorithm using its settings."""
        settings_str = self.ID_SETTINGS_SEPARATOR.join(map(str, settings))
        alg_str = self.shortname
        return f'{alg_str}{self.ID_SEPARATOR}{settings_str}'

    def evaluate(self, base_instance: ProblemInstance, settings) -> Evaluation:
        """Evaluates the given instance."""
        time_start = time.perf_counter()

        model = self._build_standard_model(base_instance)
        base_solution = self._solver.solve(base_instance, model)

        modified_instance, solution = self._run(base_instance, base_solution, settings)

        duration = time.perf_counter() - time_start
        return Evaluation(base_instance, base_solution, modified_instance, solution, self.represent(settings), duration)

    @staticmethod
    def _build_standard_model(instance: ProblemInstance):
        return build_model(instance) \
               .with_precedences().with_resource_constraints() \
               .optimize_model() \
               .get_model()


def evaluate_algorithms(instance: ProblemInstance,
                        algorithms_settings: Iterable[EvaluationAlgorithm | tuple[EvaluationAlgorithm, dict | Iterable[Any]]],
                        cache_manager=None,
                        save_results: bool = True,
                        ) -> list[list[Evaluation]]:
    """
    Evaluates a list of algorithms on a given problem instance.
    The evaluations are cached if a cache manager is provided.
    The list of algorithms can be a list of EvaluationAlgorithm objects or a list of tuples
    containing an EvaluationAlgorithm object and a list of settings or a dictionary of setting values from which
    combinations are generated.

    Args:
        instance (ProblemInstance): The problem instance.
        algorithms_settings (Iterable[EvaluationAlgorithm | tuple[EvaluationAlgorithm, dict | Iterable[Any]]]):
            The list of algorithms and their settings.
        cache_manager (CacheManager): The cache manager.
        save_results (bool): Whether to save the results.

    Returns:
        list[list[Evaluation]]: A list of evaluations for each algorithm.
    """
    def timeit():
        delta = time.time() - start_time
        return f'{delta//60:0>2.0f}\'{delta % 60:0>2.0f}"'

    # Construct the settings of the algorithms
    algorithms, alg_settings = __construct_settings(algorithms_settings)

    print_n_algs, print_n_algs_digits = len(algorithms), int(math.log10(len(algorithms))) + 1
    start_time = time.time()
    print(f"Evaluating {print_n_algs} algorithms on instance \"{instance.name}\"...")

    evaluations = []
    for i_alg, (algorithm, settings) in enumerate(zip(algorithms, alg_settings)):  # Evaluate each algorithm
        computed = 0
        cached = 0
        print_n_settings, print_n_settings_digits = len(settings), int(math.log10(len(settings))) + 1
        print_prefix = f"\t{1+i_alg:>{print_n_algs_digits}d}/{print_n_algs}"
        print(f"\r{print_prefix}: (cached {0:>{print_n_settings_digits}d} + computed {0:>{print_n_settings_digits}d} = {0:>{print_n_settings_digits}d}/{print_n_settings}) >> {timeit()}", end='')

        if cache_manager:  # Load evaluation caches
            cache_manager.load_evaluations_caches(instance.name, algorithm.name)

        algorithm_evaluations = []
        for i_setting, setting in enumerate(settings):  # Evaluate each setting
            evaluation_id = algorithm.represent(setting)
            if cache_manager and cache_manager.is_evaluation_cached(instance.name, evaluation_id):  # Load cached evaluation
                result = cache_manager.load_evaluation(instance.name, evaluation_id)
                cached += 1
            else:  # Compute evaluation
                result = algorithm.evaluate(instance, setting)
                computed += 1

            algorithm_evaluations.append(result)

            print(f"\r{print_prefix}: (cached {cached:>{print_n_settings_digits}d} + computed {computed:>{print_n_settings_digits}d} = {1+i_setting:>{print_n_settings_digits}d}/{print_n_settings}) >> {timeit()}", end='')

        evaluations.append(algorithm_evaluations)

        if cache_manager and save_results:  # Save evaluations and modified instances
            cache_manager.save_modified_instances(evaluation.modified_instance for evaluation in algorithm_evaluations)
            cache_manager.save_evaluations(algorithm_evaluations)
            cache_manager.flush()
            cache_manager.clear_caches()

    print(f"\rEvaluation completed in {timeit()}")

    return evaluations


def compute_evaluation_kpis(evaluations: list[Evaluation | list[Evaluation]],
                            addition_price: int, migration_price: int,
                            ) -> list[EvaluationKPIs | list[EvaluationKPIs]]:
    """
    Computes the Key Performance Indicators (KPIs) for a list of evaluations.

    Args:
        evaluations (list[Evaluation | list[Evaluation]]): A list of evaluations or a list of lists of evaluations.
        addition_price (int): The price for each unit of addition capacity.
        migration_price (int): The price for each unit of migration capacity.

    Returns:
        list[EvaluationKPIs | list[EvaluationKPIs]]: A list of EvaluationKPIs objects or a list of lists of EvaluationKPIs objects,
        containing the computed KPIs for each evaluation.
    """
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
