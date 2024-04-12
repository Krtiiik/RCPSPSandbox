import os.path
import random

from bottlenecks.drawing import plot_evaluations, plot_solution
from bottlenecks.evaluations import evaluate_algorithms, compute_evaluation_kpis
from bottlenecks.improvements import TimeVariableConstraintRelaxingAlgorithm, MetricsRelaxingAlgorithm
from manager import ExperimentManager
from utils import flatten


DATA_DIRECTORY = os.path.join('..', 'data')
DATA_DIRECTORY_STRUCTURE = {
    'base_instances_location': os.path.join(DATA_DIRECTORY, 'base_instances'),
    'modified_instances_location': os.path.join(DATA_DIRECTORY, 'modified_instances'),
    'evaluations_location': os.path.join(DATA_DIRECTORY, 'evaluations'),
    'evaluations_kpis_location': os.path.join(DATA_DIRECTORY, 'evaluations_kpis'),
}
PLOT_DIRECTORY = os.path.join('..', 'plots')


def main():
    random.seed(42)

    instances_to_evaluate = [
        "instance01",
        "instance02",
        "instance03",
        "instance04",
    ]

    with ExperimentManager(**DATA_DIRECTORY_STRUCTURE) as manager:
        for instance_name in instances_to_evaluate:
            instance = manager.load_base_instance(instance_name)
            from solver.solver import Solver; plot_solution(Solver().solve(instance))

            evaluations = evaluate_algorithms(
                instance,
                [
                    (TimeVariableConstraintRelaxingAlgorithm(), {
                        "max_iterations": [1, 2, 3],
                        "relax_granularity": [1],
                        "max_improvement_intervals": [1, 2, 3]
                    }),
                    (MetricsRelaxingAlgorithm(), {
                        "metric": ["auac"],
                        "granularity": [4],
                        "convolution_mask": ["pre1"],
                        "max_iterations": [1, 2, 3],
                        "max_improvement_intervals": [1, 2, 3],
                        "capacity_addition": [2, 4, 6, 8, 10],
                    }),
                ],
                cache_manager=manager
            )
            evaluations_kpis = compute_evaluation_kpis(evaluations, 5, 1)

            manager.save_modified_instances(evaluation.modified_instance for evaluation in flatten(evaluations))
            manager.save_evaluations(flatten(evaluations))
            manager.save_evaluations_kpis(flatten(evaluations_kpis))

            plot_evaluations(evaluations_kpis)


if __name__ == "__main__":
    main()
