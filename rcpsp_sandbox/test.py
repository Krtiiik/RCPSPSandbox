import os.path
import random

from bottlenecks.drawing import plot_solution, compute_shifting_interval_levels
from bottlenecks.evaluations import evaluate_algorithms, ProblemSetup, compute_evaluation_kpis
from bottlenecks.improvements import TimeVariableConstraintRelaxingAlgorithm
from instances.drawing import plot_components
from manager import ExperimentManager
from utils import flatten


DATA_DIRECTORY = os.path.join('..', 'Data')
DATA_DIRECTORY_STRUCTURE = {
    'base_instances_location': os.path.join(DATA_DIRECTORY, 'base_instances'),
    'modified_instances_location': os.path.join(DATA_DIRECTORY, 'modified_instances'),
    'evaluations_location': os.path.join(DATA_DIRECTORY, 'evaluations'),
    'evaluations_kpis_location': os.path.join(DATA_DIRECTORY, 'evaluations_kpis'),
}
PLOT_DIRECTORY = os.path.join('..', 'plots')


def main():
    random.seed(42)

    with ExperimentManager(**DATA_DIRECTORY_STRUCTURE) as manager:
        instance = manager.load_base_instance("instance30")
        evaluations = evaluate_algorithms(ProblemSetup(instance, 32), [
            (TimeVariableConstraintRelaxingAlgorithm(), {
                "max_iterations": [1, 2, 3],
                "relax_granularity": [1],
                "max_improvement_intervals": [1, 2, 3]
            }),
        ])
        evaluations_kpis = compute_evaluation_kpis(evaluations, 5, 1)

        manager.save_modified_instances(evaluation.modified_instance for evaluation in flatten(evaluations))
        manager.save_evaluations(flatten(evaluations))
        manager.save_evaluations_kpis(flatten(evaluations_kpis))


if __name__ == "__main__":
    main()
