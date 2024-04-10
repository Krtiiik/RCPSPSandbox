import itertools
import os.path
import random

from bottlenecks.evaluations import evaluate_algorithms, ProblemSetup, compute_evaluation_kpis
from bottlenecks.improvements import TimeVariableConstraintRelaxingAlgorithm
from manager import ExperimentManager
from utils import flatten


def main():
    random.seed(42)

    with ExperimentManager(os.path.join('..', 'Data', 'base_instances'),
                           os.path.join('..', 'Data', 'modified_instances'),
                           os.path.join('..', 'Data', 'evaluations'),
                           os.path.join('..', 'Data', 'evaluations_kpis'),
                           ) as manager:
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
