import math
import os.path
import pickle
import random
from collections import defaultdict

from bottlenecks.drawing import plot_evaluations, plot_solution
from bottlenecks.evaluations import evaluate_algorithms, compute_evaluation_kpis, EvaluationKPIs, Evaluation, \
    EvaluationKPIsLightweight, EvaluationLightweight
from bottlenecks.improvements import TimeVariableConstraintRelaxingAlgorithm, MetricsRelaxingAlgorithm
from generate_instances import experiment_instances
from manager import ExperimentManager
from utils import flatten, group_evaluations_kpis_by_instance_type

DATA_DIRECTORY = os.path.join('..', 'data_mod')
DATA_DIRECTORY_STRUCTURE = {
    'base_instances_location': os.path.join(DATA_DIRECTORY, 'base_instances'),
    'modified_instances_location': os.path.join(DATA_DIRECTORY, 'modified_instances'),
    'evaluations_location': os.path.join(DATA_DIRECTORY, 'evaluations'),
    'evaluations_kpis_location': os.path.join(DATA_DIRECTORY, 'evaluations_kpis'),
}
PLOT_DIRECTORY = os.path.join('..', 'plots')
KPIS_PICKLE_FILENAME = os.path.join(DATA_DIRECTORY, 'kpis.pickle')


def main():
    random.seed(42)

    if os.path.exists(KPIS_PICKLE_FILENAME):
        with open(KPIS_PICKLE_FILENAME, "rb") as f:
            instance_evaluations_kpis = pickle.load(f)
    else:
        instances = list(experiment_instances)

        instance_evaluations_kpis = dict()
        with ExperimentManager(**DATA_DIRECTORY_STRUCTURE) as manager:
            for instance_name in instances:
                instance = manager.load_base_instance(instance_name)
                # from solver.solver import Solver; plot_solution(Solver().solve(instance), block=False, save_as=os.path.join('..', 'insts', instance_name+'.png'))

                evaluations = evaluate_algorithms(
                    instance,
                    [
                        (TimeVariableConstraintRelaxingAlgorithm(), {
                            "max_iterations": [1, 2, 3],
                            "relax_granularity": [1],
                            "max_improvement_intervals": [1, 2, 3, 4, 5, 6],
                            "interval_sort": ["improvement"]
                        }),
                        (TimeVariableConstraintRelaxingAlgorithm(), {
                            "max_iterations": [1, 2, 3],
                            "relax_granularity": [1],
                            "max_improvement_intervals": [1, 2, 3, 4, 5, 6],
                            "interval_sort": ["time"]
                        }),
                        (MetricsRelaxingAlgorithm(), {
                            "metric": ["auac"],
                            "granularity": [4, 8],
                            "convolution_mask": ["pre1", "around",
                                                 "post"
                                                 ],
                            "max_iterations": [1, 2, 3],
                            "max_improvement_intervals": [1, 2, 3, 4],
                            "capacity_addition": [4, 10],
                        }),
                        (MetricsRelaxingAlgorithm(), {
                            "metric": ["mrur"],
                            "granularity": [4, 8],
                            "convolution_mask": ["pre1", "around",
                                                 "post"
                                                 ],
                            "max_iterations": [1, 2, 3],
                            "max_improvement_intervals": [1, 2, 3, 4],
                            "capacity_addition": [4, 10],
                        }),
                    ],
                    cache_manager=manager,
                )
                evaluations_kpis = compute_evaluation_kpis(evaluations, 5, 1)

                manager.save_evaluations_kpis(flatten(evaluations_kpis))

                instance_evaluations_kpis[instance_name] = evaluations_kpis

        with open(KPIS_PICKLE_FILENAME, "wb") as f:
            pickle.dump(instance_evaluations_kpis, f)

    GROUPED = True
    SCALE = False
    PARETO = True
    if GROUPED:
        instance_evaluations_kpis = group_evaluations_kpis_by_instance_type(instance_evaluations_kpis, scale=SCALE)

    plot_evaluations(instance_evaluations_kpis, value_axes=("cost", "improvement"), pareto_front=PARETO)
    plot_evaluations(instance_evaluations_kpis, value_axes=("improvement", "schedule difference"), pareto_front=PARETO)
    plot_evaluations(instance_evaluations_kpis, value_axes=("duration", "improvement"), pareto_front=PARETO)


if __name__ == "__main__":
    main()
