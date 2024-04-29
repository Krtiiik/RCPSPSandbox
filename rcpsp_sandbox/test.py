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

    # with ExperimentManager(**DATA_DIRECTORY_STRUCTURE) as manager:
    #     instance = manager.load_base_instance("instance06")
    #     from solver.solver import Solver; plot_solution(Solver().solve(instance), block=False)
    #
    #     evaluation = TimeVariableConstraintRelaxingAlgorithm().evaluate(instance, TimeVariableConstraintRelaxingAlgorithmSettings(3,1,3,"time"))
    #     plot_solution(evaluation.solution, split_consumption=True)
    #
    # exit()

    if os.path.exists(KPIS_PICKLE_FILENAME):
        with open(KPIS_PICKLE_FILENAME, "rb") as f:
            instance_evaluations_kpis = pickle.load(f)
    else:
        instances = []
        # instances += ["instance01", "instance02", "instance03", "instance04", "instance05", "instance06", "instance07"]
        # instances += ["instance01", "instance01_1", "instance01_2", "instance01_3", "instance01_4"]
        # instances += ["instance02", "instance02_1", "instance02_2", "instance02_3", "instance02_4"]
        # instances += ["instance03", "instance03_1", "instance03_2", "instance03_3", "instance03_4"]
        # instances += ["instance04", "instance04_1", "instance04_2", "instance04_3", "instance04_4"]
        # instances += ["instance05", "instance05_1", "instance05_2", "instance05_3", "instance05_4"]
        # instances += ["instance06", "instance06_1", "instance06_2", "instance06_3", "instance06_4"]
        # instances += ["instance07", "instance07_1", "instance07_2", "instance07_3", "instance07_4"]
        # instances += ["instance08", "instance08_1", "instance08_2", "instance08_3", "instance08_4"]
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

                # manager.save_modified_instances(evaluation.modified_instance for evaluation in flatten(evaluations))
                # manager.save_evaluations(flatten(evaluations))
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
