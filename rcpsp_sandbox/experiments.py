import argparse
import os.path
import pickle

import numpy as np
import tabulate

from bottlenecks.drawing import plot_evaluations
from bottlenecks.evaluations import evaluate_algorithms, compute_evaluation_kpis, EvaluationKPIs
from bottlenecks.improvements import TimeVariableConstraintRelaxingAlgorithm, MetricsRelaxingAlgorithm
from generate_instances import experiment_instances, experiment_instances_info
from manager import ExperimentManager
from utils import group_evaluations_kpis_by_instance_type, pareto_front_kpis

DATA_DIRECTORY = os.path.join('..', 'data')
DATA_DIRECTORY_STRUCTURE = {
    'base_instances_location': os.path.join(DATA_DIRECTORY, 'base_instances'),
    'modified_instances_location': os.path.join(DATA_DIRECTORY, 'modified_instances'),
    'evaluations_location': os.path.join(DATA_DIRECTORY, 'evaluations'),
    'evaluations_kpis_location': os.path.join(DATA_DIRECTORY, 'evaluations_kpis'),
}
RESULTS_DIRECTORY = os.path.join('..', 'results')
EVALUATIONS_PICKLE_FILENAME = os.path.join(DATA_DIRECTORY, 'evaluations.pickle')
EVALUATIONS_KPIS_PICKLE_FILENAME = os.path.join(DATA_DIRECTORY, 'evaluations_kpis.pickle')
ALGORITHMS_SETTINGS = [
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
        "convolution_mask": ["pre1", "around", "post"],
        "max_iterations": [1, 2, 3],
        "max_improvement_intervals": [1, 2, 3, 4],
        "capacity_addition": [4, 10],
    }),
    (MetricsRelaxingAlgorithm(), {
        "metric": ["mrur"],
        "granularity": [4, 8],
        "convolution_mask": ["pre1", "around", "post"],
        "max_iterations": [1, 2, 3],
        "max_improvement_intervals": [1, 2, 3, 4],
        "capacity_addition": [4, 10],
    }),
]

I_improvement = 0
I_cost = 1
I_schedule_difference = 2
I_duration = 3


def get_evaluations_kpis(args: argparse.Namespace):
    if os.path.exists(EVALUATIONS_KPIS_PICKLE_FILENAME):  # if cached evaluations kpis exist...
        with open(EVALUATIONS_KPIS_PICKLE_FILENAME, "rb") as f:
            evaluations_kpis = pickle.load(f)
    else:
        # Compute evaluations kpis
        if os.path.exists(EVALUATIONS_PICKLE_FILENAME):  # if cached evaluations exist...
            with open(EVALUATIONS_PICKLE_FILENAME, "rb") as f:
                instances_evaluations = pickle.load(f)
        else:  # compute the evaluations
            instances = list(experiment_instances)

            instances_evaluations = dict()
            with ExperimentManager(**DATA_DIRECTORY_STRUCTURE) as manager:
                for instance_name in instances:  # evaluate each instance, either from cache or computed
                    instance = manager.load_base_instance(instance_name)
                    instance_evaluations = evaluate_algorithms(instance, ALGORITHMS_SETTINGS, cache_manager=manager)
                    instances_evaluations[instance_name] = instance_evaluations

            with open(EVALUATIONS_PICKLE_FILENAME, "wb") as f:
                pickle.dump(instances_evaluations, f)

        evaluations_kpis = {instance_name: compute_evaluation_kpis(evaluations, args.addition, args.migration)
                            for instance_name, evaluations in instances_evaluations.items()}

        with open(EVALUATIONS_KPIS_PICKLE_FILENAME, "wb") as f:
            pickle.dump(evaluations_kpis, f)

    return evaluations_kpis


def compute_statistics(evaluations_kpis: dict[str, list[list[EvaluationKPIs]]], args):
    if args.aggregate:
        evaluations_kpis = group_evaluations_kpis_by_instance_type(evaluations_kpis, args.scale)

    def extract_kpis(_kpis):
        return [[
            _e_kpi.improvement,
            _e_kpi.cost,
            _e_kpi.schedule_difference,
            _e_kpi.evaluation.duration,
        ] for _e_kpi in _kpis]

    kpis_by_alg_by_inst = np.array(
        [[
            np.array(extract_kpis(algorithms_evaluations_kpis[0]) + extract_kpis(algorithms_evaluations_kpis[1])),
            np.array(extract_kpis(algorithms_evaluations_kpis[2]) + extract_kpis(algorithms_evaluations_kpis[3])),
        ] for instance, algorithms_evaluations_kpis in evaluations_kpis.items()],
        dtype=object
    )

    i_instances = {instance: i for i, instance in enumerate(evaluations_kpis)}

    i_ssira = 0
    i_iira = 1

    n_instances = kpis_by_alg_by_inst.shape[0]
    maxs = np.array([[np.max(alg_kpis, axis=0) for alg_kpis in inst_kpis] for inst_kpis in kpis_by_alg_by_inst])
    mins = np.array([[np.min(alg_kpis, axis=0) for alg_kpis in inst_kpis] for inst_kpis in kpis_by_alg_by_inst])
    avgs = np.array([[np.average(alg_kpis, axis=0) for alg_kpis in inst_kpis] for inst_kpis in kpis_by_alg_by_inst])
    inst_maxs = np.max(maxs, axis=1)
    inst_mins = np.min(mins, axis=1)
    inst_avgs = np.average(avgs, axis=1)

    improving_kpis_by_alg_by_inst = np.array([[kpis[kpis[:, I_improvement] > 0]
                                               for kpis in algs_kpis]
                                              for algs_kpis in kpis_by_alg_by_inst], dtype=object)
    improving_maxs = np.array([[np.max(alg_kpis, initial=0, axis=0) for alg_kpis in inst_kpis] for inst_kpis in improving_kpis_by_alg_by_inst])
    improving_mins = np.array([[np.min(alg_kpis, initial=0, axis=0) for alg_kpis in inst_kpis] for inst_kpis in improving_kpis_by_alg_by_inst])
    improving_avgs = np.array([[np.average(alg_kpis, axis=0) for alg_kpis in inst_kpis] for inst_kpis in improving_kpis_by_alg_by_inst])
    improving_inst_maxs = np.max(improving_maxs, axis=1)
    improving_inst_mins = np.min(improving_mins, axis=1)
    improving_inst_avgs = np.average(improving_avgs, axis=1)

    # Found improvements
    total_improved_instances = np.sum(inst_maxs[:, I_improvement] > 0)
    ssira_improved_instances = np.sum(maxs[:, i_ssira, I_improvement] > 0)
    iira_improved_instances = np.sum(maxs[:, i_iira, I_improvement] > 0)

    # Best found improvements
    ssira_best_improvements = np.sum(inst_maxs[:, I_improvement] == maxs[:, i_ssira, I_improvement])
    iira_best_improvements = np.sum(inst_maxs[:, I_improvement] == maxs[:, i_iira, I_improvement])
    both_best_improvements = np.sum(maxs[:, i_ssira, I_improvement] == maxs[:, i_iira, I_improvement])

    # Cost
    ssira_avg_cost_per_improvement = improving_avgs[:, i_ssira, I_cost] / improving_avgs[:, i_ssira, I_improvement]
    iira_avg_cost_per_improvement = improving_avgs[:, i_iira, I_cost] / improving_avgs[:, i_iira, I_improvement]

    # Schedule difference
    inst_ns = np.array([experiment_instances_info[instance]["n"] for instance in evaluations_kpis])
    ssira_avg_diff_per_job = improving_avgs[:, i_ssira, I_schedule_difference] / inst_ns
    iira_avg_diff_per_job = improving_avgs[:, i_iira, I_schedule_difference] / inst_ns

    print("total_improved_instances", ": ", total_improved_instances)
    print("ssira_improved_instances", ": ", ssira_improved_instances)
    print("iira_improved_instances",  ": ", iira_improved_instances)
    print("ssira_best_improvements",  ": ", ssira_best_improvements)
    print("iira_best_improvements",   ": ", iira_best_improvements)
    print("both_best_improvements",   ": ", both_best_improvements)

    table_data = [[
            instance,
            ssira_avg_diff_per_job[i_instances[instance]],
            ssira_avg_cost_per_improvement[i_instances[instance]],
            iira_avg_diff_per_job[i_instances[instance]],
            iira_avg_cost_per_improvement[i_instances[instance]],
        ] for instance in evaluations_kpis
        ]
    t = tabulate.tabulate(
        table_data,
        ["Instance", "SSIRA avg diff", "SSIRA avg cost/improvement", "IIRA avg diff", "IIRA avg cost/improvement"],
        tablefmt='tsv',
    )
    print(t)

    with open(os.path.join(RESULTS_DIRECTORY, ('aggregated_data.tsv' if args.aggregate else 'data.tsv')), "wt") as f:
        def row(*_vals): print(*_vals, sep='\t', end='\n', file=f)
        row("Instance", "SSIRA avg diff", "IIRA avg diff", "SSIRA avg cost/improvement", "IIRA avg cost/improvement")
        for instance_data in table_data:
            row(*instance_data)


# noinspection PyTypeChecker
def plot(evaluations_kpis, args: argparse.Namespace):
    # Determine whether to save plots to files
    cost_improv = improv_diff = duration_improv = None
    if args.save_plots:
        if args.aggregate:
            cost_improv = os.path.join(RESULTS_DIRECTORY, 'aggregated_cost_improv.pdf')
            improv_diff = os.path.join(RESULTS_DIRECTORY, 'aggregated_improv_diff.pdf')
            duration_improv = os.path.join(RESULTS_DIRECTORY, 'aggregated_duration_improv.pdf')
        else:
            cost_improv = os.path.join(RESULTS_DIRECTORY, 'cost_improv.pdf')
            improv_diff = os.path.join(RESULTS_DIRECTORY, 'improv_diff.pdf')
            duration_improv = os.path.join(RESULTS_DIRECTORY, 'duration_improv.pdf')

    cost_improv_kpis = {instance: [pareto_front_kpis(alg_kpis, x="cost", y="improvement") for alg_kpis in algs_kpis] for instance, algs_kpis in evaluations_kpis.items()}
    improv_diff_kpis = {instance: [pareto_front_kpis(alg_kpis, x="improvement", y="schedule difference") for alg_kpis in algs_kpis] for instance, algs_kpis in evaluations_kpis.items()}
    duration_improv_kpis = {instance: [pareto_front_kpis(alg_kpis, x="duration", y="improvement") for alg_kpis in algs_kpis] for instance, algs_kpis in evaluations_kpis.items()}

    if args.aggregate:
        cost_improv_kpis = group_evaluations_kpis_by_instance_type(cost_improv_kpis, args.scale)
        improv_diff_kpis = group_evaluations_kpis_by_instance_type(improv_diff_kpis, args.scale)
        duration_improv_kpis = group_evaluations_kpis_by_instance_type(duration_improv_kpis, args.scale)

    # Plotting
    ncols = 2 if args.aggregate else 8
    dimensions = (8, 11) if args.aggregate else (35, 24)
    layout = ({"hspace": 0.50, "wspace": 0.3, "top": 0.97, "bottom": 0.13, "left": 0.10, "right": 0.99} if args.aggregate else
              {"hspace": 0.25, "wspace": 0.3, "top": 0.98, "bottom": 0.06, "left": 0.03, "right": 0.99})
    labels = ["SSIRA (improvement sort)", "SSIRA (time sort)", "IIRA (AUAU)", "IIRA (MRUR)"]
    plot_evaluations(cost_improv_kpis, value_axes=("cost", "improvement"), save_as=cost_improv, ncols=ncols, dimensions=dimensions, layout=layout, inverse_order=not args.aggregate, labels=labels)
    plot_evaluations(improv_diff_kpis, value_axes=("improvement", "schedule difference"), save_as=improv_diff, ncols=ncols, dimensions=dimensions, layout=layout, inverse_order=not args.aggregate, labels=labels)
    plot_evaluations(duration_improv_kpis, value_axes=("duration", "improvement"), save_as=duration_improv, ncols=ncols, dimensions=dimensions, layout=layout, inverse_order=not args.aggregate, labels=labels)


def main(args: argparse.Namespace):
    evaluations_kpis = get_evaluations_kpis(args)
    compute_statistics(evaluations_kpis, args)
    plot(evaluations_kpis, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true", default=False, help="Determines whether to aggregate instances by type")
    parser.add_argument("--scale", action="store_true", default=False, help="Determines whether to scale evaluations KPIs")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Determines whether to save plots to files")
    parser.add_argument("--addition", action="store", default=5, type=int, help="Cost of capacity addition")
    parser.add_argument("--migration", action="store", default=1, type=int, help="Cost of capacity migration")
    main(parser.parse_args())
