# Script to demonstrate the usage of the bottleneck algorithms on a simple example instance.

import os

from bottlenecks.drawing import plot_solution
from bottlenecks.improvements import IdentificationIndicatorRelaxingAlgorithm, IdentificationIndicatorRelaxingAlgorithmSettings, \
    ScheduleSuffixIntervalRelaxingAlgorithm, ScheduleSuffixIntervalRelaxingAlgorithmSettings
from bottlenecks.io import serialize_evaluations
from instances.io import parse_json
from solver.solver import Solver


SAVE = True
EXAMPLE_DIRECTORY = os.path.join('..', 'example')


def plot(solution, name):
    plot_solution(solution,
                  dimensions=(8, 6),
                  panel_heights=[1, 2, 2],
                  job_interval_levels={1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 2, 7: 0, 8: 0, 9: 0},
                  component_colors={6: "#c1ff7f", 9: "#7fffff"},
                  title=False,
                  scale=0.9,
                  legend=False,
                  # split_consumption=True,
                  offset_deadlines=False,
                  save_as=(os.path.join(EXAMPLE_DIRECTORY, name+'.pdf') if SAVE else None),
                  highlight_non_periodical_consumption=True,
                  )


def main():
    instance = parse_json(os.path.join(EXAMPLE_DIRECTORY, "instance.json"), name_as="example", is_extended=True)

    base_solution = Solver().solve(instance)

    iira = IdentificationIndicatorRelaxingAlgorithm()
    ssira = ScheduleSuffixIntervalRelaxingAlgorithm()

    # The intermediate solution plots were obtained by calling the `plot` function from within the algorithms
    # using step-into debugging as the `EvaluationAlgorithm` currently does not support internal plotting.
    evaluation_iira = iira.evaluate(instance, IdentificationIndicatorRelaxingAlgorithmSettings("auau", 8, "around", 1, 1, 8))
    evaluation_ssira = ssira.evaluate(instance, ScheduleSuffixIntervalRelaxingAlgorithmSettings(1, 1, 1, "time"))

    plot(base_solution, "solution_base")
    plot(evaluation_iira.solution, "solution_iira")
    plot(evaluation_ssira.solution, "solution_ssira")

    serialize_evaluations([evaluation_iira, evaluation_ssira], EXAMPLE_DIRECTORY)


if __name__ == "__main__":
    main()