import os

from bottlenecks.drawing import plot_solution
from bottlenecks.improvements import MetricsRelaxingAlgorithm, MetricsRelaxingAlgorithmSettings, \
    TimeVariableConstraintRelaxingAlgorithm, TimeVariableConstraintRelaxingAlgorithmSettings
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
    plot(base_solution, "solution_base")

    iira = MetricsRelaxingAlgorithm()
    ssira = TimeVariableConstraintRelaxingAlgorithm()

    # The intermediate solution plots were obtained by calling the `plot` function from within the algorithms
    # using step-into debugging as the `EvaluationAlgorithm` currently does not support internal plotting.
    evaluation_iira = iira.evaluate(instance, MetricsRelaxingAlgorithmSettings("auac", 8, "around", 1, 1, 8))
    evaluation_ssira = ssira.evaluate(instance, TimeVariableConstraintRelaxingAlgorithmSettings(1, 1, 1, "time"))

    plot(evaluation_iira.solution, "solution_iira")
    plot(evaluation_ssira.solution, "solution_ssira")
    serialize_evaluations([evaluation_iira, evaluation_ssira], EXAMPLE_DIRECTORY)


if __name__ == "__main__":
    main()
