import random

import instances.io as iio
from instances.drawing import draw_instance_graph
from solver.drawing import print_difference
from solver.model_builder import build_model, edit_model
from solver.solver import Solver
from instances.problem_modifier import modify_instance

# INSTANCE_LOC = "./psplib"
INSTANCE_LOC = "../../Data/RCPSP/j30/j3040_10.sm"

# SPLIT_COMPONENTS = True
SPLIT_RESOURCE_CONSUMPTION = False


def main():
    random.seed(42)

    instance = iio.parse_psplib(INSTANCE_LOC)
    draw_instance_graph(instance, block=False)

    instance = modify_instance(instance) \
               .split_job_components(split="paths") \
               .assign_resource_availabilities() \
               .generate_modified_instance()

    model = build_model(instance) \
            .optimize_model() \
            .get_model()

    solution = Solver().solve(instance, model=model)
    solution.plot(split_components=False, split_resource_consumption=SPLIT_RESOURCE_CONSUMPTION)
    solution.plot(split_components=True, split_resource_consumption=SPLIT_RESOURCE_CONSUMPTION)

    model = edit_model(model, instance) \
            .change_resource_capacities({instance.resources[2]: [(6, 10, 27)]}) \
            .get_model()

    solution_alt = Solver().solve(instance, model=model)
    solution_alt.plot(split_components=False, split_resource_consumption=SPLIT_RESOURCE_CONSUMPTION)
    solution_alt.plot(split_components=True, split_resource_consumption=SPLIT_RESOURCE_CONSUMPTION)

    print_difference(solution, instance, solution_alt, instance)


if __name__ == "__main__":
    main()
