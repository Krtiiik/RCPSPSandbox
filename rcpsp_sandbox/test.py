import functools
import os.path
import random

import bottlenecks.metrics as mtr
import instances.io as iio
from instances.algorithms import compute_earliest_completion_times
from instances.drawing import draw_instance_graph, draw_components_graph
from solver.drawing import print_difference, plot_solution
from solver.model_builder import build_model, edit_model
from solver.solver import Solver
from instances.problem_modifier import modify_instance

# INSTANCE_LOC = "./psplib"
# INSTANCE_LOC = "../../Data/j30/j3040_10.sm"
# INSTANCE_LOC = "../../Data/j30/j301_1.sm"
INSTANCE_LOC = "../../Data/j30/j306_1.sm"

# SPLIT_COMPONENTS = True
SPLIT_RESOURCE_CONSUMPTION = False


def main():
    def evaluate(inst, sol):
        mtr.print_evaluation(inst, [
            # mtr.evaluate_solution(solution, mtr.machine_workload, evaluation_name="MU"),
            # mtr.evaluate_solution(solution, mtr.machine_utilization_rate, evaluation_name="MUR"),
            # mtr.evaluate_solution(solution, mtr.average_uninterrupted_active_duration, evaluation_name="AUAD"),
            mtr.evaluate_solution(sol, mtr.machine_resource_workload, evaluation_name="MRW"),
            mtr.evaluate_solution(sol, mtr.machine_resource_utilization_rate, evaluation_name="MRUR"),
            mtr.evaluate_solution(sol, functools.partial(mtr.average_uninterrupted_active_consumption, average_over="consumption"), evaluation_name="AUAC_1"),
            mtr.evaluate_solution(sol, functools.partial(mtr.average_uninterrupted_active_consumption, average_over="consumption ratio"), evaluation_name="AUAC_2"),
            mtr.evaluate_solution(sol, functools.partial(mtr.average_uninterrupted_active_consumption, average_over="averaged consumption"), evaluation_name="AUAC_3"),
            mtr.evaluate_solution(sol, functools.partial(mtr.cumulative_delay, earliest_completion_times=compute_earliest_completion_times(inst)), evaluation_name="CUMULATIVE_DELAY"),
        ])

    instance = iio.parse_psplib(INSTANCE_LOC)
    draw_instance_graph(instance)
    instance = modify_instance(instance) \
               .split_job_components(split="gradual", gradual_level=2) \
               .assign_job_due_dates(choice="gradual", gradual_base=-10, gradual_interval=(-10, 0)) \
               .assign_resource_availabilities({r.id_resource: [(6, 22)] for r in instance.resources}) \
               .generate_modified_instance()
    draw_instance_graph(instance)
    model = get_model(instance)
    solution = Solver().solve(instance, model)

    evaluate(instance, solution)
    solution.plot(split_components=False, split_resource_consumption=SPLIT_RESOURCE_CONSUMPTION)
    solution.plot(split_components=True, split_resource_consumption=SPLIT_RESOURCE_CONSUMPTION)

    instance_alt = modify_availability(instance, {
        instance.resources[2]: [(6, 10, 27)]
    })
    model_alt = get_model(instance_alt, solution_difference=solution)
    solution_alt = Solver().solve(instance_alt, model_alt)

    evaluate(instance_alt, solution_alt)
    plot_solution(instance_alt, solution_alt, split_components=False, split_resource_consumption=SPLIT_RESOURCE_CONSUMPTION)
    plot_solution(instance_alt, solution_alt, split_components=True, split_resource_consumption=SPLIT_RESOURCE_CONSUMPTION)

    print_difference(solution, instance, solution_alt, instance)


def get_model(inst,
              resource_constraint=True,
              solution_difference=None
              ):
    mdl = build_model(inst).with_precedences()
    if resource_constraint:
        mdl = mdl.with_resource_constraints()
    mdl.optimize_model()
    if solution_difference:
        mdl = mdl.minimize_model_solution_difference(solution_difference)
    return mdl.get_model()


def modify_availability(inst, changes):
    return modify_instance(inst).change_resource_availability(changes).generate_modified_instance()


if __name__ == "__main__":
    random.seed(42)
    # main()

    # base = os.path.join('..', '..', 'Data', 'j60')
    # for i in range(100):
    #     random.seed(42)
    #     name = f'j60{1+i//10}_{1+i%10}.sm'
    #     instance = iio.parse_psplib(os.path.join(base, name))
    #     instance = modify_instance(instance) \
    #         .split_job_components(split="gradual", gradual_level=2) \
    #         .generate_modified_instance()
    #     draw_components_graph(instance, save_as=os.path.join('..', 'data', 'j60', name+'.jpg'))

    solver = Solver()

    instance = iio.parse_psplib(os.path.join("..", "..", "Data", "j60", "j602_6.sm"))
    # instance = iio.parse_psplib(os.path.join("..", "..", "Data", "j120", "j1202_6.sm"))
    instance = modify_instance(instance) \
               .split_job_components(split="gradual", gradual_level=2) \
               .assign_job_due_dates(choice="gradual", gradual_base=0, gradual_interval=(-1, 1)) \
               .assign_resource_availabilities({r.id_resource: [(6, 22)] for r in instance.resources}) \
               .generate_modified_instance()
    draw_components_graph(instance)

    model = get_model(instance, resource_constraint=False)
    solution = solver.solve(instance, model)
    plot_solution(instance, solution, split_resource_consumption=True, plot_resource_capacity=False)

    model_alt = get_model(instance)
    solution_alt = solver.solve(instance, model_alt)
    plot_solution(instance, solution_alt, split_resource_consumption=True, plot_resource_capacity=True)

    instance_alt = modify_instance(instance).change_resource_availability({instance.resources[0]: [(6, 14, 20)]}).generate_modified_instance()
    model_alt = get_model(instance_alt)
    solution_alt = solver.solve(instance_alt, model_alt)
    plot_solution(instance_alt, solution_alt)
    plot_solution(instance_alt, solution_alt, split_resource_consumption=True, plot_resource_capacity=True)
    plot_solution(instance_alt, solution_alt, split_components=True, plot_resource_capacity=True)

    # print_difference(solution, instance, solution_alt, instance_alt)
