import os
from collections import namedtuple
from typing import Iterable

import instances.io as iio
from bottlenecks.utils import compute_longest_shift_overlap
from instances.problem_instance import ProblemInstance
from instances.problem_modifier import modify_instance


InstanceSetup = namedtuple("InstanceSetup", ("base_filename", "name", "gradual_level", "shifts", "due_dates",
                                             "tardiness_weights", "target_job", "scaledown_durations"))


MORNING = 1
AFTERNOON = 2
NIGHT = 4
SHIFT_INTERVALS = [
    [],                     # 0 =         |           |
    [( 6, 14)],             # 1 = Morning |           |
    [(14, 22)],             # 2 =         | Afternoon |
    [( 6, 22)],             # 3 = Morning | Afternoon |
    [( 0,  6), (22, 24)],   # 4 =         |           | Night
    [( 0, 14), (22, 24)],   # 5 = Morning |           | Night
    [( 0,  6), (14, 24)],   # 6 =         | Afternoon | Night
    [( 0, 24)],             # 7 = Morning | Afternoon | Night
]


def build_shifts(shifts: dict[str, int]) -> dict[str, list[tuple[int, int]]]:
    return {r_key: SHIFT_INTERVALS[shift_flags] for r_key, shift_flags in shifts.items()}


experiment_instances: dict[str, InstanceSetup] = {
    # ------------------------------------------------------------------------------------------------------------------
    "instance01": InstanceSetup(
        base_filename="j3011_4.sm",
        name="instance01",
        gradual_level=2,
        shifts={
            "R1": MORNING | AFTERNOON,
            "R2": MORNING | AFTERNOON,
            "R3": MORNING | AFTERNOON,
            "R4": MORNING | AFTERNOON,
        },
        due_dates={
            23: 46,
            28: 46,
            29: 46,
            30: 46,
            32: 46,
        },
        tardiness_weights={
            23: 2,
            28: 1,
            29: 3,
            30: 1,
            32: 1,
        },
        target_job=29,
        scaledown_durations=False,
    ),
    # ------------------------------------------------------------------------------------------------------------------
    "instance02": InstanceSetup(
        base_filename="j3010_2.sm",
        name="instance02",
        gradual_level=2,
        shifts={
            "R1": MORNING | AFTERNOON | NIGHT,
            "R2": MORNING | AFTERNOON,
        },
        due_dates={
            26: 22,
            27: 22,
            29: 22,
            30: 22,
            32: 22,
        },
        tardiness_weights={
            26: 1,
            27: 1,
            29: 1,
            30: 1,
            32: 1,
        },
        target_job=26,
        scaledown_durations=False,
    ),
    # ------------------------------------------------------------------------------------------------------------------
    "instance03": InstanceSetup(
        base_filename="j6010_7.sm",
        name="instance03",
        gradual_level=1,
        shifts={
            "R1": MORNING | AFTERNOON,
        },
        due_dates={
            59: 70,
            60: 70,
            62: 70,
        },
        tardiness_weights={
            59: 1,
            60: 1,
            62: 3,
        },
        target_job=62,
        scaledown_durations=False,
    ),
    # ------------------------------------------------------------------------------------------------------------------
    "instance04": InstanceSetup(
        base_filename="j6011_10.sm",
        name="instance04",
        gradual_level=1,
        shifts={
            "R1": MORNING | AFTERNOON,
            "R2": MORNING | AFTERNOON,
            "R3":           AFTERNOON | NIGHT,
            "R4": MORNING | AFTERNOON | NIGHT,
        },
        due_dates={
            59: 70,
            60: 70,
            62: 70,
        },
        tardiness_weights={
            59: 1,
            60: 1,
            62: 3,
        },
        target_job=62,  # TODO
        scaledown_durations=True,
    ),
    # ------------------------------------------------------------------------------------------------------------------
}


def parse_and_process(data_directory: str, output_directory: str,
                      setup: InstanceSetup,
                      ) -> ProblemInstance:
    shifts = build_shifts(setup.shifts)

    # Parse
    instance = iio.parse_psplib(os.path.join(data_directory, setup.base_filename))

    # Modify
    instance_builder = modify_instance(instance)

    # Remove unused resources
    if len(shifts) < len(instance.resources):
        instance_builder.remove_resources(set(r.key for r in instance.resources) - set(shifts))

    # Component splitting, availabilities, due dates
    instance = instance_builder.split_job_components(split="gradual", gradual_level=setup.gradual_level) \
               .assign_resource_availabilities(availabilities=shifts) \
               .assign_job_due_dates('uniform', interval=(0, 0)) \
               .assign_job_due_dates(due_dates=setup.due_dates, overwrite=True) \
               .with_target_job(setup.target_job) \
               .generate_modified_instance(setup.name)

    # Scaling down long job durations
    if setup.scaledown_durations:
        longest_overlap = compute_longest_shift_overlap(instance)
        assert longest_overlap != 0, "There is no shift overlap"
        instance_builder.scaledown_job_durations(longest_overlap)

    # Tardiness weights
    for component in instance.components:
        component.weight = setup.tardiness_weights[component.id_root_job]

    return instance


def build_instance(instance_name: str,
                   base_instance_directory: str,
                   output_directory: str,
                   serialize: bool = True,
                   ) -> ProblemInstance:
    if instance_name not in experiment_instances:
        raise ValueError(f'Unrecognized experiment instance "{instance_name}"')

    instance_setup = experiment_instances[instance_name]
    instance = parse_and_process(base_instance_directory, output_directory, instance_setup)

    if serialize:
        iio.serialize_json(instance, os.path.join(output_directory, instance_name+'.json'), is_extended=True)

    return instance


if __name__ == "__main__":

    import os
    import functools
    from bottlenecks.drawing import plot_solution
    from instances.drawing import plot_components
    from solver.solver import Solver
    from bottlenecks.metrics import evaluate_solution, average_uninterrupted_active_consumption, print_evaluation

    instance_name = 'instance04'
    instance = build_instance(instance_name, os.path.join('..', 'data', 'base_instances'), os.path.join('..', 'data', 'base_instances'))
    instance.horizon = 2000
    plot_components(instance)
    solution = Solver().solve(instance)
    plot_solution(solution, split_consumption=True, orderify_legends=True,
                  # dimensions=(8, 8)
                  )
    print_evaluation(solution.instance, [
         evaluate_solution(solution, functools.partial(average_uninterrupted_active_consumption, average_over="consumption ratio"))
    ])

    exit()

    from instances.drawing import plot_components
    from instances.problem_modifier import modify_instance
    import os
    inst = 'j60'
    base_dir = os.path.join('..', '..', 'Data', inst)
    filenames = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    instances = [iio.parse_psplib(f, os.path.basename(f).split('.')[0]) for f in filenames]

    instances_ = []
    for instance in instances:
        random.seed(42)
        instances_.append(modify_instance(instance)
                          .split_job_components(split="gradual", gradual_level=1)
                          .assign_job_due_dates('gradual', gradual_base=0, gradual_interval=(0, 1))
                          .generate_modified_instance(instance.name + '_split'))
    instances = instances_
    for instance in instances[:50]:
        plot_components(instance, save_as=os.path.join('..', 'insts', inst, instance.name+'.png'))

