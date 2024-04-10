import argparse
import functools
import os
import random
from os import path
from typing import Iterable, Tuple, Callable

import instances.io as iio
from instances.drawing import plot_components
from instances.problem_instance import ProblemInstance, AvailabilityInterval
from instances.problem_modifier import modify_instance


parser = argparse.ArgumentParser()
parser.add_argument("--data_directory", type=str, required=True, default='.', help="Location of the required PSPLIB instance files")
parser.add_argument("--output_directory", type=str, help="Output location for the generated instances")
parser.add_argument("--output_format", type=str, choices=["psplib", "json"], help="Format of the generated instances")


MORNING = 1
AFTERNOON = 2
NIGHT = 4
shift_intervals = [
    [],                     # 0 =         |           |
    [( 6, 14)],             # 1 = Morning |           |
    [(14, 22)],             # 2 =         | Afternoon |
    [( 6, 22)],             # 3 = Morning | Afternoon |
    [( 0,  6), (22, 24)],   # 4 =         |           | Night
    [( 0, 14), (22, 24)],   # 5 = Morning |           | Night
    [( 0,  6), (14, 24)],   # 6 =         | Afternoon | Night
    [( 0, 24)],             # 7 = Morning | Afternoon | Night
]
experiment_instances = {
    "instance01", "instance02", "instance03", "instance04", "instance05",
    "instance06", "instance07", "instance08", "instance09", "instance10",
}


def build_01(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "j301_1.sm"  # TODO
    inst = iio.parse_psplib(os.path.join(location, instance_filename))
    inst = modify_instance(inst).split_job_components("gradual", 2).assign_job_due_dates("gradual", gradual_base=0, gradual_interval=(-1, 1)).generate_modified_instance()
    return inst

    shifts = [
    ]



def build_02(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "j602_6.sm"  # TODO
    inst = iio.parse_psplib(os.path.join(location, instance_filename))
    inst = modify_instance(inst).split_job_components("gradual", 2).assign_job_due_dates("gradual", gradual_base=0, gradual_interval=(-1, 0)).generate_modified_instance()
    return inst

    instance_filename = "" # TODO
    pass


def build_03(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "" # TODO
    pass


def build_04(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "" # TODO
    pass


def build_05(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "" # TODO
    pass


def build_06(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "" # TODO
    pass


def build_07(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "" # TODO
    pass


def build_08(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "" # TODO
    pass


def build_09(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "" # TODO
    pass


def build_10(data_directory, output_directory) -> ProblemInstance:
    instance_filename = "" # TODO
    pass


def parse_and_process(args: argparse.Namespace,
                      instance_name: str,
                      split_level: int,
                      availabilities: dict[int, Iterable[Tuple[int, int]]],
                      generated_instance_name: str,
                      ) -> ProblemInstance:
    instance = iio.parse_psplib(path.join(args.data_directory, instance_name))
    instance = modify_instance(instance) \
               .split_job_components(split="gradual", gradual_level=split_level) \
               .assign_resource_availabilities(availabilities=availabilities) \
               .generate_modified_instance(generated_instance_name)  # TODO

    return instance


instance_build_functions = {
    "instance01": build_01, "instance02": build_02, "instance03": build_03, "instance04": build_04, "instance05": build_05,
    "instance06": build_06, "instance07": build_07, "instance08": build_08, "instance09": build_09, "instance10": build_10,
}


def build_instance(instance_name: str,
                   base_instance_directory: str,
                   output_directory: str,
                   serialize: bool = True,
                   ) -> ProblemInstance:
    if instance_name not in instance_build_functions:
        raise ValueError(f"Cannot find build method for instance {instance_name}")

    instance = instance_build_functions[instance_name](base_instance_directory, output_directory)

    if serialize:
        iio.serialize_json(instance, os.path.join(output_directory, instance_name+'.json'), is_extended=True)

    return instance


def main(args: argparse.Namespace):
    serializer = functools.partial(iio.serialize_json if args.output_format == "json" else iio.serialize_psplib,
                                   is_extended=True)

    builds = [
        build_01, build_02, build_03, build_04, build_05, build_06, build_07, build_08, build_09, build_10,
    ]

    for build in builds:
        random.seed(42)
        instance = build(args.data_directory, args.output_directory)
        plot_components(instance)
        serializer(instance, os.path.join(args.output_directory, instance.name))


if __name__ == "__main__":
    # main(parser.parse_args())

    args = argparse.Namespace(data_directory="../data/instances", output_directory="../data/extended_instances", output_format='psplib')
    main(args)
