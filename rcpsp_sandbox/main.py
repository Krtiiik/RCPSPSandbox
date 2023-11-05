import argparse
import random

from instances.io import parse_psplib, parse_json, serialize_json, serialize_psplib
from instances.drawing import draw_instance_graph
from utils import print_error
from solver.drawing import plot_solution
from solver.solver import Solver
from instances.problem_modifier import modify_instance

parser = argparse.ArgumentParser()
parser.add_argument("input_file", default=None, type=str, help="Path to instance file")
parser.add_argument("--input_format", default="psplib", type=str, help="Format of the input instance file (psplib or json)")
parser.add_argument("--output_file", default="psplib", type=str, help="Output file path")
parser.add_argument("--output_format", default="psplib", type=str, help="Format of the output instance file (psplib or json)")
parser.add_argument("--seed", default=42, type=int, help="Seed for random generators")


def main(args: argparse.Namespace):
    random.seed(args.seed)

    if args.input_format == "psplib" or args.input_file.endswith((".sm", ".mm", ".rp")):
        instance = parse_psplib(args.input_file, name_as="Test")
    elif args.input_format == "json" or args.input_file.endswith((".json",)):
        instance = parse_json(args.input_file)
    else:
        print_error(f"Invalid input format specified: {args.input_format}")
        return

    draw_instance_graph(instance, block=True)

    instance = modify_instance(instance) \
        .split_job_components(split="paths")\
        .generate_modified_instance()

    draw_instance_graph(instance, block=True, highlighted_nodes=[c.id_root_job for c in instance.components])

    # result = Solver().solve(instance)
    # plot_solution(instance, result.solution)

    if args.output_format == "psplib":
        serialize_psplib(instance, args.output_file)
    elif args.output_format == "json":
        serialize_json(instance, args.output_file)


if __name__ == "__main__":
    main(parser.parse_args())
