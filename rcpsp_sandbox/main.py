import sys

from instances.io import parse_psplib, serialize_json
from instances.drawing import draw_instance_graph
from utils import print_error


def main():
    if len(sys.argv) != 2:
        print_error("Invalid command arguments count")
        if len(sys.argv) == 1:
            filename = input("Enter input file name: ")
        else:
            return
    else:
        filename = sys.argv[1]

    instance = parse_psplib(filename, name_as="Test")
    draw_instance_graph(instance, block=True)
    serialize_json(instance, "super_cupr_output.json")
    print("Done")


if __name__ == "__main__":
    main()
