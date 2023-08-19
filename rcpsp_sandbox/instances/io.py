import os
from typing import IO, Iterable

from instances.problem_instance import ProblemInstance, Project, Job, Precedence, Resource, ResourceConsumption,\
                                       ResourceType
from instances.instance_builder import InstanceBuilder
from instances.utils import try_open, chunk

PSPLIB_KEY_VALUE_SEPARATOR: str = ':'


def parse_psplib(filename: str,
                 name_as: str or None = None) -> ProblemInstance:
    return try_open(filename, __parse_psplib_internal, name_as=name_as)


def serialize_psplib(instance: ProblemInstance,
                     filename: str) -> None:
    # TODO
    pass


def __parse_psplib_internal(file: IO,
                            name_as: str or None) -> ProblemInstance:
    def skip_lines(count: int) -> None:
        nonlocal line_num
        for _ in range(count):
            file.readline()
            line_num += 1

    def asterisks() -> None:
        skip_lines(1)

    def parse_split_line(target_indices: tuple,
                         end_of_line_array_index: int = -1,
                         move_line: bool = True) -> tuple:
        nonlocal line_num

        line = file.readline()
        content = line.split(maxsplit=end_of_line_array_index)
        if (len(content) - 1) < end_of_line_array_index:  # If the end-of-line array is empty...
            content.append("")  # ...insert empty array string
        elif (len(content) - 1) < max(target_indices):
            raise ParseError(file, line_num, "Line contains less values than expected")

        if move_line:
            line_num += 1

        return tuple(content[i] for i in target_indices)

    def try_parse_value(value_str) -> int:
        nonlocal line_num

        if not value_str.isdecimal():
            raise ParseError(file, line_num, "Integer value expected on key-value line")

        value = int(value_str)
        return value

    def _check_key(expected_key: str,
                   key: str):
        nonlocal line_num

        if key != expected_key:
            raise ParseError(file, line_num, "Unexpected key on key-value line")

    def parse_key_value_line(key_value_indices: tuple[int, int],
                             expected_key: str) -> int:
        nonlocal line_num

        key: str
        value_str: str
        key, value_str = parse_split_line(key_value_indices, move_line=False)
        _check_key(expected_key, key)
        value = try_parse_value(value_str)

        line_num += 1
        return value

    def parse_colon_key_value_line(expected_key):
        nonlocal line_num

        key: str
        value_str: str
        key, value_str = file.readline().split(sep=PSPLIB_KEY_VALUE_SEPARATOR)
        _check_key(expected_key, key.strip())
        value = try_parse_value(value_str.strip())

        line_num += 1
        return value

    line_num = 1

    asterisks()
    skip_lines(2)  # file with basedata   &   initial value random generator

    asterisks()
    project_count = parse_colon_key_value_line("projects")
    job_count = parse_colon_key_value_line("jobs (incl. supersource/sink )")
    horizon = parse_colon_key_value_line("horizon")
    skip_lines(1)  # RESOURCES list header
    _renewable_resource_count = parse_key_value_line((1, 3), "renewable")
    _nonrenewable_resource_count = parse_key_value_line((1, 3), "nonrenewable")
    _doubly_constrained_resource_count = parse_key_value_line((1, 4), "doubly")  # "doubly constrained" split as two...

    asterisks()
    skip_lines(2)  # PROJECT INFORMATION   &   projects header (pronr. #jobs rel.date duedate tardcost  MPM-Time)
    projects: list[Project] = []
    for _ in range(project_count):
        id_project_str, due_date_str, tardiness_cost_str = parse_split_line((0, 3, 4))  # ignore pronr, rel.date, MPM-Time
        projects.append(Project(try_parse_value(id_project_str),
                                try_parse_value(due_date_str),
                                try_parse_value(tardiness_cost_str)))

    asterisks()
    skip_lines(2)  # PRECEDENCE RELATIONS   &   precedences header (jobnr. #modes #successors successors)
    precedences: list[Precedence] = []
    for _ in range(job_count):
        id_job_str, successor_count_str, successors_str = parse_split_line((0, 2, 3), end_of_line_array_index=3)  # ignore #mode
        if try_parse_value(successor_count_str) > 0:
            id_job = try_parse_value(id_job_str)
            precedences += [Precedence(id_child=id_job, id_parent=try_parse_value(successor_str))
                            for successor_str in successors_str.split()]

    asterisks()
    skip_lines(1)  # REQUESTS/DURATIONS
    resources_str, = parse_split_line((3,), end_of_line_array_index=3)
    resource_type_id_pairs = chunk(resources_str.split(), 2)
    resource_data: list[tuple[int, ResourceType]] = [(try_parse_value(id_resource_str), ResourceType(resource_type_str))
                                                     for resource_type_str, id_resource_str in resource_type_id_pairs]

    asterisks()
    job_data: list[tuple[int, int, dict[int, int]]] = []
    for _ in range(job_count):
        id_job_str, duration, consumptions = parse_split_line((0, 2, 3), end_of_line_array_index=3)  # ignore mode
        consumption_by_resource_id = {resource[0]: amount
                                      for resource, amount in zip(resource_data,
                                                                  map(try_parse_value, consumptions.split()))}
        job_data.append((try_parse_value(id_job_str), try_parse_value(duration), consumption_by_resource_id))

    asterisks()
    skip_lines(2)  # RESOURCE AVAILABILITIES   &   Resource name headers (assuming same order of resources as above)
    capacities_str, = parse_split_line((0,), end_of_line_array_index=0)
    capacities: Iterable[int] = map(try_parse_value, capacities_str.split())

    resources: list[Resource] = [Resource(id_resource, resource_type, capacity)
                                 for (id_resource, resource_type), capacity in zip(resource_data, capacities)]
    jobs: list[Job] = [Job(id_job,
                           ResourceConsumption(duration,
                                               {resource: consumption_by_resource_id[resource.id_resource]
                                                for resource in resources}))
                       for id_job, duration, consumption_by_resource_id in job_data]

    asterisks()

    builder = InstanceBuilder()

    builder.add_projects(projects)
    builder.add_resources(resources)
    builder.add_jobs(jobs)
    builder.add_precedences(precedences)

    builder.set(name=(name_as if (name_as is not None) else os.path.basename(file.name)),
                horizon=horizon)

    return builder.build_instance()


class ParseError(Exception):
    def __init__(self,
                 file: IO,
                 line_num: int,
                 message: str):
        super().__init__(f"[{file.name}:{line_num}] {message}")
