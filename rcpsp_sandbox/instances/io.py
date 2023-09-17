import json
import os
from typing import IO, Iterable, Any
from collections import defaultdict

from instances.problem_instance import ProblemInstance, Project, Job, Precedence, Resource, ResourceConsumption, \
    ResourceType, Component, ResourceShiftMode
from instances.instance_builder import InstanceBuilder
from instances.utils import try_open_read, chunk

PSPLIB_KEY_VALUE_SEPARATOR: str = ':'


def parse_psplib(filename: str,
                 name_as: str or None = None,
                 is_extended: bool = False) -> ProblemInstance:
    return try_open_read(filename, __parse_psplib_internal, name_as=name_as, is_extended=is_extended)


def serialize_psplib(instance: ProblemInstance,
                     filename: str,
                     is_extended: bool = False) -> None:
    psplib_str = __serialize_psplib_internal(instance, is_extended)
    with open(filename, "wt") as file:
        file.write(psplib_str)


def parse_json(filename: str,
               name_as: str or None = None) -> ProblemInstance:
    instance_object = try_open_read(filename, json.load)

    __check_json_parse_object(instance_object)

    instance_builder = InstanceBuilder()

    # instance properties
    instance_builder.set(name=name_as if name_as is not None else instance_object["Name"],
                         horizon=instance_object["Horizon"])

    # resources
    resources = [Resource(r["Id"], ResourceType(r["Type"]), r["Capacity"]) for r in instance_object["Resources"]]
    resources_by_key = {r.key: r for r in resources}
    instance_builder.add_resources(resources)

    # projects
    instance_builder.add_projects(Project(p["Id"], p["Due date"], p["Tardiness cost"])
                                  for p in instance_object["Projects"])

    # jobs
    jobs = [Job(j["Id"],
                ResourceConsumption(j["Resource consumption"]["Duration"],
                                    ({resources_by_key[key]: consumption
                                      for key, consumption in j["Resource consumption"]["Consumptions"].items()})))
            for j in instance_object["Jobs"]]
    instance_builder.add_jobs(jobs)

    # precedences - serialized as successors of each job, need to be extracted from job data
    instance_builder.add_precedences(Precedence(j["Id"], successor)
                                     for j in instance_object["Jobs"]  # child
                                     for successor in j["Successors"])  # successor parent of child

    return instance_builder.build_instance()


def serialize_json(instance: ProblemInstance,
                   filename: str) -> None:
    json_str = json.dumps(instance, cls=ProblemInstanceJSONSerializer)
    with open(filename, "wt") as file:
        file.write(json_str)


def __parse_psplib_internal(file: IO,
                            name_as: str or None,
                            is_extended: bool) -> ProblemInstance:
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
            raise ParseError.in_file(file, line_num, "Line contains less values than expected")

        if move_line:
            line_num += 1

        return tuple(content[i] for i in target_indices)

    def try_parse_value(value_str) -> int:
        nonlocal line_num

        if not value_str.isdecimal():
            raise ParseError.in_file(file, line_num, "Integer value expected on key-value line")

        value = int(value_str)
        return value

    def _check_key(expected_key: str,
                   key: str):
        nonlocal line_num

        if key != expected_key:
            raise ParseError.in_file(file, line_num, "Unexpected key on key-value line")

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

    def build():
        builder = InstanceBuilder()

        builder.add_projects(projects)
        builder.add_resources(resources)
        builder.add_jobs(jobs)
        builder.add_precedences(precedences)

        if is_extended:
            builder.add_components(components)

        builder.set(name=(name_as if (name_as is not None) else os.path.basename(file.name)),
                    horizon=horizon)

        return builder.build_instance()

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

    if not is_extended:
        return build()

    skip_lines(1)  # DUE DATES
    jobs_by_id: dict[int, Job] = {j.id_job: j for j in jobs}
    for _ in range(job_count):
        id_job_str, due_date_str = parse_split_line((0, 1))
        id_job, due_date = try_parse_value(id_job_str), try_parse_value(due_date_str)
        jobs_by_id[id_job].due_date = due_date

    asterisks()
    skip_lines(1)  # FINISHED_TASKS
    completed_jobs_str, = parse_split_line((0,), end_of_line_array_index=0)
    completed_jobs = map(try_parse_value, completed_jobs_str.split())
    for job_id in completed_jobs:
        jobs_by_id[job_id].completed = True

    asterisks()
    skip_lines(1)  # COMPONENTS
    components_str, = parse_split_line((0,), end_of_line_array_index=0)
    components_root_jobs = map(try_parse_value, components_str.split())

    asterisks()
    skip_lines(1)  # COMPONENT_WEIGHTS
    components_weights_str, = parse_split_line((0,), end_of_line_array_index=0)
    components_weights = map(try_parse_value, components_weights_str.split())

    components = [Component(root_job_id, weight)
                  for root_job_id, weight in zip(components_root_jobs, components_weights)]

    asterisks()
    skip_lines(1)  # RESOURCE SHIFT MODES
    resource_by_id = {r.id_resource: r for r in resources}
    for _ in range(len(resources)):
        id_resource_str, shift_mode_str = parse_split_line((0, 1))
        id_resource, shift_mode = try_parse_value(id_resource_str), ResourceShiftMode(try_parse_value(shift_mode_str))
        resource_by_id[id_resource].shift_mode = shift_mode

    return build()


def __serialize_psplib_internal(instance: ProblemInstance, is_extended: bool) -> str:
    output = ""

    def line(content):
        nonlocal output
        output += content.rstrip() + "\n"

    def asterisks():
        line(73 * '*')

    def dashes():
        line(73 * '-')

    def format_values(space_count, *values):
        return (space_count * ' ').join(map(str, values))

    def key_value_line(key, *values):
        line(f"{key:<30}: {format_values(1, *values)}")

    def list_item_line(item, *values):
        line(f"  - {item:<26}: {format_values(1, *values)}")

    def header_line(header):
        line(f"{header}:")

    def table_header_setup(*headers):
        return list(headers), [len(header) for header in headers]

    def table_header(*headers):
        line(format_values(2, *headers))

    def table_line(header_lengths, *values):
        string = "  ".join("{:<" + str(length) + "}" for length in header_lengths)
        line(string.format(*values))

    def table_line_with_last_array(header_lengths, *values):
        vals = list(values)
        string = "  ".join("{:<" + str(length) + "}" for length in header_lengths[:-1])
        line(string.format(*vals) + "  " + format_values(1, *vals[-1]))  # format last array

    asterisks()
    key_value_line("file with basedata", "None")
    key_value_line("initial value random generator", 0)

    asterisks()
    key_value_line("projects", len(instance.projects))
    key_value_line("jobs (incl. supersource/sink )", len(instance.jobs))
    key_value_line("horizon", instance.horizon)
    line("RESOURCES")
    list_item_line("renewable", sum(1 for r in instance.resources if r.type == ResourceType.RENEWABLE), 'R')
    list_item_line("nonrenewable", sum(1 for r in instance.resources if r.type == ResourceType.NONRENEWABLE), 'N')
    list_item_line("doubly constrained", sum(1 for r in instance.resources if r.type == ResourceType.DOUBLY_CONSTRAINED), 'D')

    asterisks()
    header_line("PROJECT INFORMATION")
    table_headers, lengths = table_header_setup("pronr.", "#jobs", "rel.date", "duedate", "tardcost", "MPM-TIME")
    table_header(*table_headers)
    for p in instance.projects:
        table_line(lengths,
                   p.id_project,
                   len(instance.jobs) - 2,  # Assume all jobs included except the super-source/sink
                   0,  # Assume all released
                   p.due_date,
                   p.tardiness_cost,
                   0)  # Assume no mpm-time set

    asterisks()
    header_line("PRECEDENCE RELATIONS")
    # Compute job successors
    job_successors = defaultdict(list)
    for precedence in instance.precedences:
        job_successors[precedence.id_child].append(precedence.id_parent)
    # Print job table
    table_headers, lengths = table_header_setup("jobnr.", "#modes", "#successors", "successors")
    table_header(*table_headers)
    for j in instance.jobs:
        table_line_with_last_array(lengths,
                                   j.id_job,
                                   1,  # Assume only one mode
                                   len(job_successors[j.id_job]),  # Successor count
                                   job_successors[j.id_job])  # Successors

    asterisks()
    header_line("REQUESTS/DURATIONS")
    resources_with_strs = {resource: f"{resource.type.title()} {resource.id_resource}"
                           for resource in instance.resources}
    table_headers, lengths = table_header_setup("jobnr.", "mode", "duration", *resources_with_strs.values())
    table_header(*table_headers)
    dashes()
    for job in instance.jobs:
        consumptions = [job.resource_consumption.consumption_by_resource[r] for r in resources_with_strs]
        table_line(lengths,
                   job.id_job,
                   1,  # Assume only one mode
                   job.resource_consumption.duration,
                   *consumptions)

    asterisks()
    header_line("RESOURCEAVAILABILITIES")
    table_headers, lengths = table_header_setup(*resources_with_strs.values())
    table_header(*table_headers)
    table_line(lengths, *(r.capacity for r in resources_with_strs))

    asterisks()

    if not is_extended:
        return output

    asterisks()
    header_line("DUE DATES")
    for job in instance.jobs:
        line(format_values(2, job.id_job, job.due_date))

    asterisks()
    header_line("FINISHED_TASKS")
    line(format_values(1, *(job.id_job for job in instance.jobs if job.completed)))

    asterisks()
    header_line("COMPONENTS")
    line(format_values(1, *(c.id_root_job for c in instance.components)))

    asterisks()
    header_line("COMPONENT_WEIGHTS")
    line(format_values(1, *(c.weight for c in instance.components)))

    asterisks()
    header_line("RESOURCE SHIFT MODES")
    for resource in instance.resources:
        line(format_values(2, resource.id_resource, resource.shift_mode))

    return output


def __check_json_parse_object(obj: dict) -> None:
    def check_key_in(key: str, d: dict, d_name: str) -> None:
        if key not in d:
            raise ParseError.in_data(f"{key} not in {d_name}")

    def check_type_of(o: Any, expected: type, o_name: str) -> None:
        if not isinstance(o, expected):
            raise ParseError.in_data(f"{o_name} should be {expected} but is {type(o)}")

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise ParseError.in_data(message)

    check_type_of(obj, dict, "Instance object")

    check_key_in("Name", obj, "Instance object")
    check_type_of(obj["Name"], str, "Instance object > Name")

    check_key_in("Horizon", obj, "Instance object")
    check_type_of(obj["Horizon"], int, "Instance object > Horizon")

    check_key_in("Resources", obj, "Instance object")
    check_type_of(obj["Resources"], list, "Instance object > Resources")

    check_key_in("Projects", obj, "Instance object")
    check_type_of(obj["Projects"], list, "Instance object > Projects")

    check_key_in("Jobs", obj, "Instance object")
    check_type_of(obj["Jobs"], list, "Instance object > Jobs")

    for r in obj["Resources"]:
        check_key_in("Type", r, "Instance object > Resources > Type")
        check_type_of(r["Type"], str, "Instance object > Resources > Type")
        check(len(r["Type"]) == 1, "Invalid resource type string length")

        check_key_in("Id", r, "Instance object > Resources > Id")
        check_type_of(r["Id"], int, "Instance object > Resources > Id")

        check_key_in("Capacity", r, "Instance object > Resources > Capacity")
        check_type_of(r["Capacity"], int, "Instance object > Resources > Capacity")

    for p in obj["Projects"]:
        check_key_in("Id", p, "Instance object > Projects > Id")
        check_type_of(p["Id"], int, "Instance object > Projects > Id")

        check_key_in("Due date", p, "Instance object > Projects > Due date")
        check_type_of(p["Due date"], int, "Instance object > Projects > Due date")

        check_key_in("Tardiness cost", p, "Instance object > Projects > Tardiness cost")
        check_type_of(p["Tardiness cost"], int, "Instance object > Projects > Tardiness cost")

    for j in obj["Jobs"]:
        check_key_in("Id", j, "Instance object > Jobs > Id")
        check_type_of(j["Id"], int, "Instance object > Jobs > Id")

        check_key_in("Resource consumption", j, "Instance object > Jobs > Resource consumption")
        check_type_of(j["Resource consumption"], dict, "Instance object > Jobs > Resource consumption")
        check_key_in("Duration", j["Resource consumption"], "Instance object > Jobs > Resource consumption > Duration")
        check_type_of(j["Resource consumption"]["Duration"], int, "Instance object > Jobs > Resource consumption > Duration")
        check_key_in("Consumptions", j["Resource consumption"], "Instance object > Jobs > Resource consumption > Consumptions")
        check_type_of(j["Resource consumption"]["Consumptions"], dict, "Instance object > Jobs > Resource consumption > Consumptions")
        for rc_key, rc_size in j["Resource consumption"]["Consumptions"].items():
            check_type_of(rc_key, str, "Instance object > Jobs > Resource consumption > Consumptions > Resource Key")
            check_type_of(rc_size, int, "Instance object > Jobs > Resource consumption > Consumptions > Size")

        check_key_in("Successors", j, "Instance object > Jobs > Successors")
        check_type_of(j["Successors"], list, "Instance object > Jobs > Successors")


class ParseError(Exception):
    def __init__(self, message):
        super().__init__(message)

    @staticmethod
    def in_file(file: IO,
                line_num: int,
                message: str):
        return ParseError(f"[{file.name}:{line_num}] {message}")

    @staticmethod
    def in_data(message):
        return ParseError(message)


class ProblemInstanceJSONSerializer(json.JSONEncoder):
    def default(self, obj: any) -> any:
        if isinstance(obj, ProblemInstance):
            precedences_by_child = defaultdict(list)
            for precedence in obj.precedences:
                precedences_by_child[precedence.id_child].append(precedence.id_parent)
            return {
                "Name": obj.name,
                "Horizon": obj.horizon,
                "Resources": [ProblemInstanceJSONSerializer.__serialize_resource(resource) for resource in obj.resources],
                "Projects": [ProblemInstanceJSONSerializer.__serialize_project(project) for project in obj.projects],
                "Jobs": [ProblemInstanceJSONSerializer.__serialize_job(job, precedences_by_child[job.id_job]) for job in obj.jobs],
            }
        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def __serialize_resource(resource: Resource) -> dict[str, any]:
        return {
            "Type": resource.type,
            "Id": resource.id_resource,
            "Capacity": resource.capacity
        }

    @staticmethod
    def __serialize_resource_consumption(resource_consumption: ResourceConsumption) -> dict[str, any]:
        return {
            "Duration": resource_consumption.duration,
            "Consumptions": {resource.key: size for resource, size in resource_consumption.consumption_by_resource.items()}
        }

    @staticmethod
    def __serialize_job(job: Job, successors: list[int]) -> dict[str, any]:
        return {
            "Id": job.id_job,
            "Resource consumption": ProblemInstanceJSONSerializer.__serialize_resource_consumption(job.resource_consumption),
            "Successors": successors,
        }

    @staticmethod
    def __serialize_project(project: Project) -> dict[str, any]:
        return {
            "Id": project.id_project,
            "Due date": project.due_date,
            "Tardiness cost": project.tardiness_cost
        }
