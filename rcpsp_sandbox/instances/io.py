import json
import os
from typing import IO, Iterable, Any
from collections import defaultdict

from instances.problem_instance import ProblemInstance, Project, Job, Precedence, Resource, ResourceConsumption, \
    ResourceType, Component, AvailabilityInterval, ResourceAvailability, CapacityChange, CapacityMigration
from instances.instance_builder import InstanceBuilder
from utils import modify_tuple, try_open_read, chunk, str_or_default

PSPLIB_KEY_VALUE_SEPARATOR: str = ':'


def parse_psplib(filename: str,
                 name_as: str or None = None,
                 is_extended: bool = False) -> ProblemInstance:
    """
    Parses a PSPLIB file and returns a ProblemInstance object.

    Args:
        filename (str): The path to the PSPLIB file.
        name_as (str or None, optional): The name to assign to the ProblemInstance object. Defaults to None.
        is_extended (bool, optional): Whether the PSPLIB file is in extended format. Defaults to False.

    Returns:
        ProblemInstance: The parsed ProblemInstance object.
    """
    return try_open_read(filename, __parse_psplib_internal, name_as=name_as, is_extended=is_extended)


def serialize_psplib(instance: ProblemInstance,
                     filename: str,
                     is_extended: bool = False) -> None:
    """
    Serializes a problem instance into a PSPLIB file.

    Args:
        instance (ProblemInstance): The ProblemInstance object to serialize.
        filename (str): The name of the file to write the serialized data to.
        is_extended (bool, optional): Whether to include extended instance information in the serialization.
            Defaults to False.
    """
    psplib_str = __serialize_psplib_internal(instance, is_extended)
    with open(filename, "wt") as file:
        file.write(psplib_str)


def parse_json(filename: str,
               name_as: str | None = None,
               is_extended: bool = False) -> ProblemInstance:
    """
    Parses a problem instance serialized in JSON.

    Args:
        filename (str): The path to the JSON file.
        name_as (str or None, optional): The name to assign to the ProblemInstance. If None, the name is extracted from the JSON file. Defaults to None.
        is_extended (bool, optional): Whether the JSON file contains an extended problem instance. Defaults to False.

    Returns:
        ProblemInstance: The constructed problem instance.
    """
    instance_object = try_open_read(filename, json.load)

    __check_json_parse_object(instance_object, is_extended)

    instance_builder = InstanceBuilder()

    # instance properties
    instance_builder.set(name=name_as if name_as is not None else instance_object["Name"],
                         target_job=instance_object["TargetJob"],
                         horizon=instance_object["Horizon"]
                         )

    # resources
    resources = [Resource(r["Id"], ResourceType(r["Type"]), r["Capacity"]) for r in instance_object["Resources"]]
    resources_by_key = {r.key: r for r in resources}
    instance_builder.add_resources(resources)

    # projects
    instance_builder.add_projects(Project(p["Id"], p["Due date"], p["Tardiness cost"])
                                  for p in instance_object["Projects"])

    # jobs
    jobs = [Job(j["Id"],
                j["Duration"],
                ResourceConsumption({resources_by_key[key]: consumption
                                     for key, consumption in j["Resource consumption"]["Consumptions"].items()}))
            for j in instance_object["Jobs"]]
    instance_builder.add_jobs(jobs)

    # precedences - serialized as successors of each job, need to be extracted from job data
    instance_builder.add_precedences(Precedence(j["Id"], successor)
                                     for j in instance_object["Jobs"]  # child
                                     for successor in j["Successors"])  # successor parent of child

    if is_extended:
        for i, r in enumerate(instance_object["Resources"]):
            availability_periodical = [AvailabilityInterval(start=av["Start"], end=av["End"], capacity=(av["Capacity"] if "Capacity" in av else r["Capacity"]))
                                       for av in r["Availability"]["Periodical"]]
            availability_additions = [CapacityChange(start=av["Start"], end=av["End"], capacity=av["Capacity"])
                                      for av in r["Availability"]["Additions"]]
            availability_migrations = [CapacityMigration(resource_to=av["ResourceTo"], start=av["Start"], end=av["End"], capacity=av["Capacity"])
                                       for av in r["Availability"]["Migrations"]]

            resources[i].availability = ResourceAvailability(availability_periodical, availability_additions, availability_migrations)

        for i, j in enumerate(instance_object["Jobs"]):
            jobs[i].completed = j["Completed"]
            jobs[i].due_date = j["Due date"]

        components = [Component(c["Root job"], c["Weight"]) for c in instance_object["Components"]]
        instance_builder.add_components(components)

    return instance_builder.build_instance()


def serialize_json(instance: ProblemInstance,
                   filename: str,
                   is_extended: bool = False) -> None:
    """
    Serializes a problem instance in JSON format.

    Args:
        instance (ProblemInstance): The ProblemInstance object to serialize.
        filename (str): The name of the file to write the serialized data to.
        is_extended (bool, optional): Whether to include extended instance information in the serialization.
            Defaults to False.
    """
    json_str = json.dumps(instance, cls=ProblemInstanceJSONSerializer, is_extended=is_extended)
    with open(filename, "wt") as file:
        file.write(json_str)


def __parse_psplib_internal(file: IO,
                            name_as: str | None,
                            is_extended: bool) -> ProblemInstance:
    """
    Parses a PSPLIB file and returns a ProblemInstance object.

    This function is... long. It reads the file line by line, parsing the data into a ProblemInstance object.
    Should a brave soul venture into this function, they will find a series of helper functions that
    make the parsing process more manageable. It follows the structure of the PSPLIB file format.
    The final steps consider the extended format, which includes additional information about the instance.
    """
    def read_line():
        nonlocal line_num
        line_num += 1
        return file.readline()

    def skip_lines(count: int) -> None:
        nonlocal line_num
        for _ in range(count):
            read_line()

    def asterisks() -> None:
        skip_lines(1)

    def process_split_line(line,
                           target_indices: tuple,
                           end_of_line_array_index: int = -1) -> tuple:
        content = line.split(maxsplit=end_of_line_array_index)
        if (len(content) - 1) < end_of_line_array_index:  # If the end-of-line array is empty...
            content.append("")  # ...insert empty array string
        elif (len(content) - 1) < max(target_indices):
            raise ParseError.in_file(file, line_num, "Line contains less values than expected")

        return tuple(content[i] for i in target_indices)

    def parse_split_line(target_indices: tuple,
                         end_of_line_array_index: int = -1,
                         move_line: bool = True) -> tuple:
        nonlocal line_num

        line = file.readline()
        result = process_split_line(line, target_indices, end_of_line_array_index)
        if move_line:
            line_num += 1

        return result

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
            builder.set(target_job=target_job)

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
    if is_extended:
        target_job = parse_colon_key_value_line("target job")
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
                           duration,
                           ResourceConsumption({resource: consumption_by_resource_id[resource.id_resource]
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
    resource_availability_line = file.readline()  # RESOURCE SHIFT MODES    or    RESOURCE AVAILABILITIES
    line_num += 1

    if resource_availability_line.startswith("RESOURCE SHIFT MODES"):
        def availability_interval_from_shift_mode(resource_shift_mode: int):
            match resource_shift_mode:
                case 1:
                    return [AvailabilityInterval(start=8, end=16, capacity=1)]
                case 2:
                    return [AvailabilityInterval(start=6, end=22, capacity=1)]
                case _:
                    raise ParseError.in_file(file, line_num, "Unexpected resource shift mode")

        resource_by_id = {r.id_resource: r for r in resources}
        for _ in range(len(resources)):
            id_resource_str, shift_mode_str = parse_split_line((0, 1))
            id_resource = try_parse_value(id_resource_str)
            availability = ResourceAvailability(availability_interval_from_shift_mode(try_parse_value(shift_mode_str)))
            resource_by_id[id_resource].availability = availability

    else:  # RESOURCE AVAILABILITIES
        skip_lines(3)  # table header and dashes and "PERIODICAL"
        periodical_intervals_by_resource_key = defaultdict(list)
        addition_intervals_by_resource_key = defaultdict(list)
        migration_intervals_by_resource_key = defaultdict(list)
        current_line = read_line()
        resource_key = ""
        while not current_line.startswith('ADDITIONS'):
            if not current_line.startswith(' '):  # New resource
                resource_key, start_str, end_str, capacity_str = process_split_line(current_line, (0, 1, 2, 3), 3)
            else:
                start_str, end_str, capacity_str = process_split_line(current_line, (0, 1, 2), 2)
            start, end = try_parse_value(start_str), try_parse_value(end_str)
            capacity = try_parse_value(capacity_str.strip()) if capacity_str != "" else None
            periodical_intervals_by_resource_key[resource_key].append(AvailabilityInterval(start, end, capacity))

            current_line = read_line()
        current_line = read_line()
        while not current_line.startswith('MIGRATIONS'):
            if not current_line.startswith(' '):  # New resource
                resource_key, start_str, end_str, capacity_str = process_split_line(current_line, (0, 1, 2, 3), 3)
            else:
                start_str, end_str, capacity_str = process_split_line(current_line, (0, 1, 2), 2)
            start, end = try_parse_value(start_str), try_parse_value(end_str)
            capacity = try_parse_value(capacity_str.strip()) if capacity_str != "" else None
            addition_intervals_by_resource_key[resource_key].append(CapacityChange(start, end, capacity))

            current_line = read_line()
        current_line = read_line()
        while (not current_line.startswith('*')) and (current_line != ""):
            if not current_line.startswith(' '):  # New resource
                resource_migration_key, start_str, end_str, capacity_str = process_split_line(current_line, (0, 1, 2, 3), 3)
            else:
                start_str, end_str, capacity_str = process_split_line(current_line, (0, 1, 2), 2)
            resource_from_key, resource_to_key = resource_migration_key.split('->')
            start, end = try_parse_value(start_str), try_parse_value(end_str)
            capacity = try_parse_value(capacity_str.strip()) if capacity_str != "" else None
            migration_intervals_by_resource_key[resource_from_key].append(CapacityMigration(resource_to_key, start, end, capacity))

            current_line = read_line()

        for resource in resources:
            resource.availability = ResourceAvailability(periodical_intervals_by_resource_key[resource.key],
                                                         addition_intervals_by_resource_key[resource.key],
                                                         migration_intervals_by_resource_key[resource.key])
            for i, availability_interval in enumerate(resource.availability.periodical_intervals):
                if availability_interval.capacity is None:
                    resource.availability.periodical_intervals[i] = modify_tuple(availability_interval, 2, resource.capacity)

    return build()


def __serialize_psplib_internal(instance: ProblemInstance, is_extended: bool) -> str:
    """
    Serializes a problem instance into a PSPLIB file.

    This function is... long. It constructs the PSPLIB file format from the ProblemInstance object.
    Should a brave soul venture into this function, they will find a series of helper functions that
    make the serialization process more manageable. It follows the structure of the PSPLIB file format.
    The final steps consider the extended format, which includes additional information about the instance.
    """
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
    if is_extended:
        key_value_line("target job", instance.target_job)
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
        consumptions = [job.resource_consumption[r] for r in resources_with_strs]
        table_line(lengths,
                   job.id_job,
                   1,  # Assume only one mode
                   job.duration,
                   *consumptions)

    asterisks()
    header_line("RESOURCEAVAILABILITIES")
    table_headers, lengths = table_header_setup(*resources_with_strs.values())
    table_header(*table_headers)
    table_line(lengths, *(r.capacity for r in resources_with_strs))

    asterisks()

    if not is_extended:
        return output

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
    header_line("RESOURCE AVAILABILITIES")
    table_headers, lengths = table_header_setup("resource", "start", "end", "(capacity)")
    table_header(*table_headers)
    dashes()
    header_line("PERIODICAL")
    for resource in instance.resources:
        first_availability: AvailabilityInterval = resource.availability.periodical_intervals[0]
        table_line(lengths,
                   resource.key,
                   first_availability.start,
                   first_availability.end,
                   str_or_default(first_availability.capacity))
        for availability in resource.availability.periodical_intervals[1:]:
            table_line(lengths,
                       "",
                       availability.start,
                       availability.end,
                       str_or_default(availability.capacity))
    header_line("ADDITIONS")
    for resource in instance.resources:
        if not resource.availability.additions:
            continue
        first_availability: CapacityChange = resource.availability.additions[0]
        table_line(lengths,
                   resource.key,
                   first_availability.start,
                   first_availability.end,
                   str_or_default(first_availability.capacity))
        for availability in resource.availability.additions[1:]:
            table_line(lengths,
                       "",
                       availability.start,
                       availability.end,
                       str_or_default(availability.capacity))
    header_line("MIGRATIONS")
    for resource in instance.resources:
        if not resource.availability.migrations:
            continue
        for migration in resource.availability.migrations:
            table_line(lengths,
                       f'{resource.key}->{migration.resource_to}',
                       migration.start,
                       migration.end,
                       str_or_default(migration.capacity))

    return output


def __check_json_parse_object(obj: dict, is_extended: bool) -> None:
    """
    Checks if the parsed JSON object is a valid instance object.
    """
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

    check_key_in("TargetJob", obj, "Instance object")
    check_type_of(obj["TargetJob"], int, "Instance object > TargetJob")

    check_key_in("Resources", obj, "Instance object")
    check_type_of(obj["Resources"], list, "Instance object > Resources")

    check_key_in("Projects", obj, "Instance object")
    check_type_of(obj["Projects"], list, "Instance object > Projects")

    check_key_in("Jobs", obj, "Instance object")
    check_type_of(obj["Jobs"], list, "Instance object > Jobs")

    for r in obj["Resources"]:
        check_type_of(r, dict, "Instance object > Jobs")

        check_key_in("Type", r, "Instance object > Resources")
        check_type_of(r["Type"], str, "Instance object > Resources > Type")
        check(len(r["Type"]) == 1, "Invalid resource type string length")

        check_key_in("Id", r, "Instance object > Resources")
        check_type_of(r["Id"], int, "Instance object > Resources > Id")

        check_key_in("Capacity", r, "Instance object > Resources")
        check_type_of(r["Capacity"], int, "Instance object > Resources > Capacity")

    for p in obj["Projects"]:
        check_type_of(p, dict, "Instance object > Jobs")

        check_key_in("Id", p, "Instance object > Projects")
        check_type_of(p["Id"], int, "Instance object > Projects > Id")

        check_key_in("Due date", p, "Instance object > Projects")
        check_type_of(p["Due date"], int, "Instance object > Projects > Due date")

        check_key_in("Tardiness cost", p, "Instance object > Projects")
        check_type_of(p["Tardiness cost"], int, "Instance object > Projects > Tardiness cost")

    for j in obj["Jobs"]:
        check_type_of(j, dict, "Instance object > Jobs")

        check_key_in("Id", j, "Instance object > Jobs")
        check_type_of(j["Id"], int, "Instance object > Jobs > Id")

        check_key_in("Duration", j, "Instance object > Jobs")
        check_type_of(j["Duration"], int, "Instance object > Jobs > Duration")

        check_key_in("Resource consumption", j, "Instance object > Jobs")
        check_type_of(j["Resource consumption"], dict, "Instance object > Jobs > Resource consumption")
        check_key_in("Consumptions", j["Resource consumption"], "Instance object > Jobs > Resource consumption")
        check_type_of(j["Resource consumption"]["Consumptions"], dict, "Instance object > Jobs > Resource consumption > Consumptions")
        for rc_key, rc_size in j["Resource consumption"]["Consumptions"].items():
            check_type_of(rc_key, str, "Instance object > Jobs > Resource consumption > Consumptions > Resource Key")
            check_type_of(rc_size, int, "Instance object > Jobs > Resource consumption > Consumptions > Size")

        check_key_in("Successors", j, "Instance object > Jobs")
        check_type_of(j["Successors"], list, "Instance object > Jobs > Successors")

    if not is_extended:
        return

    check_key_in("Components", obj, "Instance object")
    check_type_of(obj["Components"], list, "Instance object > Components")

    for c in obj["Components"]:
        check_type_of(c, dict, "Instance object > Components")

        check_key_in("Root job", c, "Instance object > Components")
        check_type_of(c["Root job"], int, "Instance object > Components > Root job")

        check_key_in("Weight", c, "Instance object > Components")
        check_type_of(c["Weight"], int, "Instance object > Components > Weight")

    for j in obj["Jobs"]:
        check_key_in("Due date", j, "Instance object > Jobs")
        check_type_of(j["Due date"], int, "Instance object > Jobs > Due date")

        check_key_in("Completed", j, "Instance object > Jobs")
        check_type_of(j["Completed"], bool, "Instance object > Jobs > Completed")

    for r in obj["Resources"]:
        check_key_in("Availability", r, "Instance object > Resources")
        check_type_of(r["Availability"], dict, "Instance object > Resources > Availability")

        check_key_in("Periodical", r["Availability"], "Instance object > Resource > Availability")
        check_type_of(r["Availability"]["Periodical"], list, "Instance object > Resource > Availability > Periodical")
        check_key_in("Additions", r["Availability"], "Instance object > Resource > Availability")
        check_type_of(r["Availability"]["Additions"], list, "Instance object > Resource > Availability > Additions")
        check_key_in("Migrations", r["Availability"], "Instance object > Resource > Availability")
        check_type_of(r["Availability"]["Migrations"], list, "Instance object > Resource > Availability > Migrations")

        for periodical in r["Availability"]["Periodical"]:
            check_key_in("Start", periodical, "Instance object > Resources > Availability > Periodical")
            check_type_of(periodical["Start"], int, "Instance object > Resources > Availability > Periodical > Start")

            check_key_in("End", periodical, "Instance object > Resources > Availability > Periodical")
            check_type_of(periodical["End"], int, "Instance object > Resources > Availability > Periodical > End")

            if "Capacity" in periodical:
                check_type_of(periodical["Capacity"], int, "Instance object > Resources > Availability > Periodical > Capacity")

        for addition in r["Availability"]["Additions"]:
            check_key_in("Start", addition, "Instance object > Resources > Availability > Addition")
            check_type_of(addition["Start"], int, "Instance object > Resources > Availability > Addition > Start")

            check_key_in("End", addition, "Instance object > Resources > Availability > Addition")
            check_type_of(addition["End"], int, "Instance object > Resources > Availability > Addition > End")

            check_key_in("Capacity", addition, "Instance object > Resources > Availability > Addition")
            check_type_of(addition["Capacity"], int, "Instance object > Resources > Availability > Addition > Capacity")

        for migration in r["Availability"]["Migrations"]:
            check_key_in("ResourceTo", migration, "Instance object > Resources > Availability > Migration")
            check_type_of(migration["ResourceTo"], str, "Instance object > Resources > Availability > Migration > ResourceTo")

            check_key_in("Start", migration, "Instance object > Resources > Availability > Migration")
            check_type_of(migration["Start"], int, "Instance object > Resources > Availability > Migration > Start")

            check_key_in("End", migration, "Instance object > Resources > Availability > Migration")
            check_type_of(migration["End"], int, "Instance object > Resources > Availability > Migration > End")

            check_key_in("Capacity", migration, "Instance object > Resources > Availability > Migration")
            check_type_of(migration["Capacity"], int, "Instance object > Resources > Availability > Migration > Capacity")


class ParseError(Exception):
    """
    Exception raised for parsing errors.
    """

    def __init__(self, message):
        super().__init__(message)

    @staticmethod
    def in_file(file: IO,
                line_num: int,
                message: str):
        """
        Create a ParseError for an error in a file.

        Args:
            file (IO): The file object where the error occurred.
            line_num (int): The line number where the error occurred.
            message (str): The error message.

        Returns:
            ParseError: The ParseError object.
        """
        return ParseError(f"[{file.name}:{line_num}] {message}")

    @staticmethod
    def in_data(message):
        """
        Create a ParseError for an error in data.

        Args:
            message (str): The error message.

        Returns:
            ParseError: The ParseError object.
        """
        return ParseError(message)


class ProblemInstanceJSONSerializer(json.JSONEncoder):
    """
    Serializes a ProblemInstance object to JSON format.
    """

    _is_extended: bool

    def __init__(self, is_extended: bool, **kwargs):
        """
        Initializes a ProblemInstanceJSONSerializer object.

        Args:
            is_extended (bool): Indicates whether to include extended information in the serialization.
            **kwargs: Additional keyword arguments to be passed to the base class constructor.
        """
        super().__init__(**kwargs)

        self._is_extended = is_extended

    def default(self, obj: any) -> any:
        """
        Overrides the default method of the base class to handle serialization of ProblemInstance objects.

        Args:
            obj (any): The object to be serialized.

        Returns:
            any: The serialized object.
        """
        if isinstance(obj, ProblemInstance):
            precedences_by_child = defaultdict(list)
            for precedence in obj.precedences:
                precedences_by_child[precedence.id_child].append(precedence.id_parent)

            instance_object = {
                "Name": obj.name,
                "Horizon": obj.horizon,
                "TargetJob": obj.target_job,
                "Resources": [self.__serialize_resource(resource) for resource in obj.resources],
                "Projects": [ProblemInstanceJSONSerializer.__serialize_project(project) for project in obj.projects],
                "Jobs": [self.__serialize_job(job, precedences_by_child[job.id_job]) for job in obj.jobs],
            }

            if self._is_extended:
                instance_object["Components"] = [ProblemInstanceJSONSerializer.__serialize_component(component) for component in obj.components]

            return instance_object
        return json.JSONEncoder.default(self, obj)

    def __serialize_resource(self, resource: Resource) -> dict[str, any]:
        """
        Serializes a Resource object to a dictionary.

        Args:
            resource (Resource): The Resource object to be serialized.

        Returns:
            dict[str, any]: The serialized resource as a dictionary.
        """
        resource_object = {
            "Type": resource.type,
            "Id": resource.id_resource,
            "Capacity": resource.capacity
        }

        if self._is_extended:
            def serialize_interval(interval: AvailabilityInterval|CapacityChange):
                return {
                    "Start": interval.start,
                    "End": interval.end,
                    "Capacity": interval.capacity,
                }

            def serialize_migration(migration: CapacityMigration):
                obj = serialize_interval(migration)
                obj["ResourceTo"] = migration.resource_to
                return obj

            resource_object["Availability"] = {
                "Periodical": [serialize_interval(interval) for interval in resource.availability.periodical_intervals],
                "Additions": [serialize_interval(addition) for addition in resource.availability.additions],
                "Migrations": [serialize_migration(migration) for migration in resource.availability.migrations]
            }

        return resource_object

    @staticmethod
    def __serialize_resource_consumption(resource_consumption: ResourceConsumption) -> dict[str, any]:
        """
        Serializes a ResourceConsumption object to a dictionary.

        Args:
            resource_consumption (ResourceConsumption): The ResourceConsumption object to be serialized.

        Returns:
            dict[str, any]: The serialized resource consumption as a dictionary.
        """
        return {
            "Consumptions": {resource.key: size for resource, size in resource_consumption.consumption_by_resource.items()}
        }

    def __serialize_job(self, job: Job, successors: list[int]) -> dict[str, any]:
        """
        Serializes a Job object to a dictionary.

        Args:
            job (Job): The Job object to be serialized.
            successors (list[int]): The list of successor job IDs.

        Returns:
            dict[str, any]: The serialized job as a dictionary.
        """
        job_object = {
            "Id": job.id_job,
            "Duration": job.duration,
            "Resource consumption": ProblemInstanceJSONSerializer.__serialize_resource_consumption(job.resource_consumption),
            "Successors": successors,
        }

        if self._is_extended:
            job_object["Due date"] = job.due_date
            job_object["Completed"] = job.completed

        return job_object

    @staticmethod
    def __serialize_project(project: Project) -> dict[str, any]:
        """
        Serializes a Project object to a dictionary.

        Args:
            project (Project): The Project object to be serialized.

        Returns:
            dict[str, any]: The serialized project as a dictionary.
        """
        return {
            "Id": project.id_project,
            "Due date": project.due_date,
            "Tardiness cost": project.tardiness_cost
        }

    @staticmethod
    def __serialize_component(component: Component) -> dict[str, any]:
        """
        Serializes a Component object to a dictionary.

        Args:
            component (Component): The Component object to be serialized.

        Returns:
            dict[str, any]: The serialized component as a dictionary.
        """
        return {
            "Root job": component.id_root_job,
            "Weight": component.weight
        }
