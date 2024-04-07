import itertools
import math
from collections import defaultdict, namedtuple
from enum import StrEnum
from typing import Optional, Collection, Self, Iterable

from instances.utils import list_of
from utils import index_groups, T_StepFunction, interval_overlap_function


# ~~~~~~~ ResourceType ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResourceType(StrEnum):
    RENEWABLE = 'R'
    NONRENEWABLE = 'N'
    DOUBLY_CONSTRAINED = 'D'


# ~~~~~~~ ResourceAvailability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AvailabilityInterval = namedtuple("AvailabilityInterval", ("start", 'end', "capacity"))
CapacityChange = namedtuple("CapacityChange", ("start", "end", "capacity"))
CapacityMigration = namedtuple("CapacityMigration", ("resource_to", "start", "end", "capacity"))


class ResourceAvailability:
    _periodical_intervals: list[AvailabilityInterval]
    _additions: list[CapacityChange]
    _migrations: list[CapacityMigration]

    def __init__(self,
                 periodical_intervals: Iterable[AvailabilityInterval],
                 additions: Iterable[CapacityChange] = None,
                 migrations: Iterable[CapacityMigration] = None,
                 ):
        self._periodical_intervals = list_of(periodical_intervals)
        self._additions = list_of(additions) if additions is not None else []
        self._migrations = list_of(migrations) if migrations is not None else []

    @property
    def periodical_intervals(self) -> list[AvailabilityInterval]:
        return self._periodical_intervals

    @periodical_intervals.setter
    def periodical_intervals(self, value: Iterable[AvailabilityInterval]):
        self._periodical_intervals = list_of(value)

    @property
    def additions(self) -> list[CapacityChange]:
        return self._additions

    @additions.setter
    def additions(self, value: Iterable[CapacityChange]):
        self._additions = list_of(value)

    @property
    def migrations(self) -> list[CapacityMigration]:
        return self._migrations

    @migrations.setter
    def migrations(self, value: Iterable[CapacityMigration]):
        self._migrations = list_of(value)

    def copy(self) -> "ResourceAvailability":
        return ResourceAvailability(periodical_intervals=self.periodical_intervals[:],
                                    additions=self._additions[:],
                                    migrations=self._migrations[:],
                                    )


# ~~~~~~~ Resource ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Resource:
    _id_resource: int
    _type: ResourceType
    _capacity: int
    _availability: ResourceAvailability

    def __init__(self,
                 id_resource: int,
                 resource_type: ResourceType,
                 capacity: int,
                 availability: ResourceAvailability = None):
        self._id_resource = id_resource
        self._type = resource_type
        self._capacity = capacity
        self._availability = availability if availability is not None else ResourceAvailability([AvailabilityInterval(0, 24, capacity)])

    @property
    def id_resource(self) -> int:
        return self._id_resource

    @id_resource.setter
    def id_resource(self, value: int):
        self._id_resource = value

    @property
    def type(self) -> ResourceType:
        return self._type

    @type.setter
    def type(self, value: ResourceType):
        self._type = value

    @property
    def capacity(self) -> int:
        return self._capacity

    @capacity.setter
    def capacity(self, value: int):
        self._capacity = value

    @property
    def availability(self) -> ResourceAvailability:
        return self._availability

    @availability.setter
    def availability(self, value: ResourceAvailability):
        self._availability = value

    @property
    def key(self) -> str:
        return f"{self.type}{self.id_resource}"

    def copy(self) -> Self:
        return Resource(id_resource=self.id_resource,
                        resource_type=self.type,
                        capacity=self.capacity,
                        availability=self.availability.copy() if self.availability is not None else None)

    def __hash__(self):
        return hash((self._id_resource, self._type))

    def __eq__(self, other):
        return isinstance(other, Resource)\
            and self.id_resource == other.id_resource \
            and self.type == other.type

    def __str__(self):
        return f"Resource{{id: {self.id_resource}, type: {self.type}}}"


# ~~~~~~~ ResourceConsumption ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResourceConsumption:
    _consumption_by_resource: dict[Resource, int]

    def __init__(self,
                 consumption_by_resource: dict[Resource, int]):
        self._consumption_by_resource = consumption_by_resource

    @property
    def consumption_by_resource(self) -> dict[Resource, int]:
        return self._consumption_by_resource

    @consumption_by_resource.setter
    def consumption_by_resource(self, value: dict[Resource, int]):
        self._consumption_by_resource = value

    def copy(self) -> Self:
        return ResourceConsumption(consumption_by_resource=self.consumption_by_resource.copy())

    def __getitem__(self, resource: Resource or int):
        return self.consumption_by_resource[resource]

    def __str__(self):
        return f"ResourceConsumption{{consumptions: {self.consumption_by_resource}}}"


# ~~~~~~~ Job ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Job:
    _id_job: int
    _duration: int
    _resource_consumption: ResourceConsumption
    _due_date: int or None
    _completed: bool

    def __init__(self,
                 id_job: int,
                 duration: int,
                 resource_consumption: ResourceConsumption,
                 due_date: int = 0,
                 completed: bool = False):
        self._id_job = id_job
        self._duration = duration
        self._resource_consumption = resource_consumption
        self._due_date = due_date
        self._completed = completed

    @property
    def id_job(self) -> int:
        return self._id_job

    @id_job.setter
    def id_job(self, value: int):
        self._id_job = value

    @property
    def duration(self) -> int:
        return self._duration

    @duration.setter
    def duration(self, value: int):
        self._duration = value

    @property
    def resource_consumption(self) -> ResourceConsumption:
        return self._resource_consumption

    @resource_consumption.setter
    def resource_consumption(self, value: ResourceConsumption):
        self._resource_consumption = value

    @property
    def due_date(self) -> int or None:
        return self._due_date

    @due_date.setter
    def due_date(self, value: int):
        self._due_date = value

    @property
    def completed(self) -> bool:
        return self._completed

    @completed.setter
    def completed(self, value: bool):
        self._completed = value

    def copy(self) -> Self:
        return Job(id_job=self.id_job,
                   duration=self.duration,
                   resource_consumption=self.resource_consumption.copy(),
                   due_date=self.due_date,
                   completed=self.completed)

    def __hash__(self):
        return self._id_job

    def __eq__(self, other):
        return isinstance(other, Job)\
            and self.id_job == other.id_job

    def __str__(self):
        return f"Job{{id: {self.id_job}}}"


# ~~~~~~~ Precedence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Precedence:
    _id_child: int
    _id_parent: int

    def __init__(self,
                 id_child: int,
                 id_parent: int):
        self._id_child = id_child
        self._id_parent = id_parent

    @property
    def id_child(self) -> int:
        return self._id_child

    @id_child.setter
    def id_child(self, value: int):
        self._id_child = value

    @property
    def id_parent(self) -> int:
        return self._id_parent

    @id_parent.setter
    def id_parent(self, value: int):
        self._id_parent = value

    def copy(self) -> Self:
        return Precedence(id_child=self.id_child,
                          id_parent=self.id_parent)

    def __hash__(self):
        return hash((self._id_child, self._id_parent))

    def __eq__(self, other):
        return isinstance(other, Precedence)\
            and self.id_child == other.id_child\
            and self.id_parent == other.id_parent

    def __str__(self):
        return f"Precedence{{child: {self.id_child}, parent: {self.id_parent}}}"


# ~~~~~~~ Project ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Project:
    _id_project: int
    _due_date: int
    _tardiness_cost: int

    def __init__(self,
                 id_project: int,
                 due_date: int,
                 tardiness_cost: int):
        self._id_project = id_project
        self._due_date = due_date
        self._tardiness_cost = tardiness_cost

    def __hash__(self):
        return self._id_project

    @property
    def id_project(self) -> int:
        return self._id_project

    @id_project.setter
    def id_project(self, value: int):
        self._id_project = value

    @property
    def due_date(self) -> int:
        return self._due_date

    @due_date.setter
    def due_date(self, value: int):
        self._due_date = value

    @property
    def tardiness_cost(self) -> int:
        return self._tardiness_cost

    @tardiness_cost.setter
    def tardiness_cost(self, value: int):
        self._tardiness_cost = value

    def copy(self) -> Self:
        return Project(id_project=self.id_project,
                       due_date=self.due_date,
                       tardiness_cost=self.tardiness_cost)

    def __str__(self):
        return f"Project{{id: {self.id_project}, due date: {self.due_date}, tardiness cost: {self.tardiness_cost}}}"


# ~~~~~~~ Component ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Component:
    _id_root_job: int
    _weight: int

    def __init__(self,
                 id_root_job: int,
                 weight: int):
        self._id_root_job = id_root_job
        self._weight = weight

    @property
    def id_root_job(self) -> int:
        return self._id_root_job

    @id_root_job.setter
    def id_root_job(self, value: int):
        self._id_root_job = value

    @property
    def weight(self) -> int:
        return self._weight

    @weight.setter
    def weight(self, value: int):
        self._weight = value

    def copy(self) -> Self:
        return Component(id_root_job=self.id_root_job,
                         weight=self.weight)

    def __str__(self):
        return f"Component{{id_root_job: {self.id_root_job}}}"


# ~~~~~~~ ProblemInstance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ProblemInstance:
    _name: Optional[str]

    _horizon: int

    _projects: list[Project] = []
    _resources: list[Resource] = []
    _jobs: list[Job] = []
    _precedences: list[Precedence] = []
    _components: list[Component] = []

    _projects_by_id: dict[int, Project] = {}
    _resources_by_id: dict[int, Resource] = {}
    _resources_by_key: dict[str, Resource] = {}
    _jobs_by_id: dict[int, Job] = {}
    _precedences_by_id_child: dict[int, Iterable[Precedence]] = {}
    _precedences_by_id_parent: dict[int, Iterable[Precedence]] = {}
    _components_by_id_root_job: dict[int, Component] = {}

    def __init__(self,
                 horizon: int,
                 projects: Collection[Project],
                 resources: Collection[Resource],
                 jobs: Collection[Job],
                 precedences: Collection[Precedence],
                 components: Collection[Component] or None = None,
                 name: str = None):
        self._name = name

        self._horizon = horizon

        self._projects = list_of(projects)
        self._resources = list_of(resources)
        self._jobs = list_of(jobs)
        self._precedences = list_of(precedences)
        self._components = list_of(components) if components is not None else [Component(self._jobs[0].id_job, 1)]

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def horizon(self) -> int:
        return self._horizon

    @horizon.setter
    def horizon(self, value: int):
        self._horizon = value

    @property
    def projects(self) -> list[Project]:
        return self._projects

    @projects.setter
    def projects(self, value: list[Project]):
        self._projects = value

    @property
    def projects_by_id(self) -> dict[int, Project]:
        if len(self._projects_by_id) != len(self._projects):
            self._projects_by_id = {p.id_project: p for p in self._projects}
        return self._projects_by_id

    @property
    def resources(self) -> list[Resource]:
        return self._resources

    @resources.setter
    def resources(self, value: list[Resource]):
        self._resources = value

    @property
    def resources_by_id(self) -> dict[int, Resource]:
        if len(self._resources_by_id) != len(self._resources):
            self._resources_by_id = {r.id_resource: r for r in self._resources}
        return self._resources_by_id

    @property
    def resources_by_key(self) -> dict[str, Resource]:
        if len(self._resources_by_key) != len(self._resources):
            self._resources_by_key = {r.key: r for r in self._resources}
        return self._resources_by_key

    @property
    def jobs(self) -> list[Job]:
        return self._jobs

    @jobs.setter
    def jobs(self, value: list[Job]):
        self._jobs = value

    @property
    def jobs_by_id(self) -> dict[int, Job]:
        if len(self._jobs_by_id) != len(self._jobs):
            self._jobs_by_id = {j.id_job: j for j in self._jobs}
        return self._jobs_by_id

    @property
    def precedences(self) -> list[Precedence]:
        return self._precedences

    @precedences.setter
    def precedences(self, value: list[Precedence]):
        self._precedences = value

        # Recompute precedence index as automatically checking and recomputing it in the appropriate properties is expensive
        self._precedences_by_id_child = defaultdict(list)
        self._precedences_by_id_parent = defaultdict(list)
        for p in self._precedences:
            self._precedences_by_id_child[p.id_child] += [p]
            self._precedences_by_id_parent[p.id_parent] += [p]

    @property
    def precedences_by_id_child(self) -> dict[int, Iterable[Precedence]]:
        # This is not recomputed automatically as it is hard to check
        return self._precedences_by_id_child

    @property
    def precedences_by_id_parent(self) -> dict[int, Iterable[Precedence]]:
        # This is not recomputed automatically as it is hard to check
        return self._precedences_by_id_parent

    @property
    def components(self) -> list[Component] or None:
        return self._components

    @components.setter
    def components(self, value: list[Component]):
        self._components = value

    @property
    def components_by_id_root_job(self) -> dict[int, Component]:
        if len(self._components_by_id_root_job) != len(self._components):
            self._components_by_id_root_job = {c.id_root_job: c for c in self._components}
        return self._components_by_id_root_job

    def copy(self) -> Self:
        return ProblemInstance(horizon=self.horizon,
                               projects=[p.copy() for p in self.projects],
                               resources=[r.copy() for r in self.resources],
                               jobs=[j.copy() for j in self.jobs],
                               precedences=[p.copy() for p in self.precedences],
                               components=[c.copy() for c in self.components],
                               name=self.name)

    def __str__(self):
        return f"ProblemInstance{{name: {self._name}, #projects: {len(self.projects)}, " \
               f"#resources: {len(self.resources)}, #jobs: {len(self.jobs)}, #precedences: {len(self.precedences)}}}"


def compute_component_jobs(problem_instance: ProblemInstance) -> dict[Job, Collection[Job]]:
    """
    Given a problem instance, returns a dictionary where each key is a root job of a component and the value is a
    collection of jobs that belong to that component.

    :param problem_instance: The problem instance to compute the component jobs for.
    :return: A dictionary where each key is a root job of a component and the value is a collection of jobs that belong
             to that component.
    """
    from instances.algorithms import traverse_instance_graph

    jobs_by_id = {j.id_job: j for j in problem_instance.jobs}
    jobs_components_grouped =\
        [[jobs_by_id[i[0]] for i in group]
         for _k, group in itertools.groupby(traverse_instance_graph(problem_instance, search="components topological generations",
                                                                    yield_state=True),
                                            key=lambda x: x[1])]  # we assume that the order in which jobs are returned is determined by the components, so we do not sort by component id
    component_jobs_by_root_job = index_groups(jobs_components_grouped,
                                              [jobs_by_id[c.id_root_job] for c in problem_instance.components])
    return component_jobs_by_root_job


def compute_resource_periodical_availability(resource: Resource, horizon: int) -> T_StepFunction:
    days_count = math.ceil(horizon / 24)
    intervals = [(i_day * 24 + start, i_day * 24 + end, capacity)
                 for i_day in range(days_count)
                 for start, end, capacity in resource.availability.periodical_intervals]
    return intervals


def compute_resource_modified_availability(resource: Resource, instance: ProblemInstance, horizon: int) -> T_StepFunction:
    days_count = math.ceil(horizon / 24)
    additions = resource.availability.additions  # additions
    out_migrations = [(s, e, -c) for _r_to, s, e, c in resource.availability.migrations]  # out-migrations
    in_migrations = []
    for resource_from in instance.resources:
        if resource_from.key == resource.key:
            continue
        in_migrations += [(s, e, c) for _r_to, s, e, c in resource_from.availability.migrations if _r_to == resource.key]
    return interval_overlap_function(additions + in_migrations + out_migrations, first_x=0, last_x=days_count*24)


def compute_resource_availability(resource: Resource, instance: ProblemInstance, horizon: int) -> T_StepFunction:
    """
    Builds a step function representing the availability of a resource over time.

    Args:
        resource (Resource): The resource to build the availability function for.
        instance (ProblemInstance): The instance of whose resources to use for building the availability.
        horizon (int): The total number of hours in the planning horizon.

    Returns:
        CpoStepFunction: A step function representing the availability of the resource.
    """
    days_count = math.ceil(horizon / 24)
    periodical = compute_resource_periodical_availability(resource, horizon)
    modified = compute_resource_modified_availability(resource, instance, horizon)
    return interval_overlap_function(periodical + modified, first_x=0, last_x=days_count*24)
