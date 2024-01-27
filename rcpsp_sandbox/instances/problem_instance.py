from collections import defaultdict
from enum import StrEnum
from typing import Optional, Collection, Self, Iterable

from instances.utils import list_of


# ~~~~~~~ ResourceType ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResourceType(StrEnum):
    RENEWABLE = 'R'
    NONRENEWABLE = 'N'
    DOUBLY_CONSTRAINED = 'D'


# ~~~~~~~ AvailabilityInterval ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AvailabilityInterval:
    _start: int
    _end: int
    _capacity: int or None

    def __init__(self,
                 start: int,
                 end: int,
                 capacity: int or None = None):
        self._start = start
        self._end = end
        self._capacity = capacity

    @property
    def start(self) -> int:
        return self._start

    @start.setter
    def start(self, value: int):
        self._start = value

    @property
    def end(self) -> int:
        return self._end

    @end.setter
    def end(self, value: int):
        self._end = value

    @property
    def capacity(self) -> int or None:
        return self._capacity

    @capacity.setter
    def capacity(self, value: int or None):
        self._capacity = value

    def copy(self) -> Self:
        return AvailabilityInterval(start=self.start,
                                    end=self.end,
                                    capacity=self.capacity)

    def __str__(self):
        return (f"[{self.start, self.end}]" if self.capacity is None
                else f"[{self.start}, {self.end}]<{self.capacity}>")

    def __iter__(self):
        return iter((self.start, self.end))


# ~~~~~~~ ResourceAvailability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResourceAvailability:
    _base_intervals: list[AvailabilityInterval]
    _override_intervals: list[AvailabilityInterval]

    def __init__(self, base_intervals: list[AvailabilityInterval]) -> None:
        self._base_intervals = base_intervals


# ~~~~~~~ Resource ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Resource:
    _id_resource: int
    _type: ResourceType
    _capacity: int
    _availability: list[AvailabilityInterval] or None

    def __init__(self,
                 id_resource: int,
                 resource_type: ResourceType,
                 capacity: int,
                 availability: list[AvailabilityInterval] or None = None):
        self._id_resource = id_resource
        self._type = resource_type
        self._capacity = capacity
        self._availability = availability

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
    def availability(self) -> list[AvailabilityInterval] or None:
        return self._availability

    @availability.setter
    def availability(self, value: list[AvailabilityInterval] or None):
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
