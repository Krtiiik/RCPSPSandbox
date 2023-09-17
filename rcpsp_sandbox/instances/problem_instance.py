from enum import StrEnum
from typing import Optional, Collection

from instances.utils import list_of


class ResourceType(StrEnum):
    RENEWABLE = 'R'
    NONRENEWABLE = 'N'
    DOUBLY_CONSTRAINED = 'D'

    def __str__(self):
        return {
            'R': "Renewable",
            'N': "Non-Renewable",
            'D': "Doubly Constrained",
        }[self]


class Resource:
    _id_resource: int
    _type: ResourceType
    _capacity: int

    def __init__(self,
                 id_resource: int,
                 resource_type: ResourceType,
                 capacity: int):
        self._id_resource = id_resource
        self._type = resource_type
        self._capacity = capacity

    @property
    def id_resource(self) -> int:
        return self._id_resource

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
    def key(self) -> str:
        return f"{self.type}{self.id_resource}"

    def __hash__(self):
        return self._id_resource

    def __str__(self):
        return f"Resource{{id: {self.id_resource}, type: {self.type}, capacity: {self.capacity}}}"


class ResourceConsumption:
    _duration: int
    _consumption_by_resource: dict[Resource, int]

    def __init__(self,
                 duration: int,
                 consumption_by_resource: dict[Resource, int]):
        self._duration = duration
        self._consumption_by_resource = consumption_by_resource

    @property
    def duration(self) -> int:
        return self._duration

    @duration.setter
    def duration(self, value: int):
        self._duration = value

    @property
    def consumption_by_resource(self) -> dict[Resource, int]:
        return self._consumption_by_resource

    @consumption_by_resource.setter
    def consumption_by_resource(self, value: dict[Resource, int]):
        self._consumption_by_resource = value

    def __str__(self):
        return f"ResourceConsumption{{duration: {self.duration}, consumptions: {self.consumption_by_resource}}}"


class Job:
    _id_job: int
    _resource_consumption: ResourceConsumption
    _due_date: int or None

    def __init__(self,
                 id_job: int,
                 resource_consumption: ResourceConsumption,
                 due_date: int or None = None):
        self._id_job = id_job
        self._resource_consumption = resource_consumption
        self._due_date = due_date

    @property
    def id_job(self) -> int:
        return self._id_job

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

    def __hash__(self):
        return self._id_job

    def __str__(self):
        return f"Job{{id: {self.id_job}, resource_consumption: {self.resource_consumption}, due_date: {self.due_date}}}"


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

    def __hash__(self):
        return hash((self._id_child, self._id_parent))

    def __str__(self):
        return f"Precedence{{child: {self.id_child}, parent: {self.id_parent}}}"


class Project:
    _id_project: int
    _due_date: int
    _tardiness_cost: int

    def __init__(self,
                 id_project,
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

    def __str__(self):
        return f"Project{{id: {self.id_project}, due date: {self.due_date}, tardiness cost: {self.tardiness_cost}}}"


class ProblemInstance:
    _name: Optional[str]

    _horizon: int

    _projects: list[Project] = []
    _resources: list[Resource] = []
    _jobs: list[Job] = []
    _precedences: list[Precedence] = []

    def __init__(self,
                 horizon: int,
                 projects: Collection[Project],
                 resources: Collection[Resource],
                 jobs: Collection[Job],
                 precedences: Collection[Precedence],
                 name: str = None):
        self._name = name

        self._horizon = horizon

        self._projects = list_of(projects)
        self._resources = list_of(resources)
        self._jobs = list_of(jobs)
        self._precedences = list_of(precedences)

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
    def resources(self) -> list[Resource]:
        return self._resources

    @resources.setter
    def resources(self, value: list[Resource]):
        self._resources = value

    @property
    def jobs(self) -> list[Job]:
        return self._jobs

    @jobs.setter
    def jobs(self, value: list[Job]):
        self._jobs = value

    @property
    def precedences(self) -> list[Precedence]:
        return self._precedences

    @precedences.setter
    def precedences(self, value: list[Precedence]):
        self._precedences = value

    def __str__(self):
        return f"ProblemInstance{{name: {self._name}, #projects: {len(self.projects)}, " \
               f"#resources: {len(self.resources)}, #jobs: {len(self.jobs)}, #precedences: {len(self.precedences)}}}"
