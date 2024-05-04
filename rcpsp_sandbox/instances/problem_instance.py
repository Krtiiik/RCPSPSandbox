import itertools
import math
from collections import defaultdict, namedtuple
from enum import StrEnum
from typing import Optional, Collection, Self, Iterable

from utils import index_groups, T_StepFunction, interval_overlap_function, list_of


# ~~~~~~~ ResourceType ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResourceType(StrEnum):
    """
    Enumeration representing the type of a resource.

    Attributes:
        RENEWABLE (str): Represents a renewable resource.
        NONRENEWABLE (str): Represents a non-renewable resource.
        DOUBLY_CONSTRAINED (str): Represents a doubly constrained resource.
    """
    RENEWABLE = 'R'
    NONRENEWABLE = 'N'
    DOUBLY_CONSTRAINED = 'D'


# ~~~~~~~ ResourceAvailability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AvailabilityInterval = namedtuple("AvailabilityInterval", ("start", 'end', "capacity"))
CapacityChange = namedtuple("CapacityChange", ("start", "end", "capacity"))
CapacityMigration = namedtuple("CapacityMigration", ("resource_to", "start", "end", "capacity"))


class ResourceAvailability:
    """
    Represents the availability of a resource over time.
    """

    def __init__(self,
                 periodical_intervals: Iterable[AvailabilityInterval],
                 additions: Iterable[CapacityChange] = None,
                 migrations: Iterable[CapacityMigration] = None,
                 ):
        """
        Initializes a new instance of the ResourceAvailability class.

        Args:
            periodical_intervals (Iterable[AvailabilityInterval]): Iterable of availability intervals.
            additions (Iterable[CapacityChange], optional): Iterable of capacity changes. Defaults to None.
            migrations (Iterable[CapacityMigration], optional): Iterable of capacity migrations. Defaults to None.
        """
        self._periodical_intervals = list_of(periodical_intervals)
        self._additions = list_of(additions) if additions is not None else []
        self._migrations = list_of(migrations) if migrations is not None else []

    @property
    def periodical_intervals(self) -> list[AvailabilityInterval]:
        """
        Gets or sets the list of periodical availability intervals.

        Returns:
            list[AvailabilityInterval]: The list of availability intervals.
        """
        return self._periodical_intervals

    @periodical_intervals.setter
    def periodical_intervals(self, value: Iterable[AvailabilityInterval]):
        """
        Sets the list of periodical availability intervals.

        Args:
            value (Iterable[AvailabilityInterval]): The list of availability intervals.
        """
        self._periodical_intervals = list_of(value)

    @property
    def additions(self) -> list[CapacityChange]:
        """
        Gets or sets the list of capacity changes.

        Returns:
            list[CapacityChange]: The list of capacity changes.
        """
        return self._additions

    @additions.setter
    def additions(self, value: Iterable[CapacityChange]):
        """
        Sets the list of capacity changes.

        Args:
            value (Iterable[CapacityChange]): The list of capacity changes.
        """
        self._additions = list_of(value)

    @property
    def migrations(self) -> list[CapacityMigration]:
        """
        Gets or sets the list of capacity migrations.

        Returns:
            list[CapacityMigration]: The list of capacity migrations.
        """
        return self._migrations

    @migrations.setter
    def migrations(self, value: Iterable[CapacityMigration]):
        """
        Sets the list of capacity migrations.

        Args:
            value (Iterable[CapacityMigration]): The list of capacity migrations.
        """
        self._migrations = list_of(value)

    def copy(self) -> "ResourceAvailability":
        """
        Creates a copy of the ResourceAvailability object.

        Returns:
            ResourceAvailability: A copy of the ResourceAvailability object.
        """
        return ResourceAvailability(periodical_intervals=self.periodical_intervals[:],
                                    additions=self._additions[:],
                                    migrations=self._migrations[:],
                                    )


# ~~~~~~~ Resource ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Resource:
    """
    Represents a resource in the problem instance.
    """

    _id_resource: int
    _type: ResourceType
    _capacity: int
    _availability: ResourceAvailability

    def __init__(self,
                 id_resource: int,
                 resource_type: ResourceType,
                 capacity: int,
                 availability: ResourceAvailability = None):
        """
        Initializes a new instance of the Resource class.

        Args:
            id_resource (int): The ID of the resource.
            resource_type (ResourceType): The type of the resource.
            capacity (int): The capacity of the resource.
            availability (ResourceAvailability, optional): The availability of the resource. Defaults to None.
        """
        self._id_resource = id_resource
        self._type = resource_type
        self._capacity = capacity
        self._availability = availability if availability is not None else ResourceAvailability([AvailabilityInterval(0, 24, capacity)])

    @property
    def id_resource(self) -> int:
        """
        int: The ID of the resource.
        """
        return self._id_resource

    @id_resource.setter
    def id_resource(self, value: int):
        """
        Sets the ID of the resource.

        Args:
            value (int): The ID of the resource.
        """
        self._id_resource = value

    @property
    def type(self) -> ResourceType:
        """
        ResourceType: The type of the resource.
        """
        return self._type

    @type.setter
    def type(self, value: ResourceType):
        """
        Sets the type of the resource.

        Args:
            value (ResourceType): The type of the resource.
        """
        self._type = value

    @property
    def capacity(self) -> int:
        """
        int: The capacity of the resource.
        """
        return self._capacity

    @capacity.setter
    def capacity(self, value: int):
        """
        Sets the capacity of the resource.

        Args:
            value (int): The capacity of the resource.
        """
        self._capacity = value

    @property
    def availability(self) -> ResourceAvailability:
        """
        ResourceAvailability: The availability of the resource.
        """
        return self._availability

    @availability.setter
    def availability(self, value: ResourceAvailability):
        """
        Sets the availability of the resource.

        Args:
            value (ResourceAvailability): The availability of the resource.
        """
        self._availability = value

    @property
    def key(self) -> str:
        """
        str: The key of the resource.
        """
        return f"{self.type}{self.id_resource}"

    def copy(self) -> 'Resource':
        """
        Creates a copy of the resource.

        Returns:
            Resource: A copy of the resource.
        """
        return Resource(id_resource=self.id_resource,
                        resource_type=self.type,
                        capacity=self.capacity,
                        availability=self.availability.copy() if self.availability is not None else None)

    def __hash__(self):
        """
        Returns the hash value of the resource.

        Returns:
            int: The hash value of the resource.
        """
        return hash((self._id_resource, self._type))

    def __eq__(self, other):
        """
        Checks if the resource is equal to another resource.

        Args:
            other (Resource): The other resource to compare.

        Returns:
            bool: True if the resources are equal, False otherwise.
        """
        return isinstance(other, Resource)\
            and self.id_resource == other.id_resource \
            and self.type == other.type

    def __str__(self):
        """
        Returns a string representation of the resource.

        Returns:
            str: A string representation of the resource.
        """
        return f"Resource{{id: {self.id_resource}, type: {self.type}}}"


# ~~~~~~~ ResourceConsumption ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResourceConsumption:
    """
    Represents the resource consumption for a task in a problem instance.
    """

    _consumption_by_resource: dict[Resource, int]

    def __init__(self, consumption_by_resource: dict[Resource, int]):
        """
        Initializes a new instance of the ResourceConsumption class.

        Args:
            consumption_by_resource (dict[Resource, int]): A dictionary mapping resources to their consumption values.
        """
        self._consumption_by_resource = consumption_by_resource

    @property
    def consumption_by_resource(self) -> dict[Resource, int]:
        """
        Gets the dictionary of resource consumption values.

        Returns:
            dict[Resource, int]: A dictionary mapping resources to their consumption values.
        """
        return self._consumption_by_resource

    @consumption_by_resource.setter
    def consumption_by_resource(self, value: dict[Resource, int]):
        """
        Sets the dictionary of resource consumption values.

        Args:
            value (dict[Resource, int]): A dictionary mapping resources to their consumption values.
        """
        self._consumption_by_resource = value

    def copy(self) -> 'ResourceConsumption':
        """
        Creates a copy of the ResourceConsumption object.

        Returns:
            ResourceConsumption: A new instance of the ResourceConsumption class with the same consumption values.
        """
        return ResourceConsumption(consumption_by_resource=self.consumption_by_resource.copy())

    def __getitem__(self, resource: Resource | int):
        """
        Gets the consumption value for the specified resource.

        Args:
            resource (Resource | int): The resource or its index.

        Returns:
            int: The consumption value for the specified resource.
        """
        return self.consumption_by_resource[resource]

    def __str__(self):
        """
        Returns a string representation of the ResourceConsumption object.

        Returns:
            str: A string representation of the ResourceConsumption object.
        """
        return f"ResourceConsumption{{consumptions: {self.consumption_by_resource}}}"


# ~~~~~~~ Job ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Job:
    """
    Represents a job in a problem instance.
    """

    def __init__(self,
                 id_job: int,
                 duration: int,
                 resource_consumption: ResourceConsumption,
                 due_date: int = 0,
                 completed: bool = False):
        """
        Initializes a new instance of the Job class.

        Args:
            id_job (int): The ID of the job.
            duration (int): The duration of the job.
            resource_consumption (ResourceConsumption): The resource consumption of the job.
            due_date (int, optional): The due date of the job. Defaults to 0.
            completed (bool, optional): Indicates whether the job has been completed. Defaults to False.
        """
        self._id_job = id_job
        self._duration = duration
        self._resource_consumption = resource_consumption
        self._due_date = due_date
        self._completed = completed

    @property
    def id_job(self) -> int:
        """
        Gets the ID of the job.

        Returns:
            int: The ID of the job.
        """
        return self._id_job

    @id_job.setter
    def id_job(self, value: int):
        """
        Sets the ID of the job.

        Args:
            value (int): The ID of the job.
        """
        self._id_job = value

    @property
    def duration(self) -> int:
        """
        Gets the duration of the job.

        Returns:
            int: The duration of the job.
        """
        return self._duration

    @duration.setter
    def duration(self, value: int):
        """
        Sets the duration of the job.

        Args:
            value (int): The duration of the job.
        """
        self._duration = value

    @property
    def resource_consumption(self) -> ResourceConsumption:
        """
        Gets the resource consumption of the job.

        Returns:
            ResourceConsumption: The resource consumption of the job.
        """
        return self._resource_consumption

    @resource_consumption.setter
    def resource_consumption(self, value: ResourceConsumption):
        """
        Sets the resource consumption of the job.

        Args:
            value (ResourceConsumption): The resource consumption of the job.
        """
        self._resource_consumption = value

    @property
    def due_date(self) -> int | None:
        """
        Gets the due date of the job.

        Returns:
            int or None: The due date of the job, or None if there is no due date.
        """
        return self._due_date

    @due_date.setter
    def due_date(self, value: int):
        """
        Sets the due date of the job.

        Args:
            value (int): The due date of the job.
        """
        self._due_date = value

    @property
    def completed(self) -> bool:
        """
        Gets a value indicating whether the job has been completed.

        Returns:
            bool: True if the job has been completed, False otherwise.
        """
        return self._completed

    @completed.setter
    def completed(self, value: bool):
        """
        Sets a value indicating whether the job has been completed.

        Args:
            value (bool): True if the job has been completed, False otherwise.
        """
        self._completed = value

    def copy(self) -> 'Job':
        """
        Creates a copy of the job.

        Returns:
            Job: A copy of the job.
        """
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
    """
    Represents a precedence relationship between two tasks in a project scheduling problem.
    """

    _id_child: int
    _id_parent: int

    def __init__(self,
                 id_child: int,
                 id_parent: int):
        """
        Initializes a new Precedence object.

        Args:
            id_child (int): The ID of the child task.
            id_parent (int): The ID of the parent task.
        """
        self._id_child = id_child
        self._id_parent = id_parent

    @property
    def id_child(self) -> int:
        """
        Gets the ID of the child task.

        Returns:
            int: The ID of the child task.
        """
        return self._id_child

    @id_child.setter
    def id_child(self, value: int):
        """
        Sets the ID of the child task.

        Args:
            value (int): The ID of the child task.
        """
        self._id_child = value

    @property
    def id_parent(self) -> int:
        """
        Gets the ID of the parent task.

        Returns:
            int: The ID of the parent task.
        """
        return self._id_parent

    @id_parent.setter
    def id_parent(self, value: int):
        """
        Sets the ID of the parent task.

        Args:
            value (int): The ID of the parent task.
        """
        self._id_parent = value

    def copy(self) -> 'Precedence':
        """
        Creates a copy of the Precedence object.

        Returns:
            Precedence: A new Precedence object with the same child and parent IDs.
        """
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
    """
    Represents a project with its ID, due date, and tardiness cost.
    """

    _id_project: int
    _due_date: int
    _tardiness_cost: int

    def __init__(self,
                 id_project: int,
                 due_date: int,
                 tardiness_cost: int):
        """
        Initializes a new Project instance.

        Args:
            id_project (int): The ID of the project.
            due_date (int): The due date of the project.
            tardiness_cost (int): The tardiness cost of the project.
        """
        self._id_project = id_project
        self._due_date = due_date
        self._tardiness_cost = tardiness_cost

    def __hash__(self):
        """
        Returns the hash value of the Project instance.

        Returns:
            int: The hash value.
        """
        return self._id_project

    @property
    def id_project(self) -> int:
        """
        Gets the ID of the project.

        Returns:
            int: The ID of the project.
        """
        return self._id_project

    @id_project.setter
    def id_project(self, value: int):
        """
        Sets the ID of the project.

        Args:
            value (int): The new ID value.
        """
        self._id_project = value

    @property
    def due_date(self) -> int:
        """
        Gets the due date of the project.

        Returns:
            int: The due date of the project.
        """
        return self._due_date

    @due_date.setter
    def due_date(self, value: int):
        """
        Sets the due date of the project.

        Args:
            value (int): The new due date value.
        """
        self._due_date = value

    @property
    def tardiness_cost(self) -> int:
        """
        Gets the tardiness cost of the project.

        Returns:
            int: The tardiness cost of the project.
        """
        return self._tardiness_cost

    @tardiness_cost.setter
    def tardiness_cost(self, value: int):
        """
        Sets the tardiness cost of the project.

        Args:
            value (int): The new tardiness cost value.
        """
        self._tardiness_cost = value

    def copy(self) -> 'Project':
        """
        Creates a copy of the Project instance.

        Returns:
            Project: A new Project instance with the same attribute values.
        """
        return Project(id_project=self.id_project,
                       due_date=self.due_date,
                       tardiness_cost=self.tardiness_cost)

    def __str__(self):
        """
        Returns a string representation of the Project instance.

        Returns:
            str: A string representation of the Project instance.
        """
        return f"Project{{id: {self.id_project}, due date: {self.due_date}, tardiness cost: {self.tardiness_cost}}}"


# ~~~~~~~ Component ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Component:
    """
    Represents an order component in the problem instance.
    """

    _id_root_job: int
    _weight: int

    def __init__(self, id_root_job: int, weight: int):
        """
        Initializes a new instance of the Component class.

        Args:
            id_root_job (int): The ID of the root job.
            weight (int): The weight of the component.
        """
        self._id_root_job = id_root_job
        self._weight = weight

    @property
    def id_root_job(self) -> int:
        """
        Gets or sets the ID of the root job.

        Returns:
            int: The ID of the root job.
        """
        return self._id_root_job

    @id_root_job.setter
    def id_root_job(self, value: int):
        """
        Sets the ID of the root job.

        Args:
            value (int): The ID of the root job.
        """
        self._id_root_job = value

    @property
    def weight(self) -> int:
        """
        Gets or sets the weight of the component.

        Returns:
            int: The weight of the component.
        """
        return self._weight

    @weight.setter
    def weight(self, value: int):
        """
        Sets the weight of the component.

        Args:
            value (int): The weight of the component.
        """
        self._weight = value

    def copy(self) -> 'Component':
        """
        Creates a copy of the component.

        Returns:
            Component: A new instance of the Component class with the same properties.
        """
        return Component(id_root_job=self.id_root_job, weight=self.weight)

    def __str__(self):
        """
        Returns a string representation of the component.

        Returns:
            str: A string representation of the component.
        """
        return f"Component{{id_root_job: {self.id_root_job}}}"


# ~~~~~~~ ProblemInstance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ProblemInstance:
    _name: Optional[str]

    _horizon: int
    _target_job: int

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
                 target_job: int,
                 projects: Collection[Project],
                 resources: Collection[Resource],
                 jobs: Collection[Job],
                 precedences: Collection[Precedence],
                 components: Collection[Component] | None = None,
                 name: str = None):
        """
        Initialize a ProblemInstance object.

        Args:
            horizon (int): The horizon of the problem instance.
            target_job (int): The target job of the problem instance.
            projects (Collection[Project]): The collection of projects in the problem instance.
            resources (Collection[Resource]): The collection of resources in the problem instance.
            jobs (Collection[Job]): The collection of jobs in the problem instance.
            precedences (Collection[Precedence]): The collection of precedences in the problem instance.
            components (Collection[Component] or None, optional): The collection of components in the problem instance. Defaults to None.
            name (str, optional): The name of the problem instance. Defaults to None.
        """
        self._name = name

        self._horizon = horizon
        self._target_job = target_job

        self._projects = list_of(projects)
        self._resources = list_of(resources)
        self._jobs = list_of(jobs)
        self._precedences = list_of(precedences)
        self._components = list_of(components) if components is not None else [Component(self._jobs[0].id_job, 1)]

    @property
    def name(self) -> Optional[str]:
        """
        Get the name of the problem instance.

        Returns:
            Optional[str]: The name of the problem instance.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Set the name of the problem instance.

        Args:
            value (str): The name of the problem instance.
        """
        self._name = value

    @property
    def horizon(self) -> int:
        """
        Get the horizon of the problem instance.

        Returns:
            int: The horizon of the problem instance.
        """
        return self._horizon

    @horizon.setter
    def horizon(self, value: int):
        """
        Set the horizon of the problem instance.

        Args:
            value (int): The horizon of the problem instance.
        """
        self._horizon = value

    @property
    def target_job(self) -> int:
        """
        Get the target job of the problem instance.

        Returns:
            int: The target job of the problem instance.
        """
        return self._target_job

    @target_job.setter
    def target_job(self, value: int):
        """
        Set the target job of the problem instance.

        Args:
            value (int): The target job of the problem instance.
        """
        self._target_job = value

    @property
    def projects(self) -> list[Project]:
        """
        Get the list of projects in the problem instance.

        Returns:
            list[Project]: The list of projects in the problem instance.
        """
        return self._projects

    @projects.setter
    def projects(self, value: list[Project]):
        """
        Set the list of projects in the problem instance.

        Args:
            value (list[Project]): The list of projects in the problem instance.
        """
        self._projects = value

    @property
    def projects_by_id(self) -> dict[int, Project]:
        """
        Get the dictionary of projects by ID in the problem instance.

        Returns:
            dict[int, Project]: The dictionary of projects by ID in the problem instance.
        """
        if len(self._projects_by_id) != len(self._projects):
            self._projects_by_id = {p.id_project: p for p in self._projects}
        return self._projects_by_id

    @property
    def resources(self) -> list[Resource]:
        """
        Get the list of resources in the problem instance.

        Returns:
            list[Resource]: The list of resources in the problem instance.
        """
        return self._resources

    @resources.setter
    def resources(self, value: list[Resource]):
        """
        Set the list of resources in the problem instance.

        Args:
            value (list[Resource]): The list of resources in the problem instance.
        """
        self._resources = value

    @property
    def resources_by_id(self) -> dict[int, Resource]:
        """
        Get the dictionary of resources by ID in the problem instance.

        Returns:
            dict[int, Resource]: The dictionary of resources by ID in the problem instance.
        """
        if len(self._resources_by_id) != len(self._resources):
            self._resources_by_id = {r.id_resource: r for r in self._resources}
        return self._resources_by_id

    @property
    def resources_by_key(self) -> dict[str, Resource]:
        """
        Get the dictionary of resources by key in the problem instance.

        Returns:
            dict[str, Resource]: The dictionary of resources by key in the problem instance.
        """
        if len(self._resources_by_key) != len(self._resources):
            self._resources_by_key = {r.key: r for r in self._resources}
        return self._resources_by_key

    @property
    def jobs(self) -> list[Job]:
        """
        Get the list of jobs in the problem instance.

        Returns:
            list[Job]: The list of jobs in the problem instance.
        """
        return self._jobs

    @jobs.setter
    def jobs(self, value: list[Job]):
        """
        Set the list of jobs in the problem instance.

        Args:
            value (list[Job]): The list of jobs in the problem instance.
        """
        self._jobs = value

    @property
    def jobs_by_id(self) -> dict[int, Job]:
        """
        Get the dictionary of jobs by ID in the problem instance.

        Returns:
            dict[int, Job]: The dictionary of jobs by ID in the problem instance.
        """
        if len(self._jobs_by_id) != len(self._jobs):
            self._jobs_by_id = {j.id_job: j for j in self._jobs}
        return self._jobs_by_id

    @property
    def precedences(self) -> list[Precedence]:
        """
        Get the list of precedences in the problem instance.

        Returns:
            list[Precedence]: The list of precedences in the problem instance.
        """
        return self._precedences

    @precedences.setter
    def precedences(self, value: list[Precedence]):
        """
        Set the list of precedences in the problem instance.

        Args:
            value (list[Precedence]): The list of precedences in the problem instance.
        """
        self._precedences = value

        # Recompute precedence index as automatically checking and recomputing it in the appropriate properties is expensive
        self._precedences_by_id_child = defaultdict(list)
        self._precedences_by_id_parent = defaultdict(list)
        for p in self._precedences:
            self._precedences_by_id_child[p.id_child] += [p]
            self._precedences_by_id_parent[p.id_parent] += [p]

    @property
    def precedences_by_id_child(self) -> dict[int, Iterable[Precedence]]:
        """
        Get the dictionary of precedences by child ID in the problem instance.

        Returns:
            dict[int, Iterable[Precedence]]: The dictionary of precedences by child ID in the problem instance.
        """
        # This is not recomputed automatically as it is hard to check
        return self._precedences_by_id_child

    @property
    def precedences_by_id_parent(self) -> dict[int, Iterable[Precedence]]:
        """
        Get the dictionary of precedences by parent ID in the problem instance.

        Returns:
            dict[int, Iterable[Precedence]]: The dictionary of precedences by parent ID in the problem instance.
        """
        # This is not recomputed automatically as it is hard to check
        return self._precedences_by_id_parent

    @property
    def components(self) -> list[Component] | None:
        """
        Get the list of components in the problem instance.

        Returns:
            list[Component] or None: The list of components in the problem instance.
        """
        return self._components

    @components.setter
    def components(self, value: list[Component]):
        """
        Set the list of components in the problem instance.

        Args:
            value (list[Component]): The list of components in the problem instance.
        """
        self._components = value

    @property
    def components_by_id_root_job(self) -> dict[int, Component]:
        """
        Get the dictionary of components by root job ID in the problem instance.

        Returns:
            dict[int, Component]: The dictionary of components by root job ID in the problem instance.
        """
        if len(self._components_by_id_root_job) != len(self._components):
            self._components_by_id_root_job = {c.id_root_job: c for c in self._components}
        return self._components_by_id_root_job

    def copy(self) -> Self:
        """
        Create a copy of the problem instance.

        Returns:
            Self: The copy of the problem instance.
        """
        return ProblemInstance(horizon=self.horizon,
                               target_job=self._target_job,
                               projects=[p.copy() for p in self.projects],
                               resources=[r.copy() for r in self.resources],
                               jobs=[j.copy() for j in self.jobs],
                               precedences=[p.copy() for p in self.precedences],
                               components=[c.copy() for c in self.components],
                               name=self.name)

    def __str__(self):
        """
        Get the string representation of the problem instance.

        Returns:
            str: The string representation of the problem instance.
        """
        return f"ProblemInstance{{name: {self._name}, target_job: {self._target_job}}}"


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
    """
    Compute the periodical availability of a resource within a given horizon.

    Args:
        resource (Resource): The resource for which to compute the availability.
        horizon (int): The length of the horizon in hours.

    Returns:
        List[Tuple[int, int, int]]: A list of tuples representing the availability intervals.
            Each tuple contains the start time, end time, and capacity of the resource.

    """
    days_count = math.ceil(horizon / 24)
    intervals = [(i_day * 24 + start, i_day * 24 + end, capacity)
                 for i_day in range(days_count)
                 for start, end, capacity in resource.availability.periodical_intervals]
    return intervals


def compute_resource_modified_availability(resource: Resource, instance: ProblemInstance, horizon: int) -> T_StepFunction:
    """
    Compute the modified availability of a resource within a given horizon.

    Args:
        resource (Resource): The resource for which to compute the modified availability.
        instance (ProblemInstance): The problem instance containing the resource and other information.
        horizon (int): The time horizon for which to compute the modified availability.

    Returns:
        T_StepFunction: The modified availability of the resource as a step function.
    """
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
