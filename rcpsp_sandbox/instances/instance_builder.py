from typing import TypeVar, Iterable, Optional

from instances.problem_instance import ProblemInstance, Resource, Job, Precedence, Project, Component

T = TypeVar("T")


class InstanceBuilder:
    """
    Class responsible for building problem instances.
    """

    def __init__(self):
        """
        Initializes a new instance of the InstanceBuilder class.
        """
        self._name = None
        self._horizon = 0
        self._target_job = 0
        self._projects = set()
        self._resources = set()
        self._jobs = set()
        self._precedences = set()
        self._components = set()

    @staticmethod
    def _check_new_entity(entities: set[T], new_entity: T) -> None:
        """
        Checks if a new entity has already been added to the set of entities.

        Args:
            entities (set[T]): The set of entities to check against.
            new_entity (T): The new entity to be added.

        Raises:
            InstanceBuilderError: If the new entity has already been added.
        """
        if new_entity in entities:
            raise InstanceBuilderError(f"Given {type(new_entity).__name__} has already been added")

    def add_project(self, project: Project) -> None:
        """
        Adds a project to the problem instance.

        Args:
            project (Project): The project to be added.
        """
        self._check_new_entity(self._projects, project)
        self._projects.add(project)

    def add_resource(self, resource: Resource) -> None:
        """
        Adds a resource to the problem instance.

        Args:
            resource (Resource): The resource to be added.
        """
        self._check_new_entity(self._resources, resource)
        self._resources.add(resource)

    def add_job(self, job: Job) -> None:
        """
        Adds a job to the problem instance.

        Args:
            job (Job): The job to be added.
        """
        self._check_new_entity(self._jobs, job)
        self._jobs.add(job)

    def add_precedence(self, precedence: Precedence) -> None:
        """
        Adds a precedence constraint to the problem instance.

        Args:
            precedence (Precedence): The precedence constraint to be added.
        """
        self._check_new_entity(self._precedences, precedence)
        self._precedences.add(precedence)

    def add_component(self, component: Component) -> None:
        """
        Adds a component to the problem instance.

        Args:
            component (Component): The component to be added.
        """
        self._components.add(component)

    def add_projects(self, projects: Iterable[Project]) -> None:
        """
        Adds multiple projects to the problem instance.

        Args:
            projects (Iterable[Project]): The projects to be added.
        """
        for new_project in projects:
            self.add_project(new_project)

    def add_resources(self, resources: Iterable[Resource]) -> None:
        """
        Adds multiple resources to the problem instance.

        Args:
            resources (Iterable[Resource]): The resources to be added.
        """
        for new_resource in resources:
            self.add_resource(new_resource)

    def add_jobs(self, jobs: Iterable[Job]) -> None:
        """
        Adds multiple jobs to the problem instance.

        Args:
            jobs (Iterable[Job]): The jobs to be added.
        """
        for new_job in jobs:
            self.add_job(new_job)

    def add_precedences(self, precedences: Iterable[Precedence]) -> None:
        """
        Adds multiple precedence constraints to the problem instance.

        Args:
            precedences (Iterable[Precedence]): The precedence constraints to be added.
        """
        for new_precedence in precedences:
            self.add_precedence(new_precedence)

    def add_components(self, components: Iterable[Component]) -> None:
        """
        Adds multiple components to the problem instance.

        Args:
            components (Iterable[Component]): The components to be added.
        """
        for new_component in components:
            self.add_component(new_component)

    def set(self, **kwargs):
        """
        Sets the properties of the problem instance.

        Args:
            **kwargs: The properties to be set.

        Raises:
            InstanceBuilderError: If an unexpected instance property is provided.
        """
        for key, value in kwargs.items():
            if key == "name":
                assert isinstance(value, str), f"string value for {key} expected"
                self._name = value
            elif key == "horizon":
                assert isinstance(value, int), f"integer value for {key} expected"
                self._horizon = value
            elif key == "target_job":
                assert isinstance(value, int), f"integer value for {key} expected"
                self._target_job = value
            else:
                raise InstanceBuilderError(f"Unexpected instance property {{{key}: {value}}}")

    def build_instance(self) -> ProblemInstance:
        """
        Builds the problem instance.

        Returns:
            ProblemInstance: The built problem instance.
        """
        return ProblemInstance(
            self._horizon,
            self._target_job,
            sorted(self._projects, key=lambda p: p.id_project),
            sorted(self._resources, key=lambda r: r.key),
            sorted(self._jobs, key=lambda j: j.id_job),
            sorted(self._precedences, key=lambda p: (p.id_child, p.id_parent)),
            sorted(self._components, key=lambda c: c.id_root_job),
            name=self._name
        )

    def clear(self):
        """
        Clears all the entities in the problem instance.
        """
        self._projects.clear()
        self._resources.clear()
        self._jobs.clear()
        self._precedences.clear()


class InstanceBuilderError(Exception):
    """
    Exception raised when an error occurs in the InstanceBuilder class.
    """
    pass
