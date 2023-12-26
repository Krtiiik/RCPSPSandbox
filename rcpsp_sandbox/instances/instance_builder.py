from typing import TypeVar, Iterable, Optional

from instances.problem_instance import ProblemInstance, Resource, Job, Precedence, Project, Component

T = TypeVar("T")


class InstanceBuilder:
    _name: Optional[str]

    _horizon: int

    _projects: set[Project]
    _resources: set[Resource]
    _jobs: set[Job]
    _precedences: set[Precedence]
    _components: set[Component]

    def __init__(self):
        self._horizon = 0

        self._projects = set()
        self._resources = set()
        self._jobs = set()
        self._precedences = set()
        self._components = set()

    @staticmethod
    def _check_new_entity(entities: set[T],
                          new_entity: T) -> None:
        if new_entity in entities:
            raise InstanceBuilderError(f"Given {type(new_entity).__name__} has already been added")

    def add_project(self,
                    project: Project) -> None:
        self._check_new_entity(self._projects, project)
        self._projects.add(project)

    def add_resource(self,
                     resource: Resource) -> None:
        self._check_new_entity(self._resources, resource)
        self._resources.add(resource)

    def add_job(self,
                job: Job) -> None:
        self._check_new_entity(self._jobs, job)
        self._jobs.add(job)

    def add_precedence(self,
                       precedence: Precedence) -> None:
        self._check_new_entity(self._precedences, precedence)
        self._precedences.add(precedence)

    def add_component(self,
                      component: Component) -> None:
        self._components.add(component)

    def add_projects(self,
                     projects: Iterable[Project]) -> None:
        for new_project in projects:
            self.add_project(new_project)

    def add_resources(self,
                      resources: Iterable[Resource]) -> None:
        for new_resource in resources:
            self.add_resource(new_resource)

    def add_jobs(self,
                 jobs: Iterable[Job]) -> None:
        for new_job in jobs:
            self.add_job(new_job)

    def add_precedences(self,
                        precedences: Iterable[Precedence]) -> None:
        for new_precedence in precedences:
            self.add_precedence(new_precedence)

    def add_components(self,
                       components: Iterable[Component]) -> None:
        for new_component in components:
            self.add_component(new_component)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if key == "name":
                assert isinstance(value, str), f"string value for {key} expected"
                self._name = value
            elif key == "horizon":
                assert isinstance(value, int), f"integer value for {key} expected"
                self._horizon = value
            else:
                raise InstanceBuilderError(f"Unexpected instance property {{{key}: {value}}}")

    def build_instance(self) -> ProblemInstance:
        return ProblemInstance(self._horizon, self._projects, self._resources, self._jobs, self._precedences, self._components, name=self._name)

    def clear(self):
        self._projects.clear()
        self._resources.clear()
        self._jobs.clear()
        self._precedences.clear()


class InstanceBuilderError(Exception):
    pass
