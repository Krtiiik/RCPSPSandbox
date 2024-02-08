import itertools
import random
from typing import Self, Literal

import networkx as nx

from instances.instance_builder import InstanceBuilder
from instances.algorithms import traverse_instance_graph, build_instance_graph, topological_sort, \
    paths_traversal, subtree_traversal
from instances.problem_instance import ProblemInstance, Job, Precedence, Resource, ResourceAvailability, AvailabilityInterval, Component
from utils import print_error


class ProblemModifier:
    _original_instance: ProblemInstance

    _jobs: list[Job] = []
    _precedences: list[Precedence] = []
    _resources: list[Resource] = []
    _components: list[Component] = []

    def __init__(self,
                 original_instance: ProblemInstance):
        self._original_instance = original_instance

        self._jobs = original_instance.jobs[:]
        self._precedences = original_instance.precedences[:]
        self._resources = original_instance.resources[:]
        self._components = original_instance.components[:]

    def assign_resource_availabilities(self) -> Self:
        for resource in self.resources:
            resource.availability = [AvailabilityInterval(0, 24, resource.capacity)]
        return self

    def assign_job_due_dates(self,
                             choice: str = Literal["uniform", "gradual"],
                             interval: tuple[int, int] = None,
                             target_jobs: list[int] = None,
                             gradual_base: int = None,
                             gradual_interval: tuple[int, int] = None,
                             due_dates: dict[int, int] = None,
                             overwrite: bool = False,) -> Self:
        def try_assign(j, dd):
            if j.due_date is None or overwrite:
                j.due_date = dd

        jobs_by_id = {job.id_job: job for job in self.jobs}

        # If an explicit mapping of due-dates was specified...
        if due_dates is not None:
            for id_job, due_date in due_dates.items():
                try_assign(jobs_by_id[id_job], due_date)
            return self

        match choice:
            case "uniform":
                if interval is None:
                    print_error("Uniform due date assignment requires a sample interval but none was given.")
                    return self
                target_jobs = (self.jobs if target_jobs is None
                               else (jobs_by_id[id_job] for id_job in target_jobs))
                for job in target_jobs:
                    due_date = round(random.uniform(interval[0], interval[1]))
                    try_assign(job, due_date)

            case "gradual":
                if gradual_interval is None:
                    gradual_interval = (0, 0)
                for node, parent in topological_sort(build_instance_graph(self), yield_state=True):
                    parent_end = jobs_by_id[parent].due_date if parent is not None else gradual_base
                    jobs_by_id[node].due_date = parent_end + jobs_by_id[node].duration + round(random.uniform(*gradual_interval))
            case _:
                print_error("Unrecognized choice type for computing due dates")

        return self

    def complete_jobs(self,
                      jobs_to_complete: list[int] or None = None,
                      choice: Literal["random", "gradual", "combined"] = None,
                      ratio: float = None,
                      combined_ratio: float = None) -> Self:
        def complete(jbs):
            for j in jbs:
                j.completed = True

        def choose_random(count):
            return random.sample(self.jobs, count)

        def choose_gradual(count):
            jobs_to_complete_traverser = traverse_instance_graph(graph=build_instance_graph(self), search="uniform")
            return itertools.islice(jobs_to_complete_traverser, count)

        if jobs_to_complete is not None:
            jobs_by_id = {j.id_job: j for j in self.jobs}
            complete(jobs_by_id[id_job] for id_job in jobs_to_complete)
        elif choice is not None and choice in ["random", "gradual", "combined"]:
            match choice:
                case "random":
                    jobs_to_complete_count = ratio * len(self.jobs)
                    jobs_to_complete = choose_random(jobs_to_complete_count)
                    complete(jobs_to_complete)
                case "gradual":
                    jobs_to_complete_count = round(ratio * len(self.jobs))
                    jobs_to_complete = choose_gradual(jobs_to_complete_count)
                    complete(jobs_to_complete)
                case "combined":
                    jobs_to_complete_count = round(ratio * len(self.jobs))
                    gradual_count = round(combined_ratio * jobs_to_complete_count)
                    random_count = round((1 - combined_ratio) * jobs_to_complete_count)
                    gradual_jobs = choose_gradual(gradual_count)
                    random_jobs = choose_random(random_count)
                    complete(gradual_jobs)
                    complete(random_jobs)
        else:
            print_error("No jobs to complete were given and neither a valid job completion choice were given.")

        return self

    def split_job_components(self,
                             split: Literal["trim source target", "random roots", "paths"]) -> Self:
        match split:
            case "trim source target":
                instance_graph = build_instance_graph(self)
                if nx.number_weakly_connected_components(instance_graph) != 1:
                    print_error("Current instance graph does not satisfy the requirements to trim the source and target nodes. The graph contains more than 1 component.")
                    return self

                topo = list(topological_sort(instance_graph))  # this is very ineffective, forgive me
                source, target = topo[0], topo[-1]
                instance_graph.remove_nodes_from([source, target])

                graph_components = list(nx.weakly_connected_components(instance_graph))
                components = [Component(next(iter(component)), 1) for component in graph_components]
                precedences = [Precedence(u, v) for u, v in instance_graph.edges]

                self.jobs = [job for job in self.jobs if job.id_job not in {source, target}]
                self.precedences = precedences
                self.components = components
            case "random roots":
                graph = build_instance_graph(self)
                subtrees = []
                while graph:
                    root = random.choice(list(graph.nodes))
                    subtree = subtree_traversal(graph, root)
                    graph.remove_nodes_from(subtree)
                    subtrees.append(subtree)

                instance_graph = build_instance_graph(self)
                instance_graph: nx.DiGraph = nx.union_all(instance_graph.subgraph(subtree) for subtree in subtrees)

                components = [Component(subtree[0], 1)
                              for subtree in subtrees]
                precedences = [Precedence(u, v)
                               for u, v in instance_graph.edges]
                self.precedences = precedences
                self.components = components
                pass
            case "paths":
                paths = paths_traversal(build_instance_graph(self))
                precedences = [Precedence(child, parent)
                               for path in paths
                               for child, parent in zip(path, path[1:])]
                components = [Component(path[0], 1) for path in paths]  # Component weight is not set, can be set manually

                self.precedences = precedences
                self.components = components
            case _:
                print_error(f"Unrecognized split option: {split}")
        return self

    def merge_with(self, other: ProblemInstance) -> Self:
        other_jobs = other.jobs[:]
        other_precedences = other.precedences[:]
        other_components = other.components[:]

        id_offset = 1 + max(job.id_job for job in self.jobs)
        for job in other_jobs:
            job.id_job += id_offset
        for precedence in other_precedences:
            precedence.id_child += id_offset
            precedence.id_parent += id_offset
        for component in other_components:
            component.id_root_job += id_offset

        self.jobs.extend(other_jobs)
        self.precedences.extend(other_precedences)
        # Resources are assumed to be the same
        self.components.extend(other_components)

        return self

    def generate_modified_instance(self, name: str = None) -> ProblemInstance:
        builder = InstanceBuilder()
        builder.add_jobs(self.jobs)
        builder.add_precedences(self.precedences)
        builder.add_components(self.components)
        builder.add_resources(self._original_instance.resources)
        builder.add_projects(self._original_instance.projects)
        builder.set(horizon=self._original_instance.horizon,
                    name=name if name is not None else f"{self._original_instance.name}_modified")
        return builder.build_instance()

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

    @property
    def resources(self) -> list[Resource]:
        return self._resources

    @resources.setter
    def resources(self, value: list[Resource]):
        self._resources = value

    @property
    def components(self) -> list[Component]:
        return self._components

    @components.setter
    def components(self, value: list[Component]):
        self._components = value


def modify_instance(problem_instance: ProblemInstance) -> ProblemModifier:
    return ProblemModifier(problem_instance)
