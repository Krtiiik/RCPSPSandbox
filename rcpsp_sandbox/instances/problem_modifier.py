import itertools
import math
import random
from typing import Self, Literal, Iterable, Tuple

import networkx as nx

from instances.instance_builder import InstanceBuilder
from instances.algorithms import traverse_instance_graph, build_instance_graph, topological_sort, \
    paths_traversal, subtree_traversal
from instances.problem_instance import ProblemInstance, Job, Precedence, Resource, AvailabilityInterval, Component, \
    ResourceAvailability, CapacityChange, CapacityMigration
from utils import print_error, list_of


class ProblemModifier:
    """
    The ProblemModifier class is responsible for modifying a ProblemInstance object by assigning resource availabilities,
    job due dates, completing jobs, and splitting job components.
    """

    _original_instance: ProblemInstance

    _jobs: list[Job] = []
    _precedences: list[Precedence] = []
    _resources: list[Resource] = []
    _components: list[Component] = []

    def __init__(self,
                 original_instance: ProblemInstance):
        """
        Initializes a ProblemModifier object.

        Args:
            original_instance (ProblemInstance): The original ProblemInstance object.
        """
        self._original_instance = original_instance

        self._jobs = [j.copy() for j in original_instance.jobs]
        self._precedences = [p.copy() for p in original_instance.precedences]
        self._resources = [r.copy() for r in original_instance.resources]
        self._components = [c.copy() for c in original_instance.components]

        self.target_job = original_instance.target_job

        self._random = random.Random(42)

    def assign_resource_availabilities(self,
                                       availabilities: dict[str, Iterable[Tuple[int, int]]] = None,
                                       ) -> Self:
        """
        Assigns resource availabilities to the ProblemModifier object.

        Args:
            availabilities (dict[str, Iterable[Tuple[int, int]]], optional): The dictionary of resource availabilities.
                Defaults to None.

        Returns:
            Self: The modified ProblemModifier object.
        """
        if availabilities is None:
            availabilities = {resource.key: [(0, 24)] for resource in self.resources}

        for resource in self.resources:
            periodical_intervals = [AvailabilityInterval(start, end, resource.capacity)
                                    for start, end in availabilities[resource.key]]
            resource.availability = ResourceAvailability(periodical_intervals)

        return self

    def assign_job_due_dates(self,
                             choice: Literal["uniform", "gradual", "earliest"] = "earliest",
                             interval: tuple[int, int] = None,
                             target_jobs: list[int] = None,
                             gradual_base: int = None,
                             gradual_interval: tuple[int, int] = None,
                             due_dates: dict[int, int] = None,
                             overwrite: bool = False,
                             ) -> Self:
        """
        Assigns job due dates to the ProblemModifier object.

        Args:
            choice (Literal["uniform", "gradual", "earliest"], optional): The choice type for computing due dates.
                Defaults to "earliest".
            interval (tuple[int, int], optional): The sample interval for uniform due date assignment.
                Defaults to None.
            target_jobs (list[int], optional): The list of target job IDs.
                Defaults to None.
            gradual_base (int, optional): The base due date for gradual due date assignment.
                Defaults to None.
            gradual_interval (tuple[int, int], optional): The interval for gradual due date assignment.
                Defaults to None.
            due_dates (dict[int, int], optional): The explicit mapping of due dates.
                Defaults to None.
            overwrite (bool, optional): Whether to overwrite existing due dates.
                Defaults to False.

        Returns:
            Self: The modified ProblemModifier object.
        """
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
                    due_date = round(self._random.uniform(interval[0], interval[1]))
                    try_assign(job, due_date)

            case "gradual":
                if gradual_interval is None:
                    gradual_interval = (0, 0)
                for node, parent in topological_sort(build_instance_graph(self), yield_state=True):
                    parent_end = jobs_by_id[parent].due_date if parent is not None else gradual_base
                    jobs_by_id[node].due_date = parent_end + jobs_by_id[node].duration + round(self._random.uniform(*gradual_interval))
            case "earliest":
                def get_duedate(j): return jobs_by_id[j].due_date
                def get_duration(j): return jobs_by_id[j].duration

                def try_update(j, pred):
                    pred_bound = get_duedate(pred) + get_duration(j)
                    if pred_bound > get_duedate(j):
                        jobs_by_id[j].due_date = pred_bound

                graph = build_instance_graph(self)
                for node in topological_sort(graph):
                    jobs_by_id[node].due_date = 0
                    for predecessor in graph.in_edges(node):
                        try_update(node, predecessor)

                if target_jobs is not None:
                    to_keep = set(target_jobs)
                    for id_job in jobs_by_id:
                        if id_job not in to_keep:
                            jobs_by_id[id_job].due_date = None
            case _:
                print_error("Unrecognized choice type for computing due dates")

        return self

    def complete_jobs(self,
                      jobs_to_complete: list[int] or None = None,
                      choice: Literal["random", "gradual", "combined"] = None,
                      ratio: float = None,
                      combined_ratio: float = None) -> Self:
        """
        Completes jobs in the instance.

        Args:
            jobs_to_complete (list[int] or None, optional): The list of job IDs to complete.
                Defaults to None.
            choice (Literal["random", "gradual", "combined"], optional): The choice type for completing jobs.
                Defaults to None.
            ratio (float, optional): The ratio of jobs to complete.
                Defaults to None.
            combined_ratio (float, optional): The ratio of combined jobs to complete.
                Defaults to None.

        Returns:
            Self: The modified ProblemModifier object.
        """
        def complete(jbs):
            for j in jbs:
                j.completed = True

        def choose_random(count):
            return self._random.sample(self.jobs, count)

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
                             split: Literal["trim source target", "random roots", "paths", "gradual"],
                             gradual_level: int = 1) -> Self:
        """
        Splits job components in the instance.

        Args:
            split (Literal["trim source target", "random roots", "paths", "gradual"]): The type of split to perform.
            gradual_level (int, optional): The depth of the component roots for gradual split.
                Defaults to 1.

        Returns:
            Self: The modified ProblemModifier object.
        """
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
                    root = self._random.choice(list(graph.nodes))
                    subtree = subtree_traversal(graph, root)
                    graph.remove_nodes_from(subtree)
                    subtrees.append(subtree)

                instance_graph = build_instance_graph(self)
                instance_graph: nx.DiGraph = nx.union_all(instance_graph.subgraph(subtree) for subtree in subtrees)

                components = [Component(subtree[0].id_job, 1)
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
            case "gradual":
                graph: nx.DiGraph = build_instance_graph(self, reverse=True)
                topo_gens = list(nx.algorithms.topological_generations(graph))

                # gradual_level indicates the depth of the component roots
                # Right side of the graph picks child edges
                for gen in topo_gens[:gradual_level]:
                    for v in gen:
                        edges = graph.out_edges(v)
                        selected = self._random.choice(list(edges))
                        graph.remove_edges_from(set(edges) - {selected})
                        graph.remove_edges_from(set(graph.in_edges(selected[1])) - {selected})

                # Left side of the graph picks parent edges
                for gen in topo_gens[gradual_level + 1:]:
                    for v in gen:
                        edges = graph.in_edges(v)
                        selected = self._random.choice(list(edges))
                        graph.remove_edges_from(set(edges) - {selected})

                first_gen = nx.topological_generations(graph).send(None)
                components = [Component(root, 1) for root in first_gen]

                self.precedences = [Precedence(v, u) for u, v in graph.edges]
                self.components = components

            case _:
                print_error(f"Unrecognized split option: {split}")
        return self

    def merge_with(self, other: ProblemInstance, target_job: int) -> Self:
        """
        Merges the given problem instance with the instance data currently modified.

        Args:
            other (ProblemInstance): The other problem instance to merge with.
            target_job (int): The target job ID.

        Returns:
            Self: The modified ProblemModifier object.
        """
        other_jobs = [j.copy() for j in other.jobs]
        other_precedences = [p.copy() for p in other.precedences]
        other_components = [c.copy() for c in other.components]

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

        self.target_job = target_job

        return self

    def change_resource_availability(self,
                                     additions: dict[str, Iterable[CapacityChange]],
                                     migrations: dict[str, Iterable[CapacityMigration]],
                                     ) -> Self:
        """
        Changes resource availabilities in the instance.

        Args:
            additions (dict[str, Iterable[CapacityChange]]): The dictionary of capacity changes.
            migrations (dict[str, Iterable[CapacityMigration]]): The dictionary of capacity migrations.

        Returns:
            Self: The modified ProblemModifier object.
        """
        for resource in self._resources:
            if resource.key in additions:
                resource.availability.additions += list_of(additions[resource.key])
                resource.availability.additions.sort(key=lambda a: (a.start, a.end))
            if resource.key in migrations:
                resource.availability.migrations += list_of(migrations[resource.key])
                resource.availability.migrations.sort(key=lambda m: (m.start, m.end))

        return self

    def remove_resources(self, resources_to_remove: Iterable[str]) -> Self:
        """
        Removes resources from the instance.

        Args:
            resources_to_remove (Iterable[str]): The list of resource keys to remove.

        Returns:
            Self: The modified ProblemModifier object.
        """
        resources_to_remove = set(resources_to_remove)

        self._resources = [resource for resource in self._resources if resource.key not in resources_to_remove]
        remaining_resources_count = len(self._resources)

        for job in self._jobs:
            removed_consumption = sum(
                c for r, c in job.resource_consumption.consumption_by_resource.items() if r.key in resources_to_remove)
            remaining_consumption = sum(
                c for r, c in job.resource_consumption.consumption_by_resource.items() if r.key not in resources_to_remove)

            consumption_by_resource = dict()
            for r in job.resource_consumption.consumption_by_resource:
                if r.key not in resources_to_remove:
                    if remaining_consumption == 0:
                        consumption_by_resource[r] = (removed_consumption // len(resources_to_remove))
                    else:
                        consumption_by_resource[r] = job.resource_consumption[r]

            job.resource_consumption.consumption_by_resource = consumption_by_resource

        return self

    def scaledown_job_durations(self, max_duration: int) -> Self:
        """
        Scales down job durations in the instance.

        Args:
            max_duration (int): The maximum duration for job scaling.

        Returns:
            Self: The modified ProblemModifier object.
        """
        max_job_duration = max(j.duration for j in self._jobs)
        if max_job_duration <= max_duration:
            return self

        scale = max_duration / max_job_duration
        for job in self._jobs:
            job.duration = 0 if job.duration == 0 else max(1, min(max_duration, math.floor(scale * job.duration)))

        return self

    def with_target_job(self, target_job: int) -> Self:
        """
        Sets the target job for the instance.

        Args:
            target_job (int): The target job ID.

        Returns:
            Self: The modified ProblemModifier object.
        """
        self.target_job = target_job
        return self

    def generate_modified_instance(self, name: str = None) -> ProblemInstance:
        """
        Generates the modified problem instance.

        Args:
            name (str, optional): The name of the modified instance. Defaults to None.

        Returns:
            ProblemInstance: The modified problem instance.
        """
        builder = InstanceBuilder()
        builder.add_jobs(self.jobs)
        builder.add_precedences(self.precedences)
        builder.add_components(self.components)
        builder.add_resources(self._resources)
        builder.add_projects(self._original_instance.projects)
        builder.set(horizon=self._original_instance.horizon,
                    target_job=self.target_job,
                    name=name if name is not None else f"{self._original_instance.name}_mod"
                    )
        return builder.build_instance()

    @property
    def jobs(self) -> list[Job]:
        """
        Gets the list of jobs.

        Returns:
            list[Job]: The list of jobs.
        """
        return self._jobs

    @jobs.setter
    def jobs(self, value: list[Job]):
        """
        Sets the list of jobs.

        Args:
            value (list[Job]): The list of jobs.
        """
        self._jobs = value

    @property
    def precedences(self) -> list[Precedence]:
        """
        Gets the list of precedences.

        Returns:
            list[Precedence]: The list of precedences.
        """
        return self._precedences

    @precedences.setter
    def precedences(self, value: list[Precedence]):
        """
        Sets the list of precedences.

        Args:
            value (list[Precedence]): The list of precedences.
        """
        self._precedences = value

    @property
    def resources(self) -> list[Resource]:
        """
        Gets the list of resources.

        Returns:
            list[Resource]: The list of resources.
        """
        return self._resources

    @resources.setter
    def resources(self, value: list[Resource]):
        """
        Sets the list of resources.

        Args:
            value (list[Resource]): The list of resources.
        """
        self._resources = value

    @property
    def components(self) -> list[Component]:
        """
        Gets the list of components.

        Returns:
            list[Component]: The list of components.
        """
        return self._components

    @components.setter
    def components(self, value: list[Component]):
        """
        Sets the list of components.

        Args:
            value (list[Component]): The list of components.
        """
        self._components = value


def modify_instance(problem_instance: ProblemInstance) -> ProblemModifier:
    """
    Modifies the given problem instance.
    Returns a ProblemModifier object that can be used to modify the instance.

    Args:
        problem_instance (ProblemInstance): The problem instance to modify.

    Returns:
        ProblemModifier: The ProblemModifier object.
    """
    return ProblemModifier(problem_instance)
