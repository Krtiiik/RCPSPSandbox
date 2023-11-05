import itertools
import random
from typing import Self, Literal

from rcpsp_sandbox.instances.algorithms import traverse_instance_graph, build_instance_graph, topological_sort, paths_traversal
from rcpsp_sandbox.instances.problem_instance import ProblemInstance, Job, Precedence, Component
from utils import print_error


class ProblemModifier:
    _original_instance: ProblemInstance

    _jobs: list[Job] = []
    _precedences: list[Precedence] = []
    _components: list[Component] = []

    def __init__(self,
                 original_instance: ProblemInstance):
        self._original_instance = original_instance

        self._jobs = original_instance.jobs[:]
        self._precedences = original_instance.precedences[:]
        self._components = original_instance.components[:]

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
                    due_date = random.uniform(interval[0], interval[1])
                    try_assign(job, due_date)

            case "gradual":
                if gradual_interval is None:
                    gradual_interval = [0,0]
                for node, parent in topological_sort(build_instance_graph(self), yield_state=True):
                    parent_end = parent.due_date if parent is not None else gradual_base
                    node.due_date = parent_end + node.duration + random.uniform(*gradual_interval)
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
                # TODO remove precedences leading from super-source and super-target
                pass
            case "random roots":
                # TODO pick uniform random node and remove its children tree as a component
                pass
            case "paths":
                # TODO traverse paths
                paths = paths_traversal(build_instance_graph(self))
                precedences = [Precedence(child.id_job, parent.id_job)
                               for path in paths
                               for child, parent in zip(path, path[1:])]
                components = [Component(path[0].id_job, 0) for path in paths]  # Component weight is not set, can be set manually

                self.precedences = precedences
                self.components = components
                pass
            case _:
                print_error("Unrecognized split option")
        return self

    def generate_modified_instance(self) -> ProblemInstance:
        # TODO implement via InstanceBuilder
        return None

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
    def components(self) -> list[Component]:
        return self._components

    @components.setter
    def components(self, value: list[Component]):
        self._components = value

def modify_instance(problem_instance: ProblemInstance) -> ProblemModifier:
    return ProblemModifier(problem_instance)
