import itertools
import random
from typing import Self

from rcpsp_sandbox.instances.algorithms import traverse_instance_graph, build_instance_graph
from rcpsp_sandbox.instances.problem_instance import ProblemInstance, Job, Precedence
from utils import print_error


class ProblemModifier:
    _original_instance: ProblemInstance

    _new_jobs: list[Job] = []
    _new_precedences: list[Precedence] = []

    def __init__(self,
                 original_instance: ProblemInstance):
        self._original_instance = original_instance

    def assign_job_due_dates(self,
                             choice: str = "uniform",
                             interval: tuple[int, int] = None,
                             target_jobs: list[int] = None,
                             due_dates: dict[int, int] = None,
                             overwrite: bool = False,) -> Self:
        # TODO


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

            case "":  # TODO
                pass

        return self

    def complete_jobs(self,
                      jobs_to_complete: list[int] or None = None,
                      completion_ratio: float or None = None,
                      choice: str = "random",
                      combined_ratio: float = 0.8) -> Self:
        def complete(jbs):
            for j in jbs:
                j.completed = True

        if jobs_to_complete is not None:
            jobs_by_id = {j.id_job: j for j in self.jobs}
            complete(jobs_by_id[id_job] for id_job in jobs_to_complete)
        elif completion_ratio is not None:
            match choice:
                case "random":
                    k = completion_ratio * len(self.jobs)
                    jobs_to_complete = random.sample(self.jobs, k)
                    complete(jobs_to_complete)
                case "gradual":
                    jobs_to_complete_count = round(completion_ratio * len(self.jobs))
                    jobs_to_complete_traverser = traverse_instance_graph(graph=build_instance_graph(self), search="uniform")
                    complete(itertools.islice(jobs_to_complete_traverser, jobs_to_complete_count))
                case "combined":
                    # TODO combined finishing
                    pass
                case _:
                    print_error(f"Unrecognized job-completion choice: {choice}")
        else:
            print_error("No jobs to complete were given and neither a ratio for finished job was given.")

        return self

    @property
    def jobs(self):
        return self._original_instance.jobs[:] + self._new_jobs

    def precedences(self):
        return self._original_instance.precedences[:] + self._new_precedences


def modify_instance(problem_instance: ProblemInstance) -> ProblemModifier:
    return ProblemModifier(problem_instance)

