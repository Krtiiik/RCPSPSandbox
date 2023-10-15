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
                             overwrite: bool = False,
                             due_dates: dict[int, int] or None = None,
                             interval: tuple[int, int] or None = None,
                             choice = "uniform") -> Self:
        if due_dates is not None:
            jobs_by_id = {job.id_job: job for job in self.jobs}
            for id_job, duedate in due_dates.items():
                if jobs_by_id[id_job].due_date is None or overwrite:
                    jobs_by_id[id_job].due_date = duedate
            # TODO what if some jobs are not specified in `due_dates` dict

        if interval is not None:
            # TODO generate due dates in the given interval
            # TODO random kind: uniform, normal, earliest
            pass

        return self

    def complete_jobs(self,
                      jobs_to_complete: list[int] or None = None,
                      finished_ratio: float or None = None,
                      choice: str = "random",
                      combined_ratio: float = 0.8) -> Self:
        if jobs_to_complete is not None:
            jobs_by_id = {j.id_job: j for j in self.jobs}
            for id_job in jobs_to_complete:
                jobs_by_id[id_job].finished = True
        elif finished_ratio is not None:
            match choice:
                case "random":
                    k = finished_ratio * len(self.jobs)
                    jobs_to_complete = random.sample(self.jobs, k)
                case "gradual":
                    jobs_to_complete_count = round(finished_ratio * len(self.jobs))
                    jobs_to_complete_traverser = traverse_instance_graph(graph=build_instance_graph(self), search="uniform")
                    for job in itertools.islice(jobs_to_complete_traverser, jobs_to_complete_count):
                        job.completed = True
                case "combined":
                    # TODO combined finishing
                    pass
                case _:
                    print_error(f"Unrecognized job-completion choice: {choice}")
        else:
            print_error("No ")

        return self

    @property
    def jobs(self):
        return self._original_instance.jobs[:] + self._new_jobs

    def precedences(self):
        return self._original_instance.precedences[:] + self._new_precedences


def modify_instance(problem_instance: ProblemInstance) -> ProblemModifier:
    return ProblemModifier(problem_instance)

