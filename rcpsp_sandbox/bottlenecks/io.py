import json
from typing import Iterable

from bottlenecks.evaluations import Evaluation, EvaluationLightweight
from solver.solution import Solution
from utils import try_open_read


def serialize_evaluations(evaluations: Iterable[Evaluation],
                          filename: str,
                          ):
    def serialize_solution(_solution: Solution):
        return [{
            "job_id": _job_id,
            "start": _interval_solution.start,
            "end": _interval_solution.end,
        } for _job_id, _interval_solution in _solution.job_interval_solutions.items()]

    def serialize_evaluation(_evaluation):
        return {
            "base_instance": _evaluation.base_instance.name,
            "base_solution": serialize_solution(_evaluation.base_solution),
            "target_job": _evaluation.target_job,
            "modified_instance": _evaluation.modified_instance.name,
            "solution": serialize_solution(_evaluation.solution),
            "by": _evaluation.by,
            "duration": _evaluation.duration,
        }

    evaluations_obj = list(map(serialize_evaluation, evaluations))

    json_str = json.dumps(evaluations_obj)
    with open(filename, "wt") as file:
        file.write(json_str)


def parse_evaluations(filename: str) -> list[EvaluationLightweight]:
    def parse_solution(_solution):
        return {int(_int_sol["job_id"]): (int(_int_sol["start"]), int(_int_sol["end"])) for _int_sol in _solution}

    def parse_evaluation(_evaluation):
        return EvaluationLightweight(
            base_instance=_evaluation["base_instance"],
            base_solution=parse_solution(_evaluation["base_solution"]),
            target_job=int(_evaluation["target_job"]),
            modified_instance=_evaluation["modified_instance"],
            solution=parse_solution(_evaluation["solution"]),
            by=_evaluation["by"],
            duration=float(_evaluation["duration"]),
        )

    evaluations_obj = try_open_read(filename, json.load)
    evaluations = list(map(parse_evaluation, evaluations_obj))
    return evaluations
