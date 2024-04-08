import json
from typing import Iterable

from bottlenecks.evaluations import Evaluation, EvaluationLightweight, EvaluationKPIs, EvaluationKPIsLightweight
from solver.solution import Solution
from utils import try_open_read


def serialize_evaluations(evaluations: Iterable[Evaluation],
                          filename: str,
                          ):
    evaluations_obj = list(map(__serialize_evaluation, evaluations))

    json_str = json.dumps(evaluations_obj)
    with open(filename, "wt") as file:
        file.write(json_str)


def serialize_evaluations_kpis(evaluations_kpis: Iterable[EvaluationKPIs],
                              filename: str,
                              ):
    def serialize_evaluation_kpis(_evaluation_kpis):
        return {
            "evaluation": __serialize_evaluation(_evaluation_kpis.evaluation),
            "cost": _evaluation_kpis.cost,
            "improvement": _evaluation_kpis.improvement,
        }

    evaluations_kpis_obj = list(map(serialize_evaluation_kpis, evaluations_kpis))

    json_str = json.dumps(evaluations_kpis_obj)
    with open(filename, "wt") as file:
        file.write(json_str)


def parse_evaluations(filename: str) -> list[EvaluationLightweight]:
    evaluations_obj = try_open_read(filename, json.load)
    evaluations = list(map(__parse_evaluation, evaluations_obj))
    return evaluations


def parse_evaluations_kpis(filename: str) -> list[EvaluationKPIsLightweight]:
    def parse_evaluation_kpis(_evaluation_kpis):
        return EvaluationKPIsLightweight(
            evaluation=__parse_evaluation(_evaluation_kpis["evaluation"]),
            cost=int(_evaluation_kpis["cost"]),
            improvement=int(_evaluation_kpis["improvement"]),
        )

    evaluations_kpis_obj = try_open_read(filename, json.load)
    evaluations_kpis = list(map(parse_evaluation_kpis, evaluations_kpis_obj))
    return evaluations_kpis


def __serialize_evaluation(evaluation):
    def serialize_solution(_solution: Solution):
        return [{
            "job_id": _job_id,
            "start": _interval_solution.start,
            "end": _interval_solution.end,
        } for _job_id, _interval_solution in _solution.job_interval_solutions.items()]

    return {
        "base_instance": evaluation.base_instance.name,
        "base_solution": serialize_solution(evaluation.base_solution),
        "target_job": evaluation.target_job,
        "modified_instance": evaluation.modified_instance.name,
        "solution": serialize_solution(evaluation.solution),
        "by": evaluation.by,
        "duration": evaluation.duration,
    }


def __parse_evaluation(evaluation):
    def parse_solution(_solution):
        return {int(_int_sol["job_id"]): (int(_int_sol["start"]), int(_int_sol["end"])) for _int_sol in _solution}

    return EvaluationLightweight(
        base_instance=evaluation["base_instance"],
        base_solution=parse_solution(evaluation["base_solution"]),
        target_job=int(evaluation["target_job"]),
        modified_instance=evaluation["modified_instance"],
        solution=parse_solution(evaluation["solution"]),
        by=evaluation["by"],
        duration=float(evaluation["duration"]),
    )
