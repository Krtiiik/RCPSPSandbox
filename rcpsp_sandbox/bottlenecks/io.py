import json
import os
from collections import defaultdict
from typing import Iterable

from bottlenecks.evaluations import Evaluation, EvaluationLightweight, EvaluationKPIs, EvaluationKPIsLightweight, \
    EvaluationAlgorithm
from solver.solution import Solution
from utils import try_open_read


def serialize_evaluations(evaluations: Iterable[Evaluation],
                          location: str,
                          ):
    evaluations_by_alg = defaultdict(list)
    for evaluation in evaluations:
        alg = evaluation.alg_string
        evaluations_by_alg[alg].append(evaluation)

    for alg, alg_evaluations in evaluations_by_alg.items():
        evaluations_obj = {evaluation.settings_string: __serialize_evaluation(evaluation)
                           for evaluation in alg_evaluations}
        json_str = json.dumps(evaluations_obj)
        with open(os.path.join(location, alg+'.json'), "wt") as file:
            file.write(json_str)


def serialize_evaluations_kpis(evaluations_kpis: Iterable[EvaluationKPIs],
                               location: str,
                               ):
    def serialize_evaluation_kpis(_evaluation_kpis):
        return {
            "evaluation": __serialize_evaluation(_evaluation_kpis.evaluation),
            "cost": _evaluation_kpis.cost,
            "improvement": _evaluation_kpis.improvement,
        }

    evaluations_kpis_by_alg = defaultdict(list)
    for evaluation_kpis in evaluations_kpis:
        alg = evaluation_kpis.evaluation.alg_string
        evaluations_kpis_by_alg[alg].append(evaluation_kpis)

    for alg, alg_evaluations_kpis in evaluations_kpis_by_alg.items():
        evaluations_kpis_obj = {evaluation_kpis.evaluation.settings_string: serialize_evaluation_kpis(evaluation_kpis)
                                for evaluation_kpis in alg_evaluations_kpis}
        json_str = json.dumps(evaluations_kpis_obj)
        with open(os.path.join(location, alg+'.json'), "wt") as file:
            file.write(json_str)


def parse_evaluations(filename: str) -> dict[str, EvaluationLightweight]:
    evaluations_obj = try_open_read(filename, json.load)
    evaluations = {settings: __parse_evaluation(evaluation_obj)
                   for settings, evaluation_obj in evaluations_obj.items()}
    return evaluations


def parse_evaluations_kpis(filename: str) -> dict[str, EvaluationKPIsLightweight]:
    def parse_evaluation_kpis(_evaluation_kpis):
        return EvaluationKPIsLightweight(
            evaluation=__parse_evaluation(_evaluation_kpis["evaluation"]),
            cost=int(_evaluation_kpis["cost"]),
            improvement=int(_evaluation_kpis["improvement"]),
        )

    evaluations_kpis_obj = try_open_read(filename, json.load)
    evaluations_kpis = {settings: parse_evaluation_kpis(evaluation_kpis_obj)
                        for settings, evaluation_kpis_obj in evaluations_kpis_obj.items()}
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
