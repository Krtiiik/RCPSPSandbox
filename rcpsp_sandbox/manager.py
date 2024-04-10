import itertools
import os
from collections import defaultdict
from typing import Iterable

import bottlenecks.io as bio
import instances.io as iio
from bottlenecks.evaluations import Evaluation, EvaluationKPIs, EvaluationAlgorithm, EvaluationLightweight, \
    EvaluationKPIsLightweight, evaluation_alg_string, evaluation_settings_string
from generate_instances import experiment_instances, build_instance
from instances.problem_instance import ProblemInstance


class ExperimentManager:
    _base_instances_location: str
    _modified_instances_location: str
    _evaluations_location: str
    _evaluations_kpis_location: str

    _base_instances_cache: dict[str, ProblemInstance]
    _modified_instances_cache: dict[str, ProblemInstance]
    _evaluations_light_cache: dict[str, dict[str, EvaluationLightweight]]
    _evaluations_cache: dict[str, Evaluation]
    _evaluations_kpis_light_cache: dict[str, dict[str, EvaluationKPIsLightweight]]
    _evaluations_kpis_cache: dict[str, EvaluationKPIs]

    def __init__(self,
                 base_instances_location: str,
                 modified_instances_location: str,
                 evaluations_location: str,
                 evaluations_kpis_location: str,
                 ):
        self._base_instances_location = base_instances_location
        self._modified_instances_location = modified_instances_location
        self._evaluations_location = evaluations_location
        self._evaluations_kpis_location = evaluations_kpis_location

        self._base_instances_cache = dict()
        self._modified_instances_cache = dict()
        self._evaluations_light_cache = dict()
        self._evaluations_cache = dict()
        self._evaluations_kpis_light_cache = dict()
        self._evaluations_kpis_cache = dict()

    # ~~ Instances ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load_base_instance(self, name: str) -> ProblemInstance:
        if name in self._base_instances_cache:
            return self._base_instances_cache[name]
        elif name in experiment_instances:
            instance = build_instance(name, self._base_instances_location, self._base_instances_location)
        else:
            filename = os.path.join(self._base_instances_location, name+'.json')
            if not os.path.exists(filename):
                raise FileNotFoundError(f'Required base instance "{name}" not found in [{filename}]')

            instance = iio.parse_json(filename, name_as=name, is_extended=True)

        self._base_instances_cache[name] = instance
        return instance

    def load_base_instances(self, names: Iterable[str]) -> list[ProblemInstance]:
        return list(map(self.load_base_instance, names))

    def load_modified_instance(self, name: str) -> ProblemInstance:
        if name in self._modified_instances_cache:
            return self._modified_instances_cache[name]
        else:
            filename = os.path.join(self._modified_instances_location, name+'.json')
            if not os.path.exists(filename):
                raise FileNotFoundError(f'Required modified instance "{name}" not found in [{filename}]')

            instance = iio.parse_json(filename, name_as=name, is_extended=True)

        self._modified_instances_cache[name] = instance
        return instance

    def load_modified_instances(self, names: Iterable[str]) -> list[ProblemInstance]:
        return list(map(self.load_modified_instance, names))

    def save_modified_instance(self, instance: ProblemInstance):
        self._modified_instances_cache[instance.name] = instance

    def save_modified_instances(self, instances: Iterable[ProblemInstance]):
        for instance in instances:
            self.save_modified_instance(instance)

    # ~~ Evaluations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load_evaluation(self, evaluation_id: str) -> Evaluation:
        if evaluation_id in self._evaluations_cache:
            return self._evaluations_cache[evaluation_id]

        alg, settings = evaluation_alg_string(evaluation_id), evaluation_settings_string(evaluation_id)
        if alg not in self._evaluations_light_cache:  # Load alg light evaluations into cache
            self._evaluations_light_cache[alg] = bio.parse_evaluations(os.path.join(self._evaluations_location, alg+'.json'))

        if settings not in self._evaluations_light_cache:  # The requested evaluation has not yet been computed or saved
            raise ValueError("Requested evaluation could not be found.")

        evaluation_light = self._evaluations_light_cache[alg][settings]
        evaluation = evaluation_light.build_full_evaluation(self.load_base_instance(evaluation_light.base_instance),
                                                            self.load_modified_instance(evaluation_light.modified_instance))
        self._evaluations_cache[evaluation_id] = evaluation
        return evaluation

    def load_evaluations(self, evaluation_ids: Iterable[str]) -> dict[str, dict[str, Evaluation]]:
        evaluations = defaultdict(dict)
        for evaluation_id in evaluation_ids:
            alg, settings = evaluation_alg_string(evaluation_id), evaluation_settings_string(evaluation_id)
            evaluations[alg][settings] = self.load_evaluation(evaluation_id)

        return evaluations

    def load_evaluation_kpis(self, evaluation_kpis_id: str) -> EvaluationKPIs:
        if evaluation_kpis_id in self._evaluations_kpis_cache:
            return self._evaluations_kpis_cache[evaluation_kpis_id]

        alg, settings = evaluation_alg_string(evaluation_kpis_id), evaluation_settings_string(evaluation_kpis_id)
        if alg not in self._evaluations_kpis_light_cache:
            self._evaluations_kpis_light_cache[alg] = bio.parse_evaluations_kpis(os.path.join(self._evaluations_kpis_location, alg+'.json'))

        if settings not in self._evaluations_kpis_light_cache[alg]:
            raise ValueError("Requested evaluation could not be found.")

        evaluation_kpis_light = self._evaluations_kpis_light_cache[alg][settings]
        evaluation_kpis = evaluation_kpis_light.build_full_evaluation_kpis(self.load_base_instance(evaluation_kpis_light.evaluation.base_instance),
                                                                           self.load_modified_instance(evaluation_kpis_light.evaluation.modified_instance))
        self._evaluations_kpis_cache[evaluation_kpis_id] = evaluation_kpis
        return evaluation_kpis

    def load_evaluations_kpis(self, evaluations_kpis_ids: Iterable[str]) -> Iterable[EvaluationKPIs]:
        evaluations_kpis = defaultdict(dict)
        for evaluation_kpis_id in evaluations_kpis_ids:
            alg, settings = evaluation_alg_string(evaluation_kpis_id), evaluation_settings_string(evaluation_kpis_id)
            evaluations_kpis[alg][settings] = self.load_evaluation_kpis(evaluation_kpis_id)

        return evaluations_kpis

    def save_evaluation(self, evaluation: Evaluation):
        self._evaluations_cache[evaluation.by] = evaluation

    def save_evaluations(self, evaluations: Iterable[Evaluation]):
        for evaluation in evaluations:
            self.save_evaluation(evaluation)

    def save_evaluation_kpis(self, evaluation_kpis: EvaluationKPIs):
        self._evaluations_kpis_cache[evaluation_kpis.evaluation.by] = evaluation_kpis

    def save_evaluations_kpis(self, evaluations_kpis: Iterable[EvaluationKPIs]):
        for evaluation_kpis in evaluations_kpis:
            self.save_evaluation_kpis(evaluation_kpis)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for modified_instance in self._modified_instances_cache.values():
            iio.serialize_json(modified_instance, os.path.join(self._modified_instances_location, modified_instance.name+'.json'), is_extended=True)
        bio.serialize_evaluations(self._evaluations_cache.values(), self._evaluations_location)
        bio.serialize_evaluations_kpis(self._evaluations_kpis_cache.values(), self._evaluations_kpis_location)
