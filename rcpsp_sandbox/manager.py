# Experiment manager for the experiments

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
    """
    Class for managing experiment instances, evaluations, and evaluation KPIs.
    Base experiment instances can be loaded or created from the `generate_instances` module.
    Evaluations and evaluation KPIs can be loaded or saved.
    """
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
        """
        Initializes a Manager object.

        Args:
            base_instances_location (str): The location of the base instances.
            modified_instances_location (str): The location of the modified instances.
            evaluations_location (str): The location of the evaluations.
            evaluations_kpis_location (str): The location of the evaluation KPIs.
        """
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
        """
        Load a base instance by name.

        Args:
            name (str): The name of the base instance to load.

        Returns:
            ProblemInstance: The loaded base instance.

        Raises:
            FileNotFoundError: If the required base instance is not found.

        """
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
        """
        Load multiple base instances given their names.

        Args:
            names (Iterable[str]): An iterable of names of the base instances to load.

        Returns:
            list[ProblemInstance]: A list of loaded ProblemInstance objects.
        """
        return list(map(self.load_base_instance, names))

    def load_modified_instance(self, name: str) -> ProblemInstance:
        """
        Loads a modified instance by name.

        Args:
            name (str): The name of the modified instance to load.

        Returns:
            ProblemInstance: The loaded modified instance.

        Raises:
            FileNotFoundError: If the required modified instance is not found.
        """
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
        """
        Load a list of modified problem instances.

        Args:
            names (Iterable[str]): A list of names of the modified instances.

        Returns:
            list[ProblemInstance]: A list of loaded modified problem instances.
        """
        return list(map(self.load_modified_instance, names))

    def save_modified_instance(self, instance: ProblemInstance):
        """
        Saves a modified instance to the cache.

        Args:
            instance (ProblemInstance): The modified instance to be saved.

        Returns:
            None
        """
        self._modified_instances_cache[instance.name] = instance

    def save_modified_instances(self, instances: Iterable[ProblemInstance]):
        """
        Saves the modified instances.

        Args:
            instances (Iterable[ProblemInstance]): A collection of modified problem instances.

        Returns:
            None
        """
        for instance in instances:
            self.save_modified_instance(instance)

    # ~~ Evaluations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load_evaluation(self, instance_name: str, evaluation_id: str) -> Evaluation:
        """
        Load and return the evaluation for a given instance and evaluation ID.

        Args:
            instance_name (str): The name of the instance.
            evaluation_id (str): The ID of the evaluation.

        Returns:
            Evaluation: The loaded evaluation.

        Raises:
            ValueError: If the requested evaluation could not be found.
        """
        _, settings, inst_alg, inst_alg_sett = get_inst_alg_sett(instance_name, evaluation_id)

        if inst_alg_sett in self._evaluations_cache:
            return self._evaluations_cache[inst_alg_sett]

        if settings not in self._evaluations_light_cache[inst_alg]:  # The requested evaluation has not yet been computed or saved
            raise ValueError("Requested evaluation could not be found.")

        evaluation_light = self._evaluations_light_cache[inst_alg][settings]
        evaluation = evaluation_light.build_full_evaluation(self.load_base_instance(evaluation_light.base_instance),
                                                            self.load_modified_instance(evaluation_light.modified_instance))
        self._evaluations_cache[inst_alg_sett] = evaluation
        return evaluation

    def load_evaluations(self, instance_name: str, evaluation_ids: Iterable[str]) -> dict[str, dict[str, Evaluation]]:
        """
        Load evaluations for a given instance and evaluation IDs.

        Args:
            instance_name (str): The name of the instance.
            evaluation_ids (Iterable[str]): An iterable of evaluation IDs.

        Returns:
            dict[str, dict[str, Evaluation]]: A dictionary containing evaluations grouped by algorithm and settings.
        """
        evaluations = defaultdict(dict)
        for evaluation_id in evaluation_ids:
            alg, settings = evaluation_alg_string(evaluation_id), evaluation_settings_string(evaluation_id)
            evaluations[alg][settings] = self.load_evaluation(instance_name, evaluation_id)

        return evaluations

    def load_evaluation_kpis(self, instance_name: str, evaluation_kpis_id: str) -> EvaluationKPIs:
        """
        Load the evaluation KPIs for a given instance and evaluation KPIs ID.

        Args:
            instance_name (str): The name of the instance.
            evaluation_kpis_id (str): The ID of the evaluation KPIs.

        Returns:
            EvaluationKPIs: The loaded evaluation KPIs.

        Raises:
            ValueError: If the requested evaluation could not be found.

        """
        _, settings, inst_alg, inst_alg_sett = get_inst_alg_sett(instance_name, evaluation_kpis_id)

        if inst_alg_sett in self._evaluations_kpis_cache:
            return self._evaluations_kpis_cache[inst_alg_sett]

        if settings not in self._evaluations_kpis_light_cache[inst_alg]:
            raise ValueError("Requested evaluation could not be found.")

        evaluation_kpis_light = self._evaluations_kpis_light_cache[inst_alg][settings]
        evaluation_kpis = evaluation_kpis_light.build_full_evaluation_kpis(self.load_base_instance(evaluation_kpis_light.evaluation.base_instance),
                                                                           self.load_modified_instance(evaluation_kpis_light.evaluation.modified_instance))
        self._evaluations_kpis_cache[inst_alg_sett] = evaluation_kpis
        return evaluation_kpis

    def load_evaluations_kpis(self, instance_name: str, evaluations_kpis_ids: Iterable[str]) -> Iterable[EvaluationKPIs]:
        """
        Load evaluation KPIs for a given instance and a list of evaluation KPIs IDs.

        Args:
            instance_name (str): The name of the instance.
            evaluations_kpis_ids (Iterable[str]): An iterable of evaluation KPIs IDs.

        Returns:
            Iterable[EvaluationKPIs]: A dictionary of evaluation KPIs, where the keys are algorithm and settings,
            and the values are the loaded evaluation KPIs.

        """
        evaluations_kpis = defaultdict(dict)
        for evaluation_kpis_id in evaluations_kpis_ids:
            alg, settings = evaluation_alg_string(evaluation_kpis_id), evaluation_settings_string(evaluation_kpis_id)
            evaluations_kpis[alg][settings] = self.load_evaluation_kpis(instance_name, evaluation_kpis_id)

        return evaluations_kpis

    def save_evaluation(self, evaluation: Evaluation):
        """
        Saves the given evaluation in the evaluations cache.

        Parameters:
        - evaluation: The evaluation object to be saved.
        """
        inst_alg_sett = f'{evaluation.base_instance.name}{EvaluationAlgorithm.ID_SEPARATOR}{evaluation.by}'
        self._evaluations_cache[inst_alg_sett] = evaluation

    def save_evaluations(self, evaluations: Iterable[Evaluation]):
        """
        Saves a collection of evaluations.

        Args:
            evaluations (Iterable[Evaluation]): The evaluations to be saved.

        Returns:
            None
        """
        for evaluation in evaluations:
            self.save_evaluation(evaluation)

    def save_evaluation_kpis(self, evaluation_kpis: EvaluationKPIs):
        """
        Saves the evaluation KPIs to the cache.

        Parameters:
        - evaluation_kpis: An instance of the EvaluationKPIs class containing the evaluation KPIs to be saved.

        Returns:
        None
        """
        inst_alg_sett = f'{evaluation_kpis.evaluation.base_instance.name}{EvaluationAlgorithm.ID_SEPARATOR}{evaluation_kpis.evaluation.by}'
        self._evaluations_kpis_cache[inst_alg_sett] = evaluation_kpis

    def save_evaluations_kpis(self, evaluations_kpis: Iterable[EvaluationKPIs]):
        """
        Saves the evaluations KPIs for each evaluation.

        Args:
            evaluations_kpis (Iterable[EvaluationKPIs]): An iterable of EvaluationKPIs objects.

        Returns:
            None
        """
        for evaluation_kpis in evaluations_kpis:
            self.save_evaluation_kpis(evaluation_kpis)

    def is_evaluation_cached(self, instance_name: str, evaluation_id: str) -> bool:
        """
        Checks if the evaluation for a given instance and evaluation ID is cached.

        Args:
            instance_name (str): The name of the instance.
            evaluation_id (str): The ID of the evaluation.

        Returns:
            bool: True if the evaluation is cached, False otherwise.
        """
        _, settings, inst_alg, inst_alg_sett = get_inst_alg_sett(instance_name, evaluation_id)

        return (inst_alg_sett in self._evaluations_cache
                or (inst_alg in self._evaluations_light_cache
                    and settings in self._evaluations_light_cache[inst_alg]))

    def is_evaluation_kpis_cached(self, instance_name: str, evaluation_kpis_id: str) -> bool:
        """
        Checks if the evaluation KPIs are cached for a given instance and evaluation KPIs ID.

        Args:
            instance_name (str): The name of the instance.
            evaluation_kpis_id (str): The ID of the evaluation KPIs.

        Returns:
            bool: True if the evaluation KPIs are cached, False otherwise.
        """
        _, settings, inst_alg, inst_alg_sett = get_inst_alg_sett(instance_name, evaluation_kpis_id)

        return (inst_alg_sett in self._evaluations_kpis_cache
                or (inst_alg in self._evaluations_kpis_light_cache
                    and settings in self._evaluations_kpis_light_cache[inst_alg]))

    def __update_evaluations_light_cache(self, inst_alg: str):
        bio.serialize_evaluations(self._evaluations_cache.values(), self._evaluations_location)
        inst_alg_filename = os.path.join(self._evaluations_location, inst_alg+'.json')
        if os.path.exists(inst_alg_filename):
            self._evaluations_light_cache[inst_alg] = bio.parse_evaluations(inst_alg_filename)

    def __update_evaluations_kpis_light_cache(self, inst_alg: str):
        bio.serialize_evaluations_kpis(self._evaluations_kpis_cache.values(), self._evaluations_kpis_location)
        inst_alg_filename = os.path.join(self._evaluations_kpis_location, inst_alg+'.json')
        if os.path.exists(inst_alg_filename):
            self._evaluations_kpis_light_cache[inst_alg] = bio.parse_evaluations_kpis(inst_alg_filename)

    def load_evaluations_caches(self, instance_name, algorithm_name):
        """
        Loads the evaluations caches for a given instance and algorithm.

        Parameters:
        - instance_name (str): The name of the instance.
        - algorithm_name (str): The name of the algorithm.

        Returns:
        None
        """
        inst_alg = get_inst_alg(instance_name, algorithm_name)
        self.__update_evaluations_light_cache(inst_alg)
        self.__update_evaluations_kpis_light_cache(inst_alg)

    def clear_caches(self):
        """
        Clears all the caches used by the manager.
        """
        self._base_instances_cache.clear()
        self._modified_instances_cache.clear()
        self._evaluations_light_cache.clear()
        self._evaluations_cache.clear()
        self._evaluations_kpis_light_cache.clear()
        self._evaluations_kpis_cache.clear()

    def flush(self):
        """
        Flushes the modified instances, evaluations, and evaluation KPIs caches to disk.

        This method serializes the modified instances, evaluations, and evaluation KPIs caches
        to their respective locations on disk. If an error occurs during serialization, it is
        silently ignored.

        Note: The modified instances are serialized as JSON files with the '.json' extension.

        Returns:
            None
        """
        for modified_instance in self._modified_instances_cache.values():
            try:
                iio.serialize_json(modified_instance,
                                   os.path.join(self._modified_instances_location, modified_instance.name+'.json'),
                                   is_extended=True)
            except:
                pass
        try:
            bio.serialize_evaluations(self._evaluations_cache.values(), self._evaluations_location)
        except:
            pass
        try:
            bio.serialize_evaluations_kpis(self._evaluations_kpis_cache.values(), self._evaluations_kpis_location)
        except:
            pass

    def __enter__(self):
        return self

    # noinspection PyBroadException
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


def get_inst_alg(instance_name, algorithm_name):
    """
    Returns a string representation of the instance and algorithm names.

    Args:
        instance_name (str): The name of the instance.
        algorithm_name (str): The name of the algorithm.

    Returns:
        str: A string representation of the instance and algorithm names.
    """
    return f'{instance_name}{EvaluationAlgorithm.ID_SEPARATOR}{algorithm_name}'


def get_inst_alg_sett(instance_name, evaluation_id):
    """
    Get the algorithm, settings, instance algorithm, and instance algorithm settings string representations.

    Args:
        instance_name (str): The name of the instance.
        evaluation_id (int): The evaluation ID.

    Returns:
        tuple: A tuple containing the algorithm, settings, instance algorithm, and instance algorithm settings string representations.
    """
    alg = evaluation_alg_string(evaluation_id)
    settings = evaluation_settings_string(evaluation_id)
    inst_alg_sett = f'{instance_name}{EvaluationAlgorithm.ID_SEPARATOR}{evaluation_id}'
    inst_alg = get_inst_alg(instance_name, alg)

    return alg, settings, inst_alg, inst_alg_sett
