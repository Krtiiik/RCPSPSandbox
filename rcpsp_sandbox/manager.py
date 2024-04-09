import itertools
import os
from typing import Iterable

import instances.io as iio
from generate_instances import experiment_instances, build_instance
from instances.problem_instance import ProblemInstance


class ExperimentManager:
    _base_instances_location: str
    _modified_instances_location: str
    _evaluations_location: str

    _base_instances_paths: dict[str, str]
    _modified_instances_paths: dict[str, str]

    def __init__(self,
                 base_instance_location: str,
                 modified_instances_location: str,
                 evaluations_location: str,
                 ):
        self._base_instances_location = base_instance_location
        self._modified_instances_location = modified_instances_location
        self._evaluations_location = evaluations_location

        self._base_instances_paths = self.__map_location(base_instance_location)
        self._modified_instances_paths = self.__map_location(self._modified_instances_location)

    def load_base_instance(self, name: str) -> ProblemInstance:
        def psplib(filename): return iio.parse_psplib(filename, name, is_extended=True)
        def json(filename): return iio.parse_json(filename, name, is_extended=True)

        if name in self._base_instances_paths:
            if '.sm' in name:
                return psplib(name)
            elif '.json' in name:
                return json(name)
            else:
                raise FileNotFoundError("Unable to deduce required instance format")
        elif f'{name}.sm' in self._base_instances_paths:
            return psplib(f'{name}.sm')
        elif f'{name}.json' in self._base_instances_paths:
            return json(f'{name}.json')
        elif name in experiment_instances:
            instance = build_instance(name, self._base_instances_location, self._base_instances_location)
            self._base_instances_paths = self.__map_location(self._base_instances_location)
            return instance
        else:
            raise FileNotFoundError("Required instance not found")

    def load_base_instances(self, names: Iterable[str]) -> list[ProblemInstance]:
        return list(map(self.load_base_instance, names))

    def load_modified_instance(self, name: str) -> ProblemInstance:
        def psplib(filename): return iio.parse_psplib(filename, name, is_extended=True)
        def json(filename): return iio.parse_json(filename, name, is_extended=True)

        if name in self._modified_instances_paths:
            if name in self._base_instances_paths:
                if '.sm' in name:
                    return psplib(name)
                elif '.json' in name:
                    return json(name)
                else:
                    raise FileNotFoundError("Unable to deduce required instance format")
        elif f'{name}.sm' in self._base_instances_paths:
            return psplib(f'{name}.sm')
        elif f'{name}.json' in self._base_instances_paths:
            return json(f'{name}.json')
        else:
            raise FileNotFoundError("Required instance not found")

    def load_modified_instances(self, names: Iterable[str]) -> list[ProblemInstance]:
        return list(map(self.load_modified_instance, names))

    def save_modified_instance(self, instance: ProblemInstance, save_as: str = None):
        if save_as is None:
            save_as = instance.name
        if not save_as.endswith('.json'):
            save_as += '.json'

        iio.serialize_json(instance, os.path.join(self._modified_instances_location, save_as), is_extended=True)
        self._modified_instances_paths = self.__map_location(self._modified_instances_location)

    def save_modified_instances(self, instances: Iterable[ProblemInstance], save_as: Iterable[str | None] = None):
        if save_as is None:
            save_as = itertools.repeat(None)

        for instance, save_as_name in zip(instances, save_as, strict=True):
            self.save_modified_instance(instance, save_as)

    @staticmethod
    def __map_location(location: str):
        return {filename: os.path.join(root, filename)
                for root, dirs, files in os.walk(location)
                for filename in files}
