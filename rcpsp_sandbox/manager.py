import os

import instances.io as iio


class ExperimentManager:
    _base_instances_location: str
    _modified_instances_location: str
    _evaluations_location: str

    _base_instances_paths: dict[str, str]

    def __init__(self,
                 base_instance_location: str,
                 modified_instances_location: str,
                 evaluations_location: str,
                 ):
        self._base_instances_location = base_instance_location
        self._modified_instances_location = modified_instances_location
        self._evaluations_location = evaluations_location

        self._base_instances_paths = self.__map_location(base_instance_location)

    def get_base_instance(self, name: str):
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
        else:
            raise FileNotFoundError("Required instance not found")

    @staticmethod
    def __map_location(location: str):
        return {filename: os.path.join(root, filename)
                for root, dirs, files in os.walk(location)
                for filename in files}


