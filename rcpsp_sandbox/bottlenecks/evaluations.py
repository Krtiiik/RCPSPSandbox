import abc
from collections import namedtuple, defaultdict
from typing import Iterable

from bottlenecks.drawing import plot_solution
from bottlenecks.utils import compute_resource_availability, compute_resource_consumption
from instances.problem_instance import ProblemInstance
from solver.solution import Solution
from utils import interval_overlap_function

CapacityChange = namedtuple("CapacityChange", ("start", "end", "capacity"))
CapacityMigration = namedtuple("CapacityMigration", ("ResourceKey", "start", "end", "capacity"))


ProblemSetup = namedtuple("ProblemSetup", ("instance", "target_job"))


class Evaluation:
    _base_instance: ProblemInstance
    _modified_instance: ProblemInstance
    _solution: Solution
    _duration: float

    _capacity_migrations: dict[str, Iterable[CapacityMigration]]
    _capacity_additions: dict[str, Iterable[CapacityChange]]

    def __init__(self, base_instance: ProblemInstance, modified_instance: ProblemInstance, solution: Solution, duration: float):
        self._base_instance = base_instance
        self._modified_instance = modified_instance
        self._solution = solution
        self._duration = duration

        self.__compute_base_migrations()
        self.__compute_reduced_capacity_changes()

    @property
    def instance(self) -> ProblemInstance:
        return self._base_instance

    @property
    def capacity_migrations(self) -> dict[str, Iterable[CapacityMigration]]:
        return self._capacity_migrations

    @property
    def capacity_additions(self) -> dict[str, Iterable[CapacityChange]]:
        return self._capacity_additions

    @property
    def duration(self) -> float:
        return self._duration

    def plot(self, block: bool = True, save_as: str = None, dimensions: tuple[int, int] = (8, 11)):
        plot_solution(self._solution, block=block, save_as=save_as, dimensions=dimensions)

    def __compute_base_migrations(self):
        capacity_changes = {r.key: set(r.availability.exception_intervals) for r in self._modified_instance.resources}
        capacity_changes_iter = {r.key: r.availability.exception_intervals for r in self._modified_instance.resources}

        def find_corresponding_addition(r_from, s, e, c):
            for r_to, chngs in capacity_changes.items():
                if r_to == r_from:
                    continue
                addition = (s, e, -c)
                if addition in chngs:
                    chngs.remove(addition)
                    return CapacityMigration(r_to, s, e, c)

            raise ValueError("No corresponding addition found to migrate")

        migrations = defaultdict(list)
        for resource_from_key, changes in capacity_changes_iter.items():
            for start, end, capacity in changes:
                if capacity > 0:
                    continue
                migration = find_corresponding_addition(resource_from_key, start, end, capacity)
                migrations[resource_from_key].append(migration)
                capacity_changes[resource_from_key].remove((start, end, capacity))

        self._capacity_migrations = migrations
        self._capacity_additions = {r.key: sorted(capacity_changes[r.key]) for r in self._modified_instance.resources}

    def __compute_reduced_capacity_changes(self):
        horizon = self._base_instance.horizon
        availabilities = {r.key: compute_resource_availability(r, horizon) for r in self._modified_instance.resources}
        consumptions = {r.key: compute_resource_consumption(self._modified_instance, self._solution, r) for r in self._modified_instance.resources}

        def find_max_consumption(rk, s, e):
            overlapping_consumptions = [c_c for c_s, c_e, c_c in consumptions[rk] if (s < c_e <= e) or (s <= c_s < e)]
            return max(overlapping_consumptions, default=0)

        def find_min_availability(rk, s, e):
            overlapping_availabilities = [a_c for a_s, a_e, a_c in availabilities[rk] if (s < a_e <= e) or (s <= a_s < e)]
            return min(overlapping_availabilities, default=0)

        def update_availability(rk, s, e, chng):
            availabilities[rk] = interval_overlap_function(availabilities[rk] + [(s, e, chng)], first_x=0, last_x=horizon)

        used_additions = defaultdict(list)
        for resource_key, additions in self._capacity_additions.items():
            for start, end, capacity in sorted(additions, key=lambda a: a[2]):
                max_consumption = find_max_consumption(resource_key, start, end)
                min_availability = find_min_availability(resource_key, start, end)
                surplus = min_availability - max_consumption

                if capacity <= surplus:
                    # the addition is redundant
                    update_availability(resource_key, start, end, -capacity)
                else:
                    used_additions[resource_key].append(CapacityChange(start, end, capacity))

        reduced_additions = defaultdict(list)
        for resource_key, additions in used_additions.items():
            for start, end, capacity in sorted(additions):
                max_consumption = find_max_consumption(resource_key, start, end)
                min_availability = find_min_availability(resource_key, start, end)
                surplus = min_availability - max_consumption
                update_availability(resource_key, start, end, -surplus)
                reduced_additions[resource_key].append(CapacityChange(start, end, capacity - surplus))

        all_migrations = sorted([(r_from, r_to, s, e, c)
                                 for r_from in self._capacity_migrations
                                 for r_to, s, e, c in self._capacity_migrations[r_from]],
                                key=lambda m: m[4])
        reduced_migrations = defaultdict(list)
        for resource_from_key, resource_to_key, start, end, capacity in all_migrations:
            max_consumption = find_max_consumption(resource_to_key, start, end)
            min_availability = find_min_availability(resource_to_key, start, end)
            surplus = min_availability - max_consumption
            if capacity <= surplus:
                # the migration is redundant
                update_availability(resource_to_key, start, end, -capacity)
                update_availability(resource_from_key, start, end, capacity)
            else:
                update_availability(resource_to_key, start, end, -surplus)
                update_availability(resource_from_key, start, end, surplus)
                reduced_migrations[resource_from_key].append(CapacityMigration(resource_to_key, start, end, capacity - surplus))

        self._capacity_additions = reduced_additions
        self._capacity_migrations = reduced_migrations


class EvaluationAlgorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, problem: ProblemSetup, settings) -> Evaluation:
        """Evaluates the given instance."""
