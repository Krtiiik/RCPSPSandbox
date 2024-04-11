import itertools
import math
from collections import defaultdict
from typing import Iterable

from instances.problem_instance import ProblemInstance, Resource, Job, compute_resource_availability, CapacityMigration, \
    CapacityChange, compute_resource_periodical_availability
from solver.solution import Solution
from utils import interval_overlap_function

T_StepFunction = list[tuple[int, int, int]]


def compute_capacity_surpluses(solution: Solution, instance: ProblemInstance,
                               ignore_changes: bool = False,
                               ) -> dict[str, T_StepFunction]:
    surpluses = dict()
    for resource in instance.resources:
        capacity_f = (compute_resource_availability(resource, instance, instance.horizon) if not ignore_changes
                      else compute_resource_periodical_availability(resource, instance.horizon))
        consumption_f = compute_resource_consumption(instance, solution, resource)
        consumption_f = [(s, e, -c) for s, e, c in consumption_f]
        surplus_f = interval_overlap_function(capacity_f + consumption_f, first_x=0, last_x=instance.horizon)
        surplus_f = [(s, e, max(0, c)) for s, e, c in surplus_f]
        surpluses[resource.key] = surplus_f
    return surpluses


def compute_capacity_migrations(instance: ProblemInstance, solution: Solution,
                                capacity_requirements: dict[str, Iterable[CapacityChange]],
                                ) -> tuple[dict[str, list[CapacityMigration]], dict[str, T_StepFunction]]:
    # assuming uniform migrations over the intervals

    def find_migrations(s, e, c, r_to):
        possible_migrations = dict()
        for r, surplus in remaining_surpluses.items():
            if r == r_to:
                continue
            surplus_capacities = (s_c for s_s, s_e, s_c in surplus if (s <= s_s < e) or (s < s_e <= e))
            possible_migration = min(surplus_capacities, default=0)
            if possible_migration > 0:
                possible_migrations[r] = possible_migration

        migs = dict()
        remaining_c = c
        for r, capacity in sorted(possible_migrations.items(), key=lambda r_c: r_c[1], reverse=True):
            migration = min(capacity, remaining_c)
            migs[r] = migration
            remaining_c -= migration

            if remaining_c == 0:
                break

        return migs

    missing_capacities, remaining_surpluses = compute_missing_capacities(instance, solution, capacity_requirements, return_reduced_surpluses=True)
    resource_migrations: dict[str, list[CapacityMigration]] = defaultdict(list)
    resource_missing_capacities: dict[str, T_StepFunction] = defaultdict(list)
    for resource, missing_caps in missing_capacities.items():
        for start, end, missing_capacity in missing_caps:
            migrations = find_migrations(start, end, missing_capacity, resource)
            for r, c in migrations.items():
                resource_migrations[r].append(CapacityMigration(resource, start, end, c))

            migrated_capacity = sum(migrations.values())
            if migrated_capacity < missing_capacity:
                # Capacity addition might need to occur
                resource_missing_capacities[resource].append(CapacityChange(start, end, missing_capacity - migrated_capacity))

    return resource_migrations, resource_missing_capacities


def compute_missing_capacities(instance: ProblemInstance, solution: Solution,
                               capacity_requirements: dict[str, Iterable[tuple[int, int, int]]],
                               return_reduced_surpluses: bool = False,
                               ) -> dict[str, T_StepFunction] | tuple[dict[str, T_StepFunction], dict[str, T_StepFunction]]:
    def find_overlapping_capacities(r, s, e):
        return [s_c for s_s, s_e, s_c in capacity_surpluses[r]
                if (s <= s_s < e) or (s < s_e <= e)]

    def update_surplus(r, requirement):
        # this is very ineffective, but works
        capacity_surpluses[r] = interval_overlap_function(capacity_surpluses[r] + [requirement], first_x=0, last_x=instance.horizon)

    capacity_surpluses = compute_capacity_surpluses(solution, instance)
    missing_capacities = defaultdict(list)
    for resource, requirements in capacity_requirements.items():
        for start, end, capacity in sorted(requirements):
            overlapping_capacities = find_overlapping_capacities(resource, start, end)
            available_capacity = min(overlapping_capacities, default=0)
            if capacity <= available_capacity:
                # this is good, the required capacity is there
                update_surplus(resource, (start, end, capacity))
            else:
                # this is bad, we don't have the required capacity
                missing_capacity = (capacity - available_capacity)
                missing_capacities[resource].append((start, end, missing_capacity))

    return missing_capacities, capacity_surpluses if return_reduced_surpluses else missing_capacities


def compute_resource_consumption(instance: ProblemInstance, solution: Solution, resource: Resource,
                                 selected: Iterable[int] = None,
                                 ) -> T_StepFunction:
    selected = set(selected if selected is not None else (j.id_job for j in instance.jobs))
    consumptions = []
    for job, consumption in jobs_consuming_resource(instance, resource, yield_consumption=True):
        if job.id_job not in selected:
            continue
        int_solution = solution.job_interval_solutions[job.id_job]
        consumptions.append((int_solution.start, int_solution.end, consumption))
    consumption_f = interval_overlap_function(consumptions, first_x=0, last_x=instance.horizon)
    return consumption_f


def jobs_consuming_resource(instance: ProblemInstance, resource: Resource, yield_consumption: bool = False) -> Iterable[Job]:
    for job in instance.jobs:
        consumption = job.resource_consumption.consumption_by_resource[resource]
        if consumption > 0:
            yield job, consumption if yield_consumption else job


def compute_resource_shift_starts(instance: ProblemInstance) -> dict[Resource, Iterable[int]]:
    shift_starts = defaultdict(list)
    for resource in instance.resources:
        resource_availability = compute_resource_availability(resource, instance, instance.horizon)
        last_c = 0
        for s, e, c in resource_availability:
            if c > 0 and last_c == 0:
                shift_starts[resource].append(s)
            last_c = c

    return shift_starts


def compute_resource_shift_ends(instance: ProblemInstance) -> dict[Resource, list[int]]:
    shift_ends = defaultdict(list)
    for resource in instance.resources:
        resource_availability = compute_resource_availability(resource, instance, instance.horizon)
        last_c = 0
        last_e = 0
        for s, e, c in resource_availability:
            if c == 0 and last_c > 0:
                shift_ends[resource].append(last_e)
            last_c = c
            last_e = e

    return shift_ends


def group_consecutive_intervals(intervals):
    result = []
    current_group = []

    for i in range(len(intervals)):
        if i == 0 or intervals[i][0] == intervals[i - 1][1]:
            # If it's the first tuple or starts immediately after the previous tuple
            current_group.append(intervals[i])
        else:
            # Start a new group
            result.append(current_group)
            current_group = [intervals[i]]

    if current_group:
        # Add the last group
        result.append(current_group)

    return result
