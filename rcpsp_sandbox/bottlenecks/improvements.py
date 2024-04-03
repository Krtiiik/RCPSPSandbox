import itertools
from collections import defaultdict
from queue import Queue
from typing import Iterable

import networkx as nx
from intervaltree import IntervalTree

from bottlenecks.utils import jobs_consuming_resource, compute_resource_shift_starts, \
    compute_resource_shift_ends
from instances.algorithms import build_instance_graph
from instances.problem_instance import ProblemInstance, ResourceConsumption, Resource
from solver.solution import Solution
from utils import interval_overlap_function


def time_relaxed_suffixes(instance: ProblemInstance, solution: Solution,
                          granularity: int = 1,
                          ):
    start_times = {job.id_job: solution.job_interval_solutions[job.id_job].start for job in instance.jobs}
    completion_times = {job.id_job: solution.job_interval_solutions[job.id_job].end for job in instance.jobs}
    durations = {job.id_job: job.duration for job in instance.jobs}
    predecessors = {job.id_job: [] for job in instance.jobs}
    for precedence in instance.precedences:
        predecessors[precedence.id_parent].append(precedence.id_child)

    t_to_job_to_start = {t: dict() for t in range(0, instance.horizon, granularity)}
    t_job_first = {t: dict() for t in range(0, instance.horizon, granularity)}

    graph = build_instance_graph(instance)
    jobs_topological = list(nx.topological_sort(graph))
    for t in range(0, instance.horizon, granularity):
        for job_id in jobs_topological:
            if start_times[job_id] <= t:
                t_to_job_to_start[t][job_id] = start_times[job_id]
                t_job_first[t][job_id] = True
            else:
                earliest_bound = max((t_to_job_to_start[t][predecessor] + durations[predecessor] for predecessor, _ in graph.in_edges(job_id)), default=0)
                t_to_job_to_start[t][job_id] = earliest_bound
                t_job_first[t][job_id] = all(start_times[predecessor] <= t for predecessor, _ in graph.in_edges(job_id))

    return t_to_job_to_start, t_job_first


def relaxed_intervals(instance: ProblemInstance, solution: Solution,
                      granularity: int = 1,
                      component: int = None,
                      ):
    t_job_start, t_job_first = time_relaxed_suffixes(instance, solution, granularity)
    durations = {j.id_job: j.duration for j in instance.jobs}
    consumptions = {j.id_job: j.resource_consumption for j in instance.jobs}
    start_times = {j.id_job: solution.job_interval_solutions[j.id_job].start for j in instance.jobs}
    component_closure = (left_closure(component, instance, solution) if component else set(j.id_job for j in instance.jobs))

    t_consumptions = defaultdict(list)
    included = set()
    for t, t1 in itertools.pairwise(sorted(t_job_start) + [instance.horizon + 1]):
        for job_id in t_job_start[t]:
            start = t_job_start[t][job_id]
            end = start + durations[job_id]
            if (t <= start < t1
                and start < start_times[job_id]
                and job_id not in included
                and t_job_first[t][job_id]
                and job_id in component_closure
            ):
                t_consumptions[t].append((start, end, consumptions[job_id]))
                included.add(job_id)

    return t_consumptions


def relaxed_interval_consumptions(instance: ProblemInstance, solution: Solution,
                                  granularity: int = 1,
                                  component: int = None,
                                  return_intervals: bool = False,
                                  ):
    interval_consumptions: dict[int, list[tuple[int, int, ResourceConsumption]]] = relaxed_intervals(instance, solution, granularity, component)
    interval_consumptions_by_resource: dict[Resource, list[tuple[int, int, int]]] = defaultdict(list)

    for start, end, consumptions in itertools.chain(*interval_consumptions.values()):
        for resource in consumptions.consumption_by_resource:
            if consumptions.consumption_by_resource[resource] > 0:
                interval_consumptions_by_resource[resource].append((start, end, consumptions.consumption_by_resource[resource]))

    result = {r: interval_overlap_function(interval_consumptions_by_resource[r], first_x=0, last_x=instance.horizon) for r in interval_consumptions_by_resource}
    return (result, interval_consumptions_by_resource) if return_intervals else result


def left_closure(id_job: int, instance: ProblemInstance, solution: Solution) -> Iterable[int]:
    start_times = {j.id_job: solution.job_interval_solutions[j.id_job].start for j in instance.jobs}
    completion_times = {j.id_job: solution.job_interval_solutions[j.id_job].end for j in instance.jobs}
    durations = {j.id_job: j.duration for j in instance.jobs}
    graph = build_instance_graph(instance)
    resource_shift_starts = {r: set(ss) for r, ss in compute_resource_shift_starts(instance).items()}
    resource_shift_ends = {r: sorted(se) for r, se in compute_resource_shift_ends(instance).items()}
    resources_interval_trees = {r: IntervalTree.from_tuples((start_times[j.id_job], completion_times[j.id_job], j.id_job)
                                                            for j, c in jobs_consuming_resource(instance, r))
                                for r in instance.resources}

    completion_time_jobs = defaultdict(set)  # Mapping from time t to jobs with completion time t
    job_consumption = {}  # Mapping from job to resources the job consumes
    for job in instance.jobs:
        job_consumption[job.id_job] = set()
        completion_time_jobs[completion_times[job.id_job]].add(job.id_job)
        for r, c in job.resource_consumption.consumption_by_resource.items():
            if c > 0:
                job_consumption[job.id_job].add(r)

    queue = Queue()
    queue.put(id_job)

    closure = set()
    while not queue.empty():
        n = queue.get(block=False)
        start = start_times[n]

        if n in closure:
            continue

        # Process precedence predecessors
        for pred in graph.predecessors(n):
            if completion_times[pred] == start:
                queue.put(pred)

        if start in completion_time_jobs:
            # Process resource predecessors
            for pred in completion_time_jobs[start]:
                if job_consumption[n] & job_consumption[pred]:
                    queue.put(pred)

        # Process resource predecessors over resource pauses
        for r in job_consumption[n]:
            if start in resource_shift_starts[r]:
                prev_shift_end = max((s_end for s_end in resource_shift_ends[r] if s_end < start), default=0)
                low_bound = prev_shift_end - durations[n]
                for s, e, pred in resources_interval_trees[r].overlap(low_bound, prev_shift_end):
                    queue.put(pred)

        closure.add(n)

    return closure
