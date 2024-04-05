import itertools
from typing import Iterable, Tuple

from docplex.cp.function import CpoStepFunction
from docplex.cp.solution import CpoIntervalVarSolution
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt
import numpy as np
import tabulate

from utils import print_error, ColorMap
from instances.problem_instance import ProblemInstance, Job, Resource, compute_component_jobs, \
    compute_resource_availability
from solver.solution import Solution, compute_job_tardiness


def plot_solution(problem_instance: ProblemInstance,
                  solution: Solution,
                  split_components: bool = False,
                  split_resource_consumption: bool = False,
                  plot_resource_capacity: bool = True,
                  resource_functions: dict[Resource, list[tuple[int, int, int]]] = None,
                  highlight_jobs: Iterable[int] = None,
                  save_as: str = None,
                  ):
    """
    See http://ibmdecisionoptimization.github.io/docplex-doc/cp/visu.rcpsp.py.html
    """
    if not visu.is_visu_enabled():
        print_error("docplex visu is not enabled, aborting plot...")
        return

    # ~~~ Indices and structures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    job_interval_solutions = solution.job_interval_solutions
    component_jobs = compute_component_jobs(problem_instance)
    job_component_index: dict[int, int] = {job.id_job: i_comp
                                           for i_comp, comp_id_root_job in enumerate(sorted(component_jobs.keys(), key=lambda j: j.id_job))
                                           for job in component_jobs[comp_id_root_job]}

    cm = ColorMap()
    highlight_jobs = set(highlight_jobs) if highlight_jobs else set(j.id_job for j in problem_instance.jobs)

    def get_color(x):
        return cm[job_component_index[x]] if x in highlight_jobs else 'grey'

    # ~~~ Load computation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if split_components:
        load: dict[int, dict[int, CpoStepFunction]] = \
            {i_comp: {resource.id_resource: CpoStepFunction()
                      for resource in problem_instance.resources}
             for i_comp, _ in enumerate(component_jobs.keys())}

        def add_load(r: Resource, j: Job, ji: CpoIntervalVarSolution, c: int):
            load[job_component_index[j.id_job]][r.id_resource].add_value(ji.get_start(), ji.get_end(), c)
    else:
        if split_resource_consumption:
            load: dict[int, list[Tuple[int, CpoStepFunction]]] = \
                {resource.id_resource: [(i_comp, CpoStepFunction())
                                        for i_comp in range(len(problem_instance.components))]
                 for resource in problem_instance.resources}

            def add_load(r: Resource, j: Job, ji: CpoIntervalVarSolution, c: int):
                # disable inspections as PyCharm has no idea what is going on there...
                # noinspection PyUnresolvedReferences,PyTypeChecker
                load[r.id_resource][job_component_index[j.id_job]][1].add_value(ji.get_start(), ji.get_end(), c)
        else:
            load: dict[int, CpoStepFunction] = {resource.id_resource: CpoStepFunction() for resource in
                                                problem_instance.resources}

            def add_load(r: Resource, _j: Job, ji: CpoIntervalVarSolution, c: int):
                # ... same as above ...
                # noinspection PyUnresolvedReferences,PyTypeChecker
                load[r.id_resource].add_value(ji.get_start(), ji.get_end(), c)

    for job in problem_instance.jobs:
        job_interval = job_interval_solutions[job.id_job]
        for resource, consumption in job.resource_consumption.consumption_by_resource.items():
            if consumption > 0:
                add_load(resource, job, job_interval, consumption)

    # ~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if split_components:
        def plot_component_consumption(i_comp):
            consumptions = sorted(load[i_comp].items(), key=lambda x: x[0], reverse=True)
            fs = reversed(list(itertools.accumulate(f for _, f in consumptions)))
            for i_f, f in enumerate(fs):
                visu.function(segments=f, style='area', color=i_f)

        visu.timeline("Solution")
        comp_count = len(component_jobs.keys())
        for i_comp, root_job in enumerate(component_jobs.keys()):
            visu.panel(f"Component {str(i_comp)}")
            for job in component_jobs[root_job]:
                interval_solution = job_interval_solutions[job.id_job]
                color = get_color(job.id_job)
                visu.interval(interval_solution, color, str(job.id_job))
            plot_component_consumption(i_comp)

        plt.rcParams["figure.figsize"] = (12, 4*(1 + comp_count))
    else:
        if split_resource_consumption:
            def plot_resource_consumption(r: Resource):
                i_comps, fs = zip(*load[r.id_resource])
                fs = itertools.accumulate(fs)
                for i_comp, f in reversed(list(zip(i_comps, fs))):
                    visu.function(segments=f, style='area', color=cm[i_comp])
        else:
            def plot_resource_consumption(r: Resource):
                # noinspection PyTypeChecker
                visu.function(segments=load[r.id_resource], style='area', color='green')

        visu.timeline("Solution", origin=0)
        visu.panel("Jobs")
        for job in problem_instance.jobs:
            interval_solution = job_interval_solutions[job.id_job]
            color = get_color(job.id_job)
            visu.interval(interval_solution, color, str(job.id_job))
        for i, resource in enumerate(sorted(problem_instance.resources, key=lambda r: r.key)):
            visu.panel(resource.key)

            if plot_resource_capacity:
                segments = compute_resource_availability(resource, problem_instance.horizon)
                visu.function(segments=segments, style='area', color='grey')

            plot_resource_consumption(resource)
            if resource_functions and resource in resource_functions:
                visu.function(segments=resource_functions[resource], style='line', color='c')

        plt.rcParams["figure.figsize"] = (12, 4*(1 + len(problem_instance.resources)))

    plt.rcParams["figure.dpi"] = 300
    if save_as is not None:
        visu.show(pngfile=save_as)
    visu.show(origin=0)


def print_difference(original: Solution, original_instance: ProblemInstance,
                     alternative: Solution, alternative_instance: ProblemInstance,
                     selected_jobs: Iterable[Job] = None) -> None:
    diff_total, diffs = original.difference_to(alternative, selected_jobs)
    print("Difference:", diff_total)
    print(compute_job_tardiness(original, selected_jobs), compute_job_tardiness(alternative, selected_jobs))

    selected = set(j.id_job for j in (selected_jobs if selected_jobs is not None else original_instance.jobs))  # Assuming both instances have the same jobs (wouldn't make much sense otherwise anyway)
    original_jobs = (j for j in original_instance.jobs if j.id_job in selected)
    alternative_jobs = (j for j in alternative_instance.jobs if j.id_job in selected)
    job_pairs = list(zip(sorted(original_jobs, key=lambda j: j.id_job),
                         sorted(alternative_jobs, key=lambda j: j.id_job)))

    orig, alt = original.job_interval_solutions, alternative.job_interval_solutions

    def org_end(j): return orig[j.id_job].end
    def alt_end(j): return alt[j.id_job].end
    def org_unit_tardiness(j): return max(0, org_end(j) - j.due_date)
    def alt_unit_tardiness(j): return max(0, alt_end(j) - j.due_date)
    def hide_null(value): return str(value) if value != 0 else '.'
    def diff_coll(diff): return ['', f'/˄\\ {diff}', f'˅˅˅ {diff}'][np.sign(diff)]

    # TODO weighted tardiness
    tardiness_data = [(j1.id_job,
                       org_end(j1),
                       alt_end(j2),
                       diff_coll(alt_end(j2) - org_end(j1)),
                       hide_null(org_unit_tardiness(j1)),
                       hide_null(alt_unit_tardiness(j2)),
                       diff_coll(alt_unit_tardiness(j2) - org_unit_tardiness(j1)))
                      for j1, j2 in job_pairs]

    print("Job Tardiness")
    print("-------------")
    print(tabulate.tabulate(tardiness_data,
                            headers=["Id", "Org End", "Alt End", "End Diff", "Org Tard", "Alt Tard", "Tard Diff"],
                            colalign=['right', 'right', 'right', 'left', 'right', 'right', 'left']))
