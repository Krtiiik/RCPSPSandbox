import itertools
import math
from typing import Iterable, Tuple

from docplex.cp.expression import INTERVAL_MIN, INTERVAL_MAX
from docplex.cp.function import CpoStepFunction
from docplex.cp.solution import CpoModelSolution, CpoIntervalVarSolution
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt
import numpy as np
import tabulate

from utils import print_error
from instances.problem_instance import ProblemInstance, Job, Resource
from solver.solution import Solution, solution_tardiness_value
from solver.utils import compute_component_jobs


COLORS = [
    'xkcd:green',
    'xkcd:orange',
    'xkcd:red',
    'xkcd:violet',
    'xkcd:blue',
    'xkcd:yellow green',
    # 'xkcd:yellow',
    'xkcd:yellow orange',
    # 'xkcd:red orange',
    'xkcd:red violet',
    'xkcd:blue violet',
    'xkcd:blue green'
]


class ColorMap:
    def __init__(self, color_count):
        self.range = color_count
        self.colors = COLORS

    def __getitem__(self, item):
        assert isinstance(item, int)
        return self.colors[item % len(self.colors)]


def plot_solution(problem_instance: ProblemInstance,
                  solution: Solution,
                  fit_to_width: int = 0,
                  split_components: bool = False,
                  split_resource_consumption: bool = False,
                  save_as: str = None):
    """
    See http://ibmdecisionoptimization.github.io/docplex-doc/cp/visu.rcpsp.py.html
    TODO values of step functions not reaching zero over the full horizon do not start at zero
    """
    if not visu.is_visu_enabled():
        print_error("docplex visu is not enabled, aborting plot...")
        return

    # ~~~ Indices and structures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    job_interval_solutions = solution.job_interval_solutions
    component_jobs = compute_component_jobs(problem_instance)
    job_component_index: dict[int, int] = {job.id_job: i_comp
                                           for i_comp, comp_id_root_job in enumerate(component_jobs.keys())
                                           for job in component_jobs[comp_id_root_job]}
    cm = ColorMap(len(problem_instance.components))
    days_count = math.ceil(max(interval_solution.get_end() for interval_solution in job_interval_solutions.values()))

    def compute_resource_pauses(r: Resource):
        values = [0] + [i_day * 24 + t
                        for i_day in range(days_count)
                        for av in r.availability.periodical_intervals
                        for t in [av.start, av.end]]
        pauses = list(itertools.pairwise(values))[::2]
        return pauses

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

        visu.timeline("Solution", horizon=fit_to_width)
        comp_count = len(component_jobs.keys())
        for i_comp, root_job in enumerate(component_jobs.keys()):
            visu.panel(f"Component {str(i_comp)}")
            for job in component_jobs[root_job]:
                interval_solution = job_interval_solutions[job.id_job]
                color = cm[job_component_index[job.id_job]]
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

        visu.timeline("Solution", horizon=fit_to_width)
        visu.panel("Jobs")
        for job in problem_instance.jobs:
            interval_solution = job_interval_solutions[job.id_job]
            color = cm[job_component_index[job.id_job]]
            visu.interval(interval_solution, color, str(job.id_job))
        for i, resource in enumerate(sorted(problem_instance.resources, key=lambda r: r.key)):
            visu.panel(resource.key)

            segments = []
            last_x = 0
            for p_start, p_end in compute_resource_pauses(resource):
                segments.append((last_x, p_start, resource.capacity))
                segments.append((p_start, p_end, 0))
                last_x = p_end

            visu.function(segments=segments, style='area', color=i)
            # visu.function(segments=[(INTERVAL_MIN, INTERVAL_MAX, resource.capacity)], style='area', color=i)
            plot_resource_consumption(resource)

        plt.rcParams["figure.figsize"] = (12, 4*(1 + len(problem_instance.resources)))

    plt.rcParams["figure.dpi"] = 300
    if save_as is not None:
        visu.show(pngfile=save_as)
    visu.show()


def print_difference(original: Solution, original_instance: ProblemInstance,
                     alternative: Solution, alternative_instance: ProblemInstance,
                     selected_jobs: Iterable[Job] = None) -> None:
    diff_total, diffs = original.difference_to(alternative, selected_jobs)
    print("Difference:", diff_total)
    print(solution_tardiness_value(original, selected_jobs), solution_tardiness_value(alternative, selected_jobs))

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
