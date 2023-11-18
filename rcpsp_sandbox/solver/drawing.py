from typing import Iterable

import matplotlib.pyplot as plt
from docplex.cp.expression import INTERVAL_MIN, INTERVAL_MAX
from docplex.cp.function import CpoStepFunction
from docplex.cp.solution import CpoModelSolution
import docplex.cp.utils_visu as visu

from rcpsp_sandbox.instances.problem_instance import ProblemInstance, Job
from rcpsp_sandbox.solver.solution_utils import solution_difference, solution_tardiness_value
from rcpsp_sandbox.solver.utils import compute_component_jobs, get_solution_job_interval_solutions

COLORS = ['gold', 'lime', 'chocolate', 'lightcoral', 'turquoise', 'orange', 'gray', 'deepskyblue', 'tomato',
          'blueviolet', 'aqua', 'darkorange', 'green', 'navy', 'greenyellow', 'darkgray', 'yellow', 'magenta',
          'crimson', 'darkred', 'orangered']


class ColorMap:
    def __init__(self, color_count):
        self.range = color_count
        self.colors = COLORS

    def __getitem__(self, item):
        assert isinstance(item, int)
        return self.colors[item % len(self.colors)]


def plot_solution(problem_instance: ProblemInstance,
                  solution: CpoModelSolution):
    """
    See http://ibmdecisionoptimization.github.io/docplex-doc/cp/visu.rcpsp.py.html
    :param problem_instance:
    :param solution:
    :return:
    """
    if not visu.is_visu_enabled():
        return

    job_intervals = {int(var_solution.get_name()[4:]): var_solution.get_var()
                     for var_solution in solution.get_all_var_solutions()}
    component_jobs = compute_component_jobs(problem_instance)
    component_jobs = {job.id_job: i_comp
                      for i_comp, comp_id_root_job in enumerate(component_jobs.keys())
                      for job in component_jobs[comp_id_root_job]}

    load: dict[int, CpoStepFunction] = {resource.id_resource: CpoStepFunction() for resource in problem_instance.resources}
    for job in problem_instance.jobs:
        job_interval = solution.get_var_solution(job_intervals[job.id_job])
        for resource, consumption in job.resource_consumption.consumption_by_resource.items():
            if consumption > 0:
                load[resource.id_resource].add_value(job_interval.get_start(), job_interval.get_end(), consumption)

    cm = ColorMap(len(problem_instance.components))
    visu.timeline("Solution")
    visu.panel("Jobs")
    for job in problem_instance.jobs:
        interval = job_intervals[job.id_job]
        interval_name = interval.get_name()[4:]
        color = cm[component_jobs[job.id_job]]
        visu.interval(solution.get_var_solution(interval), color, interval_name)
    for resource in sorted(problem_instance.resources, key=lambda r: r.key):
        visu.panel(resource.key)
        visu.function(segments=[(INTERVAL_MIN, INTERVAL_MAX, resource.capacity)], style='area', color='lightgrey')
        visu.function(segments=load[resource.id_resource], style='area', color='green')
    plt.rcParams["figure.figsize"] = (14, 20)
    visu.show()


def print_difference(original: CpoModelSolution, original_instance: ProblemInstance,
                     alternative: CpoModelSolution, alternative_instance: ProblemInstance,
                     selected_jobs: Iterable[Job]) -> None:
    diff_total, diffs = solution_difference(original, alternative)
    print("Difference:", diff_total)
    print(solution_tardiness_value(original, original_instance, selected_jobs), solution_tardiness_value(alternative, alternative_instance, selected_jobs))
    # print(*diffs.items(), sep='\n')
    print("job", "orig", "alt")
    orig, alt = get_solution_job_interval_solutions(original), get_solution_job_interval_solutions(alternative)
    job_pairs = zip(sorted(original_instance.jobs, key=lambda j: j.id_job),
                    sorted(alternative_instance.jobs, key=lambda j: j.id_job))
    for j1, j2 in job_pairs:
        assert j1.id_job == j2.id_job
        print(j1.id_job,
              max(0, orig[j1.id_job].end - j1.due_date),
              max(0, alt[j2.id_job].end - j2.due_date),
              ("!!!" if orig[j1.id_job] != alt[j2.id_job] else ''))
