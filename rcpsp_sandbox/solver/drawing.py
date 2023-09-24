import matplotlib.pyplot as plt
from docplex.cp.expression import INTERVAL_MIN, INTERVAL_MAX
from docplex.cp.function import CpoStepFunction
from docplex.cp.solution import CpoModelSolution
import docplex.cp.utils_visu as visu

from instances.problem_instance import ProblemInstance


def plot_solution(problem_instance: ProblemInstance,
                  solution: CpoModelSolution):
    if not visu.is_visu_enabled():
        return

    job_intervals = {int(var_solution.get_name()[4:]): var_solution.get_var()
                     for var_solution in solution.get_all_var_solutions()}
    load: dict[int, CpoStepFunction] = {resource.id_resource: CpoStepFunction() for resource in problem_instance.resources}
    for job in problem_instance.jobs:
        job_interval = solution.get_var_solution(job_intervals[job.id_job])
        for resource, consumption in job.resource_consumption.consumption_by_resource.items():
            if consumption > 0:
                load[resource.id_resource].add_value(job_interval.get_start(), job_interval.get_end(), consumption)

    visu.timeline("Solution")
    visu.panel("Jobs")
    for i, job in enumerate(problem_instance.jobs):
        interval = job_intervals[job.id_job]
        visu.interval(solution.get_var_solution(interval), i, interval.get_name())
    for resource in problem_instance.resources:
        visu.panel(resource.key)
        visu.function(segments=[(INTERVAL_MIN, INTERVAL_MAX, resource.capacity)], style='area', color='lightgrey')
        visu.function(segments=load[resource.id_resource], style='area', color='green')
    plt.rcParams["figure.figsize"] = (14, 20)
    visu.show()
