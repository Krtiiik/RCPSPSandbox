import itertools
import math
import random
from collections import defaultdict

from docplex.cp import modeler
from docplex.cp.expression import interval_var
from docplex.cp.function import CpoStepFunction
from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoSolveResult

from instances.drawing import draw_instance_graph
from instances.algorithms import traverse_instance_graph
from instances.problem_instance import ProblemInstance, Resource, Job
from drawing import plot_solution
from instances.problem_modifier import modify_instance
from utils import index_groups


class Solver:
    def build_model(self, problem_instance: ProblemInstance) -> CpoModel:
        resource_availabilities = {resource: self.__build_resource_availability(resource, problem_instance.horizon)
                                   for resource in problem_instance.resources}

        job_intervals = {job.id_job: interval_var(name=f"Job {job.id_job}",
                                                  size=job.duration,
                                                  intensity=self.__build_job_execution_availability(job,
                                                                                                    resource_availabilities))
                         for job in problem_instance.jobs}
        precedence_constraints = [
            modeler.end_before_start(job_intervals[precedence.id_child], job_intervals[precedence.id_parent])
            for precedence in problem_instance.precedences]

        jobs_consuming_resource = {resource: [job for job in problem_instance.jobs
                                              if job.resource_consumption[resource] > 0]
                                   for resource in problem_instance.resources}
        resource_capacity_constraints = [modeler.sum(modeler.pulse(job_intervals[job.id_job], job.resource_consumption[resource])
                                                     for job in jobs_consuming_resource[resource])
                                         <= resource.capacity
                                         for resource in problem_instance.resources]

        # optimization_goal = modeler.minimize(modeler.max(modeler.end_of(job_interval)
        #                                                  for job_interval in job_intervals.values()))

        # TODO tardiness
        jobs_by_id = {j.id_job: j for j in problem_instance.jobs}
        jobs_components_grouped = [[jobs_by_id[i[0]] for i in group]
                                   for _k, group in itertools.groupby(traverse_instance_graph(problem_instance, search="components topological generations", yield_state=True),
                                                                      key=lambda x: x[1])]  # we assume that the order in which jobs are returned is determined by the components, so we do not sort by component id
        component_jobs_by_root_job = index_groups(jobs_components_grouped, [jobs_by_id[c.id_root_job] for c in problem_instance.components])
        weights_by_id_root_job = {c.id_root_job: c.weight for c in problem_instance.components}

        sum_all_ends = modeler.sum(modeler.end_of(job_intervals[job.id_job])
                                   for job in problem_instance.jobs)
        weighted_tardiness = (problem_instance.projects[0].tardiness_cost  # assuming only a single project
                              * modeler.sum(weights_by_id_root_job[root_job.id_job]
                                            * modeler.max(modeler.max(0, modeler.end_of(job_intervals[job.id_job]) - job.due_date)
                                                          for job in jobs)
                                            for root_job, jobs in component_jobs_by_root_job.items()))
        optimization_goal = modeler.minimize(sum_all_ends + weighted_tardiness)

        model = CpoModel(problem_instance.name)
        model.add(job_intervals.values())
        model.add(precedence_constraints)
        model.add(resource_capacity_constraints)
        model.add(optimization_goal)
        return model

    def solve(self,
              problem_instance: ProblemInstance or None = None,
              model: CpoModel or None = None) -> CpoSolveResult:
        if model is not None:
            return self.__solve_model(model)
        elif problem_instance is not None:
            model = self.build_model(problem_instance)
            return self.__solve_model(model)
        else:
            raise TypeError("No problem instance nor model was specified to solve")

    @staticmethod
    def __solve_model(model: CpoModel) -> CpoSolveResult:
        return model.solve()

    @staticmethod
    def __build_resource_availability(resource: Resource, horizon: int) -> CpoStepFunction:
        day_operating_hours = resource.availability if resource.availability is not None else [(0, 24)]
        days_count = math.ceil(horizon / 24)
        step_values = dict()
        for i_day in range(days_count):
            day_offset = i_day * 24
            for start, end in day_operating_hours:
                step_values[day_offset + start] = 1
                step_values[day_offset + end] = 0

        steps = sorted(step_values.items())
        return CpoStepFunction(steps)

    @staticmethod
    def __build_resource_capacity(resource: Resource, resource_availability: CpoStepFunction) -> CpoStepFunction:
        steps = [(step[0], resource.capacity if step[1] == 1 else 0)
                 for step in resource_availability.get_step_list()]
        return CpoStepFunction(steps)

    @staticmethod
    def __build_job_execution_availability(job: Job,
                                           resource_availabilities: dict[Resource, CpoStepFunction]) -> CpoStepFunction:
        used_resources = [resource for resource, consumption in job.resource_consumption.consumption_by_resource.items()
                          if consumption > 0]
        step_values = defaultdict(lambda: True)  # Default dict with true boolean as the default value
        for resource in used_resources:
            for step in resource_availabilities[resource].get_step_list():
                step_values[step[0]] &= (step[1] == 1)

        steps = sorted((step[0], 100 if step[1] else 0)
                       for step in step_values.items())
        return CpoStepFunction(steps)


if __name__ == "__main__":
    import rcpsp_sandbox.instances.io as ioo
    random.seed(42)

    # inst = ioo.parse_psplib("../../../Data/RCPSP/extended/instance_11.rp", is_extended=True)
    inst = ioo.parse_psplib("../../../Data/RCPSP/j30/j301_2.sm")
    draw_instance_graph(inst)

    inst = (modify_instance(inst)
            .split_job_components(split="paths")
            .assign_job_due_dates(choice="gradual", gradual_base=10, gradual_interval=(0, 10))
            .generate_modified_instance())
    draw_instance_graph(inst)

    s = Solver()
    solve_result = s.solve(inst)
    if solve_result.is_solution():
        solution = solve_result.get_solution()
        solution.print_solution()
        plot_solution(inst, solution)
