import math
from collections import defaultdict

from docplex.cp import modeler
from docplex.cp.expression import interval_var
from docplex.cp.function import CpoStepFunction
from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoSolveResult

import rcpsp_sandbox.instances.drawing
from instances.problem_instance import ProblemInstance, Resource, Job
from drawing import plot_solution
from rcpsp_sandbox.solver.utils import compute_topological_components


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
        resource_capacity_constraints = [modeler.less_or_equal(
            modeler.sum(modeler.pulse(job_intervals[job.id_job], job.resource_consumption[resource])
                        for job in jobs_consuming_resource[resource]),
            resource.capacity)
                                         for resource in problem_instance.resources]

        optimization_goal = modeler.minimize(modeler.max(modeler.end_of(job_interval)
                                                         for job_interval in job_intervals.values()))

        components = compute_topological_components(problem_instance)
        weights_by_root_job = {c.id_root_job: c.weight for c in problem_instance.components}
        modeler.minimize(modeler.sum(modeler.end_of(job_intervals[job.id_job])
                                     for job in problem_instance.jobs)
                         + (problem_instance.projects[0].tardiness_cost
                            * modeler.sum(weights_by_root_job[id_root_job]
                                          * modeler.max(modeler.max(0, modeler.end_of(job_intervals[job.id_job]) - job.due_date)
                                                        for job in jobs)
                                          for id_root_job, jobs in components)))

        model = CpoModel("Test")
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
        day_operating_hours = resource.availability
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

    inst = ioo.parse_psplib("../../../Data/RCPSP/extended/instance_11.rp", is_extended=True)
    rcpsp_sandbox.instances.drawing.draw_instance_graph(inst)
    s = Solver()
    solve_result = s.solve(inst)
    if solve_result.is_solution():
        solution = solve_result.get_solution()
        solution.print_solution()
        plot_solution(inst, solution)
