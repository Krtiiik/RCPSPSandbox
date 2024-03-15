import math

from docplex.cp.expression import interval_var, CpoIntervalVar
from docplex.cp.model import CpoModel

from instances.problem_instance import Resource
from utils import interval_overlap_function


def get_model_job_intervals(model: CpoModel) -> dict[int, interval_var]:
    return {int(var.get_name()[4:]): var
            for var in model.get_all_variables()
            if isinstance(var, CpoIntervalVar) and var.get_name().startswith("Job")}


def build_resource_availability(resource: Resource, horizon: int) -> list[tuple[int, int, int]]:
    """
    Builds a step function representing the availability of a resource over time.

    Args:
        resource (Resource): The resource to build the availability function for.
        horizon (int): The total number of hours in the planning horizon.

    Returns:
        CpoStepFunction: A step function representing the availability of the resource.
    """
    days_count = math.ceil(horizon / 24)
    intervals = [(i_day * 24 + start, i_day * 24 + end, capacity)
                 for i_day in range(days_count)
                 for start, end, capacity in resource.availability.periodical_intervals]
    return interval_overlap_function(intervals + resource.availability.exception_intervals,
                                     first_x=0, last_x=days_count*24)
