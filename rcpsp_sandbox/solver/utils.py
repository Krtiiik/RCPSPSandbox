from docplex.cp.expression import interval_var, CpoIntervalVar
from docplex.cp.model import CpoModel


def get_model_job_intervals(model: CpoModel) -> dict[int, interval_var]:
    return {int(var.get_name()[4:]): var
            for var in model.get_all_variables()
            if isinstance(var, CpoIntervalVar) and var.get_name().startswith("Job")}
