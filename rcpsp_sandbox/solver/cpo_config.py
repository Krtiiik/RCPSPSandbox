from docplex.cp.config import context


def default():
    pass


def efficient():
    context.verbose = 1
    context.model.add_source_location = False
    context.model.length_for_alias = 10
    context.model.name_all_constraints = False
    context.model.dump_directory = None
    context.model.sort_names = None
    context.solver.trace_log = False
    context.solver.trace_cpo = False
    context.solver.add_log_to_solution = False


efficient()
