from instances.problem_instance import ProblemInstance
from rcpsp_sandbox.utils import print_error


def compute_topological_components(problem_instance: ProblemInstance) -> dict[int, list[int]]:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from([job.id_job for job in problem_instance.jobs])
    graph.add_edges_from([(p.id_child, p.id_parent) for p in problem_instance.precedences])

    try:
        sorted_components = [list(nx.topological_sort(graph.subgraph(component))) for component in nx.weakly_connected_components(graph)]
    except nx.NetworkXUnfeasible:
        print_error("Instance graph is not acyclic")
        raise

    components_by_root_job = {c[0]: c for c in sorted_components}

    if len(components_by_root_job) != len(problem_instance.components):
        raise KeyError("The number of specified components differs from the number of actual components in the instance graph")
    for component in problem_instance.components:
        if component.id_root_job not in components_by_root_job:
            raise KeyError("Specified component root is not a valid component root in the actual instance graph")

    return components_by_root_job
