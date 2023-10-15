import random
from typing import Generator

import networkx as nx

from instances.problem_instance import ProblemInstance, Job
from utils import print_error


def build_instance_graph(instance: ProblemInstance) -> nx.DiGraph:
    """
    Builds a job-graph of the given problem instance.
    :param instance: The instance to build the graph of.
    :return: The oriented job-graph of the problem instance.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(job.id_job for job in instance.jobs)
    graph.add_edges_from((precedence.id_child, precedence.id_parent) for precedence in instance.precedences)
    return graph


def traverse_instance_graph(problem_instance: ProblemInstance = None,
                            graph: nx.DiGraph = None,
                            search: str = "topological",
                            yield_state: bool = False) -> Generator[Job, None, None]:
    """
    Traverses the job-graph of a given problem instance, yielding jobs in the order of visiting. The available
    search type options are:

    - topological (default): Traverse the whole graph in topological order.
    - components: Traverse topologically each component before traversing the next component.
    - uniform: Traverse from the first (child) jobs of each connected component, choosing a random reachable parent job.

    :param problem_instance: The problem instance whose job-graph to traverse.
    :param graph: The existing job-graph to traverse.
    :param search: Determines the type of search to use for the traversal. Options are "topological" (default),
                   "components" or "uniform".
    :param yield_state: Determines whether a search state is yielded with each node. The search state is
    (node, i_gen) for topological search, (node, i_comp, i_gen) for components search and (node, ) for uniform search.
    :return: Each job from the instance graph in an order given by the specified search type.
    """
    if problem_instance is None and graph is None:
        print_error("Neither problem instance nor job-graph were given to traverse")

    if search not in ["topological", "components", "uniform"]:
        print_error(f"Unrecognized search kind: {search}")
        return

    def enumerate_topological_generations(g):
        for i, gen in enumerate(nx.topological_generations(g)):
            for n in gen:
                yield i, n

    if graph is None:
        graph = build_instance_graph(problem_instance)

    match search:
        case "topological":
            for i_gen, node in enumerate_topological_generations(graph):
                yield (node, i_gen) if yield_state else node
        case "components":
            for i_comp, component in enumerate(nx.weakly_connected_components(graph)):
                for i_gen, node in enumerate_topological_generations(graph.subgraph(component)):
                    yield (node, i_comp, i_gen) if yield_state else node
        case "uniform":
            def pop(f):
                """
                Pops a uniformly-random node from the frontier.
                :param f: The frontier.
                :return: The popped node.
                """
                i_node = random.randint(0, len(f) - 1)
                n = f[i_node]
                f[i_node], f[-1] = f[-1], None
                f.pop()
                return n

            first_generation = nx.topological_generations(graph).send(None)  # `send` will yield the first generation
            frontier = list(first_generation)
            while frontier:
                node = pop(frontier)
                yield (node, ) if yield_state else node
                frontier += [e[1] for e in graph.out_edges(node)]
