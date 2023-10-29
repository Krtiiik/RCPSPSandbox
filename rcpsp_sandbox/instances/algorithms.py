import itertools
from queue import Queue
import random
from typing import Generator

import networkx as nx

from instances.problem_instance import ProblemInstance, Job
from utils import print_error


def build_instance_graph(instance) -> nx.DiGraph:
    """
    Builds a job-graph of the given problem instance.
    :param instance: The instance to build the graph of. Any object with `jobs` and `precedences` properties can be given.
    :return: The oriented job-graph of the problem instance.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(job.id_job for job in instance.jobs)
    graph.add_edges_from((precedence.id_child, precedence.id_parent) for precedence in instance.precedences)
    return graph


def enumerate_topological_generations_nodes(graph):
    """
    Enumerates the topological generations of a given graph.
    :param graph: The graph whose topological generations to enumerate
    :return:
    """
    for i_gen, gen in enumerate(nx.topological_generations(graph)):
        for node in gen:
            yield i_gen, node


def topological_sort(graph: nx.DiGraph,
                     yield_state: bool = False):
    degrees = {v: d for v, d in graph.in_degree if d > 0}
    no_in = Queue()
    for v, d in graph.in_degree:
        if d == 0:
            no_in.put((None, v))

    while no_in:
        parent, n = no_in.get()
        for n_from, n_to in graph.edges(n):
            degrees[n_to] -= 1
            if degrees[n_to] == 0:
                no_in.put((n_from, degrees[n_to]))
                del degrees[n_to]
        yield (n, parent) if yield_state else n

    if no_in:
        print_error("The instance graph contains a cycle")


def uniform_traversal(graph,
                      yield_state: bool = False):
    def pop(f):  # Selects and pops a uniformly chosen node
        i_node = random.randint(0, len(f) - 1)
        n_rand = f[i_node]
        f[i_node], f[-1] = f[-1], None
        f.pop()
        return n_rand

    first_generation = nx.topological_generations(graph).send(None)  # `send` will yield the first generation
    frontier = list(first_generation)
    while frontier:
        node = pop(frontier)
        yield (node,) if yield_state else node
        frontier += [e[1] for e in graph.out_edges(node)]


def traverse_instance_graph(problem_instance: ProblemInstance = None,
                            graph: nx.DiGraph = None,
                            search: str = "topological generations",
                            yield_state: bool = False) -> Generator[Job, None, None]:
    """
    Traverses the job-graph of a given problem instance, yielding jobs in the order of visiting. The available
    search type options are:

    - topological generations (default): Traverse the graph by the order of topological generations.
    - components topological generations: Traverse topologically each weakly-connected component before traversing the next component.
    - topological: Traverse the whole graph in topological order.
    - uniform: Traverse from the first (child) jobs of each connected component, choosing a random reachable parent job.

    :param problem_instance: The problem instance whose job-graph to traverse.
    :param graph: The existing job-graph to traverse.
    :param search: Determines the type of search to use for the traversal. Options are "topological generations" (default),
                   "components topological generations", "topological" or "uniform".
    :param yield_state: Determines whether a search state is yielded with each node. The yielded search state is
    (node, i_gen) for topological generations, (node, i_comp, i_gen) for components topological generations,
    (node, parent) for topological and (node, ) for uniform.
    :return: Each job from the instance graph in an order given by the specified search type.
    """
    if problem_instance is None and graph is None:
        print_error("Neither problem instance nor job-graph were given to traverse")

    if search not in ["topological generations", "components topological generations", "topological", "uniform"]:
        print_error(f"Unrecognized search kind: {search}")
        return

    if graph is None:
        graph = build_instance_graph(problem_instance)

    match search:
        case "topological generations":
            for i_gen, node in enumerate_topological_generations_nodes(graph):
                yield (node, i_gen) if yield_state else node
        case "components topological generations":
            for i_comp, component in enumerate(nx.weakly_connected_components(graph)):
                for i_gen, node in enumerate_topological_generations_nodes(graph.subgraph(component)):
                    yield (node, i_comp, i_gen) if yield_state else node
        case "topological":
            yield from topological_sort(graph, yield_state=yield_state)
        case "uniform":
            yield from uniform_traversal(graph, yield_state=yield_state)


def compute_jobs_in_components(problem_instance: ProblemInstance) -> dict[int, list[Job]]:
    jobs_grouped = itertools.groupby(traverse_instance_graph(problem_instance, search="components topological generations", yield_state=True),
                                     key=lambda x: x[1])
    component_jobs_by_root_job: dict[int, list[Job]] = dict()
    for _i_comp, job_states in jobs_grouped:
        jobs = [job for job, _, _ in job_states]
        for component in problem_instance.components:
            root_job = component.id_root_job
            if any(job.id_job == root_job for job in jobs):
                component_jobs_by_root_job[root_job] = jobs
                break
            else:
                print_error("No root job specified for an existing component")`
                return {}
    return component_jobs_by_root_job
