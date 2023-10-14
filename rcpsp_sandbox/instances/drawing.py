import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import matplotlib.pyplot

from instances.problem_instance import ProblemInstance, Job


def draw_instance_graph(instance: ProblemInstance,
                        block: bool = False,
                        highlighted_nodes: set[int] or None = None,
                        save_as: str or None = None):
    graph = __build_graph(instance)
    is_planar, planar_graph = nx.check_planarity(graph)
    if is_planar:
        planar_graph.check_structure()
        node_locations = nx.combinatorial_embedding_to_pos(planar_graph)
        print("planar")
    else:
        node_locations = __compute_node_locations(graph)

    __draw_graph(graph, node_locations, block, highlighted_nodes=highlighted_nodes, save_as=save_as)


def __build_graph(instance: ProblemInstance) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(job.id_job for job in instance.jobs)
    graph.add_edges_from((precedence.id_child, precedence.id_parent) for precedence in instance.precedences)
    return graph


def __compute_node_locations(graph: nx.DiGraph) -> dict[Job, tuple[int, int]]:
    node_locations = dict()
    for i, component in enumerate(nx.weakly_connected_components(graph)):
        generations = nx.topological_generations(graph.subgraph(component))
        y_base = i * 50
        for gen_i, generation in enumerate(generations):
            x = gen_i * 100
            y_scale = 10
            y_offset = ((y_scale * (len(generation) - 1)) // 2)
            for j, node in enumerate(generation):
                y = y_base + (y_scale * j) - y_offset
                node_locations[node] = (x, y)

    return node_locations


def __draw_graph(graph: nx.DiGraph,
                 node_locations: dict[Job, tuple[int, int]],
                 block: bool,
                 highlighted_nodes: set[int] or None = None,
                 save_as: str or None = None) -> None:
    if highlighted_nodes is None:
        highlighted_nodes = set()

    matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.gca()
    for id_job, loc in node_locations.items():
        # ax.add_patch(matplotlib.patches.Circle(loc, 2, color='b'))
        matplotlib.pyplot.text(*loc, str(id_job), ha='center', va="center", size=5,
                               bbox=dict(boxstyle="round",
                                         ec="red",
                                         fc=("green" if id_job in highlighted_nodes else "lightcoral")))

    edge_lines = [[node_locations[e[0]], node_locations[e[1]]] for e in graph.edges]
    ax.add_collection(matplotlib.collections.LineCollection(edge_lines))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.autoscale()

    if save_as is not None:
        matplotlib.pyplot.savefig(save_as, dpi=300)

    plt.show(block=block)
