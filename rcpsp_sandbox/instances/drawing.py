import itertools

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import matplotlib.pyplot

from instances.algorithms import build_instance_graph, traverse_instance_graph
from instances.problem_instance import ProblemInstance, Job


def draw_instance_graph(instance: ProblemInstance,
                        block: bool = False,
                        highlighted_nodes: set[int] or None = None,
                        save_as: str or None = None):
    graph = build_instance_graph(instance)
    is_planar, planar_graph = nx.check_planarity(graph)
    if is_planar:
        planar_graph.check_structure()
        node_locations = nx.combinatorial_embedding_to_pos(planar_graph)
        print("planar")
    else:
        node_locations = __compute_node_locations(graph)

    __draw_graph(graph, node_locations, block, highlighted_nodes=highlighted_nodes, save_as=save_as)


def __compute_node_locations(graph: nx.DiGraph) -> dict[Job, tuple[int, int]]:
    y_scale = 10

    node_locations = dict()
    traversed_nodes = list(traverse_instance_graph(graph=graph, search="components topological generations", yield_state=True))
    grouped_nodes = ((k_comp, k_gen, [n[0] for n in nodes])
                     for k_comp, comp in itertools.groupby(traversed_nodes, key=lambda n: n[1])
                     for k_gen, nodes in itertools.groupby(comp, key=lambda n: n[2]))
    for i_comp, i_gen, nodes in grouped_nodes:
        y_base = i_comp * 50
        x = i_gen * 100
        y_offset = ((y_scale * (len(nodes) - 1)) // 2)
        for i_node, node in enumerate(nodes):
            y = y_base + (y_scale * i_node) - y_offset
            node_locations[node] = (x, y)

    print(node_locations)
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


if __name__ == "__main__":
    import rcpsp_sandbox.instances.io as ioo
    inst = ioo.parse_psplib("../../../Data/RCPSP/extended/instance_11.rp", is_extended=True)
    draw_instance_graph(inst)
