import functools
import heapq
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib

from instances.algorithms import build_instance_graph, traverse_instance_graph, compute_earliest_completion_times
from instances.problem_instance import ProblemInstance, Job, compute_component_jobs
from matplotlib.axes import Axes

from utils import print_error, ColorMap


def draw_instance_graph(instance: ProblemInstance = None,
                        graph: nx.DiGraph = None,
                        block: bool = False,
                        highlighted_nodes: set[int] or None = None,
                        highlight_component_roots: bool = False,
                        save_as: str or None = None):
    if graph is None:
        if instance is not None:
            graph = build_instance_graph(instance)
        else:
            print_error("No instance nor graph were given to draw")
            return

    # is_planar, planar_graph = nx.check_planarity(graph)
    # if is_planar:
    #     planar_graph.check_structure()
    #     node_locations = nx.combinatorial_embedding_to_pos(planar_graph)
    #     print("planar")
    # else:
    #     node_locations = __compute_node_locations(graph)

    node_locations = __compute_node_locations(graph)

    if highlight_component_roots:
        highlighted_nodes = [c.id_root_job for c in instance.components]

    __draw_graph(graph, node_locations, block, highlighted_nodes=highlighted_nodes, save_as=save_as)


def draw_components_graph(instance: ProblemInstance,
                          save_as: str or None = None):
    jobs_by_id = {j.id_job: j for j in instance.jobs}
    component_jobs = compute_component_jobs(instance)
    earliest_completion_times = compute_earliest_completion_times(instance)

    layered = []
    ns = []
    for i_comp, component in enumerate(sorted(instance.components, key=lambda c: c.id_root_job)):
        root_job = jobs_by_id[component.id_root_job]
        intervals = [(earliest_completion_times[job] - job.duration, earliest_completion_times[job], job.id_job) for job in component_jobs[root_job]]
        events = sorted([(itv[0], +1) for itv in intervals]
                        + [(itv[1], -1) for itv in intervals])
        n = l = 0
        for e in events:
            l += e[1]
            if l > n:
                n = l
        intervals = sorted(intervals, key=lambda x: x[0])
        smin = intervals[0][0]
        h = []
        for lvl in range(0, n):
            heapq.heappush(h, (smin, lvl))
        layered.append([])
        for itv in intervals:
            lvl = heapq.heappop(h)[1]
            layered[-1].append((itv[0], itv[1], str(itv[2]), lvl))
            heapq.heappush(h, (itv[1], lvl))

        ns.append(n)

    SCALE = 1
    f, axarr = plt.subplots(len(instance.components),
                            sharex='col',
                            # gridspec_kw=dict(height_ratios=[n + 1.5 for n in ns]),
                            height_ratios=[SCALE*n + 1.5 for n in ns],
                            )
    f.subplots_adjust(hspace=0)
    f.set_figheight(SCALE*sum(ns)/3)

    cm = ColorMap()

    horizon = max(earliest_completion_times[job] + job.duration for job in instance.jobs)
    for i_comp, component in enumerate(sorted(instance.components, key=lambda c: c.id_root_job)):
        n = ns[i_comp]
        ax = axarr[i_comp] if not isinstance(axarr, Axes) else axarr
        ax.autoscale(enable=None, axis="y", tight=True)
        color = cm[i_comp]
        for x in range(0, horizon, 5):
            ax.vlines(x, 0, 1, transform=ax.get_xaxis_transform(), colors="lightgray", linestyle="dotted", lw=1)
        ax.set_yticks([])
        ax.set_yticklabels([])
        for itv in layered[i_comp]:
            ax.plot([itv[0], itv[1]], [SCALE*(n - itv[3]), SCALE*(n - itv[3])], marker='|', markeredgecolor=(0.52, 0.52, 0.52), markersize=8, color=color)
            ax.hlines(SCALE*(n - itv[3]), itv[0], itv[1], colors=color, lw=8)
            ax.text(float(itv[0] + itv[1]) / 2, SCALE*(n - itv[3]), itv[2], horizontalalignment='center', verticalalignment='center')
        ax.vlines(jobs_by_id[component.id_root_job].due_date, 0, SCALE*(n+1), transform=ax.get_xaxis_transform(), colors="red", linestyle='-', lw=1)

    plt.margins(0.05)
    plt.xlim(left=0)
    plt.tight_layout()

    if save_as is not None:
        plt.savefig(save_as)

    plt.show()


def __compute_node_locations(graph: nx.DiGraph) -> dict[Job, tuple[int, int]]:
    y_scale = 10
    gen_diff = 100

    node_locations = dict()
    traversed_nodes = list(traverse_instance_graph(graph=graph, search="components topological generations", yield_state=True))

    comp_gen_nodes_dict: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i_comp, comp in itertools.groupby(traversed_nodes, key=lambda n: n[1]):
        for k_gen, nodes in itertools.groupby(comp, key=lambda n: n[2]):
            comp_gen_nodes_dict[i_comp][k_gen] = [n[0] for n in nodes]
    comp_gen_nodes: list[list[list[int]]] = [None] * len(comp_gen_nodes_dict)
    for i_comp, comp in sorted(comp_gen_nodes_dict.items()):
        comp_gen_nodes[i_comp] = [None] * len(comp)
        for i_gen, gen in sorted(comp.items()):
            comp_gen_nodes[i_comp][i_gen] = gen

    component_heights: list[int] = [y_scale * max(len(gen) for gen in comp) for comp in comp_gen_nodes]
    component_base_y_offsets: list[int] = [0] + list(itertools.accumulate([((component_heights[i] // 2) + (component_heights[i+1] // 2)) for i in range(len(component_heights) - 1)]))
    for i_comp, comp in enumerate(comp_gen_nodes):
        y_base = component_base_y_offsets[i_comp]
        for i_gen, gen in enumerate(comp):
            x = i_gen * gen_diff
            y_offset = ((y_scale * (len(gen) - 1)) // 2)
            for i_node, node in enumerate(gen):
                y = y_base + (y_scale * i_node) - y_offset
                node_locations[node] = (x, y)

    return node_locations


def __draw_graph(graph: nx.DiGraph,
                 node_locations: dict[Job, tuple[int, int]],
                 block: bool,
                 highlighted_nodes: set[int] or None = None,
                 save_as: str or None = None) -> None:
    if highlighted_nodes is None:
        highlighted_nodes = set()

    x_max, y_max = max(x[0] for x in node_locations.values()), max(x[1] for x in node_locations.values())
    plt.figure(
        figsize=(x_max / 100, y_max / 10),
        dpi=300,
    )

    ax = plt.gca()
    for id_job, loc in node_locations.items():
        # ax.add_patch(matplotlib.patches.Circle(loc, 2, color='b'))
        plt.text(*loc, str(id_job), ha='center', va="center", size=5,
                 bbox=dict(boxstyle="round",
                           ec="red",
                           fc=("green" if id_job in highlighted_nodes else "lightcoral")))

    edge_lines = [[node_locations[e[0]], node_locations[e[1]]] for e in graph.edges]
    ax.add_collection(matplotlib.collections.LineCollection(edge_lines))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.autoscale()

    if save_as is not None:
        plt.savefig(save_as, dpi=300)

    plt.show(block=block)


if __name__ == "__main__":
    import rcpsp_sandbox.instances.io as ioo
    inst = ioo.parse_psplib("../../../Data/RCPSP/extended/instance_11.rp", is_extended=True)
    draw_instance_graph(inst)
