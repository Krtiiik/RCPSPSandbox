import functools
import itertools
import math
from collections import namedtuple
from typing import Iterable

import matplotlib.colors
import matplotlib.spines
import matplotlib.patches
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.ticker import MaxNLocator

import utils
from bottlenecks.evaluations import EvaluationKPIs
from bottlenecks.utils import compute_resource_consumption
from instances.problem_instance import ProblemInstance, compute_component_jobs, compute_resource_availability
from utils import interval_overlap_function
from solver.solution import Solution


Interval = namedtuple("Interval", ("key", "start", "end"))
Deadline = namedtuple("Deadline", ("id_root_job", "time"))
PlotParameters = namedtuple("PlotParameters", ("origin", "horizon", "colormap", "dividers"))

T_StepFunction = list[tuple[int, int, int]]


COLOR_JOB = "green"
COLOR_JOB_DISABLED = "gray"
COLOR_RESOURCE_CAPACITY = "silver"
COLOR_RESOURCE_CONSUMPTION = "green"
COLOR_RESOURCE_CONSUMPTION_DIMMED = "gray"
COLOR_DIVIDERS = "lightgray"


class ColorMap:
    _cm = utils.ColorMap()
    _job_component_index: dict[int, int]
    _highlight: set[int]

    def __init__(self, instance: ProblemInstance,
                 highlight: Iterable[int] = None,
                 ):
        component_jobs = {root_job.id_job: [j.id_job for j in c_jobs] for root_job, c_jobs in compute_component_jobs(instance).items()}

        self._job_component_index = dict()
        for i_comp, id_root_job in enumerate(sorted(component_jobs)):
            for id_job in component_jobs[id_root_job]:
                self._job_component_index[id_job] = i_comp

        self._highlight = set(highlight) if highlight else None

    def interval(self, job_id):
        if self._highlight:
            return self.__job_component_color(job_id) if job_id in self._highlight else COLOR_JOB_DISABLED
        else:
            return self.__job_component_color(job_id)

    def resource_capacity(self, resource_key):
        return COLOR_RESOURCE_CAPACITY

    def resource_consumption(self, resource_key, job_id=None):
        return COLOR_RESOURCE_CONSUMPTION if not job_id else self.__job_component_color(job_id)

    def resource_consumption_mixed(self, resource_key, job_id=None):
        return COLOR_RESOURCE_CONSUMPTION_DIMMED if not job_id else self.__job_component_color(job_id)

    def component(self, job_id):
        return self.__job_component_color(job_id)

    def fill_shade_of(self, color):
        r, g, b, a = matplotlib.colors.to_rgba(color)
        fill_color = (
            clamp(r + 0.1, 0, 1),
            clamp(g + 0.1, 0, 1),
            clamp(b + 0.1, 0, 1),
            a,
        )
        return fill_color

    def __job_component_color(self, job_id):
        return self._cm[self._job_component_index[job_id]]


def plot_evaluation(evaluation, block: bool = True, save_as: list[str] = None, dimensions: list[tuple[int, int]] = ((8, 11), (8, 11))):
    def get(iterable, i): return None if iterable is None else iterable[i]
    horizon = max(max(int_sol.end for int_sol in evaluation.base_solution.job_interval_solutions.values()),
                  max(int_sol.end for int_sol in evaluation.solution.job_interval_solutions.values()))
    plot_solution(evaluation.base_solution, block=block, save_as=get(save_as, 0), dimensions=get(dimensions, 0), horizon=horizon)
    plot_solution(evaluation.solution, block=block, save_as=get(save_as, 1), dimensions=get(dimensions, 1), horizon=horizon)


def plot_evaluations(evaluations_kpis: list[EvaluationKPIs | list[EvaluationKPIs]],
                     ):
    if not isinstance(evaluations_kpis[0], list):
        evaluations_kpis = [evaluations_kpis]

    plt.grid(which='both', axis='both', ls='--')
    for evaluation_kpis in evaluations_kpis:
        costs = [evaluation_kpi.cost for evaluation_kpi in evaluation_kpis]
        improvements = [evaluation_kpi.improvement for evaluation_kpi in evaluation_kpis]
        plt.scatter(costs, improvements)

    def get_name(_evaluation):
        _alg, _settings = _evaluation.by.split('--')
        _alg = ''.join(itertools.filterfalse(str.islower, _alg))
        return f'{_alg}-{_settings}'

    interesting_evaluation_kpis = [kpi for kpis in evaluations_kpis
                                   for kpi in [
                                       min(kpis, key=lambda e_kpi: e_kpi.cost),
                                       max(kpis, key=lambda e_kpi: e_kpi.cost),
                                       min(kpis, key=lambda e_kpi: e_kpi.improvement),
                                       max(kpis, key=lambda e_kpi: e_kpi.improvement),
                                   ]]
    texts = [plt.text(evaluation_kpi.cost, evaluation_kpi.improvement, get_name(evaluation_kpi.evaluation), ha='center', va='top')
             for evaluation_kpi in interesting_evaluation_kpis]
    adjust_text(texts, expand_axes=True, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.xlabel("Cost")
    plt.ylabel("Improvement")
    plt.title(evaluations_kpis[0][0].evaluation.base_instance.name)
    plt.legend(labels=[evaluations_kpis_iter[0].evaluation.alg_string for evaluations_kpis_iter in evaluations_kpis])

    plt.show()


def plot_solution(solution: Solution,
                  highlight: Iterable[int] = None,
                  split_consumption: bool = False,
                  block: bool = True,
                  save_as: str = None,
                  dimensions: tuple[int, int] = (8, 11),
                  component_legends: dict[int, str] = None,
                  orderify_legends: bool = False,
                  horizon: int = None,
                  job_interval_levels: dict[int, int] = None,
                  ):
    instance = solution.instance

    horizon = (24 * math.ceil(max(i.end for i in __build_intervals(solution)) / 24) if horizon is None else horizon)
    params = PlotParameters(0, horizon, ColorMap(instance, highlight), [0]+list(range(6, horizon, 8)))

    f: plt.Figure
    axarr: list[plt.Axes]
    resource_count = len(instance.resources)
    f, axarr = plt.subplots(1 + resource_count, sharex="col", height_ratios=__compute_height_ratios(solution, horizon))

    __intervals_panel(solution, axarr[0], params,
                      component_legends=component_legends, orderify_legends=orderify_legends,
                      job_interval_levels=job_interval_levels)
    __resources_panels(solution, axarr[1:], params,
                       split_consumption=split_consumption, highlight_consumption=highlight)

    f.tight_layout()
    f.subplots_adjust(hspace=0.1, top=0.95, bottom=0.05, left=0.1, right=0.95)
    f.set_size_inches(dimensions)
    f.set_dpi(300)

    if save_as:
        plt.savefig(save_as)
    else:
        plt.show(block=block)


def __compute_height_ratios(solution, horizon):
    interval_panel_height = [__compute_max_interval_overlap(list(__build_intervals(solution)))]
    resource_panel_heights = [max(c for s, e, c in compute_resource_availability(r, solution.instance, horizon)) // 5
                              for r in solution.instance.resources]
    return interval_panel_height + resource_panel_heights


def plot_intervals(solution: Solution,
                   highlight: Iterable[int] = None,
                   block: bool = False,
                   save_as: str = None,
                   dimensions: tuple[int, int] = (8, 11),
                   component_legends: dict[int, str] = None,
                   orderify_legends: bool = False,
                   horizon: int = None,
                   job_interval_levels: dict[int, int] = None,
                   ):
    instance = solution.instance

    horizon = (24 * math.ceil(max(i.end for i in __build_intervals(solution)) / 24) if horizon is None else horizon)
    params = PlotParameters(0, horizon, ColorMap(instance, highlight), list(range(6, horizon, 8)))

    f: plt.Figure
    axarr: plt.Axes
    f, axarr = plt.subplots(1)

    __intervals_panel(solution, axarr, params,
                      component_legends=component_legends, orderify_legends=orderify_legends,
                      job_interval_levels=job_interval_levels)

    f.tight_layout()
    f.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95)
    f.set_size_inches(dimensions)
    f.set_dpi(300)

    if save_as:
        plt.savefig(save_as)
    else:
        plt.show(block=block)


def plot_resources(solution: Solution,
                   split_consumption: bool = False,
                   block: bool = False,
                   save_as: str = None,
                   dimensions: tuple[int, int] = (8, 11),
                   horizon: int = None,
                   ):
    instance = solution.instance

    horizon = (24 * math.ceil(max(i.end for i in __build_intervals(solution)) / 24) if horizon is None else horizon)
    params = PlotParameters(0, horizon, ColorMap(instance), list(range(6, horizon, 8)))

    f: plt.Figure
    axarr: list[plt.Axes]
    resource_count = len(instance.resources)
    f, axarr = plt.subplots(resource_count, sharex="col")

    __resources_panels(solution, axarr, params, split_consumption)

    f.tight_layout()
    f.subplots_adjust(hspace=0.1, top=0.95, bottom=0.05, left=0.1, right=0.95)
    f.set_size_inches(dimensions)
    f.set_dpi(300)

    if save_as:
        plt.savefig(save_as)
    else:
        plt.show(block=block)


def __intervals_panel(solution: Solution,
                      axes: plt.Axes, params: PlotParameters,
                      component_legends: dict[int, str] = None,
                      orderify_legends: bool = False,
                      job_interval_levels: dict[int, int] = None,
                      ):
    instance = solution.instance
    deadlines = {j.id_job: j.due_date for j in instance.jobs}
    deadlines = [Deadline(c.id_root_job, deadlines[c.id_root_job]) for c in instance.components]
    intervals = list(__build_intervals(solution))

    # Plotting
    __plot_dividers(params.dividers, axes, params)
    __plot_deadlines(deadlines, axes, params)
    __plot_intervals(intervals, axes, params, job_interval_levels=job_interval_levels)

    # Styling
    axes.set_ylabel("Intervals")
    axes.yaxis.set_label_coords(-0.05, 0.5)
    axes.yaxis.set_ticks([])
    axes.xaxis.tick_top()
    axes.xaxis.set_tick_params(labeltop=True)
    axes.set_xticks(params.dividers, labels=map(str, params.dividers), rotation=(0 if params.horizon < 100 else 90))
    axes.autoscale(True, axis='x', tight=True)

    if component_legends is None and orderify_legends:
        component_legends = {rj: f'$\mathcal{{O}}_{1+i} = {rj}$' for i, rj in enumerate(sorted(c.id_root_job for c in solution.instance.components))}
    legend_elements = [matplotlib.patches.Patch(color=params.colormap.component(d.id_root_job), label=str(d.id_root_job))
                       for d in deadlines]
    labels = [component_legends[d.id_root_job] for d in deadlines] if component_legends is not None else None
    axes.legend(handles=legend_elements, labels=labels, fancybox=True, shadow=True)


def __resources_panels(solution: Solution,
                       axarr: list[plt.Axes], params: PlotParameters,
                       split_consumption: bool = False,
                       highlight_consumption: Iterable[int] = None,
                       ):
    instance = solution.instance
    if highlight_consumption:
        highlight_consumption = set(highlight_consumption)

    for i_resource, resource in enumerate(instance.resources):
        axes = axarr[i_resource]

        availability = compute_resource_availability(resource, instance, params.horizon)

        __plot_dividers(params.dividers, axes, params)
        __plot_step_function(availability, axes, params, color=params.colormap.resource_capacity(resource.key), fill=True)

        if split_consumption:
            component_jobs = {root_job.id_job: {j.id_job for j in jobs}
                              for root_job, jobs in compute_component_jobs(instance).items()}
            if highlight_consumption:
                highlight_component_jobs = {root_job_id: jobs & highlight_consumption for root_job_id, jobs in component_jobs.items()}
                remaining_jobs = set(j.id_job for j in instance.jobs) - highlight_consumption
                consumptions = [compute_resource_consumption(solution.instance, solution, resource, selected=highlight_component_jobs[root_job_id])
                                for root_job_id in sorted(highlight_component_jobs)] \
                               + [compute_resource_consumption(solution.instance, solution, resource, selected=remaining_jobs)]
                accumulated_consumptions = map(functools.partial(interval_overlap_function, first_x=0, last_x=params.horizon),
                                               itertools.accumulate(consumptions))
                for id_root_job, consumption in reversed(list(zip(list(sorted(component_jobs)) + [None], accumulated_consumptions))):
                    __plot_step_function(consumption, axes, params, color=params.colormap.resource_consumption_mixed(resource.key, id_root_job), fill=True)
            else:
                consumptions = [compute_resource_consumption(solution.instance, solution, resource, selected=component_jobs[root_job_id])
                                for root_job_id in sorted(component_jobs)]
                accumulated_consumptions = map(functools.partial(interval_overlap_function, first_x=0, last_x=params.horizon),
                                               itertools.accumulate(consumptions))
                for id_root_job, consumption in reversed(list(zip(sorted(component_jobs), accumulated_consumptions))):
                    __plot_step_function(consumption, axes, params, color=params.colormap.resource_consumption(resource.key, job_id=id_root_job), fill=True)
        else:
            if highlight_consumption:
                highlighted_consumption = compute_resource_consumption(instance, solution, resource, selected=highlight_consumption)
                full_consumption = compute_resource_consumption(instance, solution, resource)
                __plot_step_function(full_consumption, axes, params, color=params.colormap.resource_consumption_mixed(resource.key), fill=True)
                __plot_step_function(highlighted_consumption, axes, params, color=params.colormap.resource_consumption(resource.key), fill=True)
            else:
                consumption = compute_resource_consumption(instance, solution, resource)
                __plot_step_function(consumption, axes, params, color=params.colormap.resource_consumption(resource.key), fill=True)

    for axes, resource in zip(axarr, instance.resources):
        axes.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10], integer=True))
        axes.set_ylabel(resource.key)
        axes.yaxis.set_label_coords(-0.05, 0.5)
        axes.set_xticks(params.dividers, labels=map(str, params.dividers), rotation=(0 if params.horizon < 100 else 90))
        axes.grid(which='both', axis='y', ls=':')


def __plot_intervals(intervals: Iterable[Interval], axes: plt.Axes, params: PlotParameters,
                     job_interval_levels: dict[int, int] = None,
                     ):
    interval_width = 16

    if job_interval_levels is None:
        interval_levels, max_level = __compute_interval_levels(intervals)
    else:
        interval_levels, max_level = job_interval_levels, max(job_interval_levels.values())

    for key, start, end in intervals:
        level = interval_levels[key]

        axes.plot([start, end], [level, level], linestyle='', marker='|', markeredgecolor='gray', markersize=interval_width)
        axes.hlines(level, start, end, colors=params.colormap.interval(key), lw=interval_width)
        axes.text(float(start + end) / 2, level, str(key), horizontalalignment='center', verticalalignment='center')


def __plot_deadlines(deadlines: list[Deadline], axes: plt.Axes, params: PlotParameters):
    scale = 0.5
    n = len(deadlines)
    for i, deadline in enumerate(deadlines):
        x = deadline.time + scale*(-n+1+i)/n
        axes.axvline(x, color=params.colormap.component(deadline.id_root_job), linestyle="--", lw=1)


def __plot_dividers(dividers: Iterable[int], axes: plt.Axes, params: PlotParameters):
    for x in dividers:
        axes.axvline(x, color=COLOR_DIVIDERS, linestyle="dotted", lw=1)


def __plot_step_function(function: T_StepFunction, axes: plt.Axes, params: PlotParameters,
                         fill: bool = False, color="green",
                         ):
    xs = [x for x, x1, v in function]
    vs = [v for x, x1, v in function]
    axes.step(xs, vs, where="post", color=color)
    if fill:
        axes.fill_between(xs, 0, vs, step="post", color=params.colormap.fill_shade_of(color))


def __compute_interval_levels(intervals: Iterable[Interval],
                              max_level: int = None,
                              ) -> tuple[dict[Interval, int], int]:
    import heapq

    intervals = sorted(intervals, key=lambda i: (i.start, i.end))
    if max_level is None:
        max_level = __compute_max_interval_overlap(intervals)

    h = [(intervals[0].start, lvl) for lvl in range(0, max_level)]
    heapq.heapify(h)

    levels = dict()
    for itv in intervals:
        lvl = heapq.heappop(h)[1]
        levels[itv.key] = lvl
        heapq.heappush(h, (itv.end, lvl))

    return {k: max_level - l for k, l in levels.items()}, max_level


def __compute_max_interval_overlap(intervals):
    events = sorted([(itv.start, +1) for itv in intervals] + [(itv.end, -1) for itv in intervals])
    return max(itertools.accumulate(events, lambda cur, event: cur + event[1], initial=0))


def compute_shifting_interval_levels(solution_a: Solution, solution_b: Solution):
    def build_intervals(_job_ids, _interval_solutions):
        return list(Interval(_job_id, _interval_solutions[_job_id].start, _interval_solutions[_job_id].end) for _job_id in _job_ids)

    interval_solutions_a = solution_a.job_interval_solutions
    interval_solutions_b = solution_b.job_interval_solutions

    same = []
    different = []
    for job_id in interval_solutions_a.keys():
        start_a, end_a = interval_solutions_a[job_id].start, interval_solutions_a[job_id].end
        start_b, end_b = interval_solutions_b[job_id].start, interval_solutions_b[job_id].end

        if (start_a, end_a) == (start_b, end_b):
            same.append(job_id)
        else:
            different.append(job_id)

    same_intervals = build_intervals(same, interval_solutions_a)
    different_intervals_a = build_intervals(different, interval_solutions_a)
    different_intervals_b = build_intervals(different, interval_solutions_b)

    max_level = max(__compute_max_interval_overlap(different_intervals_a), __compute_max_interval_overlap(different_intervals_b))

    same_levels, same_max_level = __compute_interval_levels(same_intervals)
    different_levels_a = {k: v + same_max_level for k, v in __compute_interval_levels(different_intervals_a, max_level=max_level)[0].items()}
    # different_levels_b = {k: v + same_max_level for k, v in __compute_interval_levels(different_intervals_b, max_level=max_level)[0].items()}

    different_levels_b = dict()
    level_ends = {l: 0 for l in different_levels_a.values()}
    for interval in sorted(different_intervals_b, key=lambda itv: (itv.start, itv.end)):
        job_id, start, end = interval
        level_a = different_levels_a[job_id]
        if start < level_ends[level_a]:
            level = min(level_ends.items(), key=lambda kv: kv[1])[0]
        else:  # start >= level_ends[end_a]
            level = level_a

        different_levels_b[job_id] = level
        level_ends[level] = end

    return same_levels | different_levels_a, same_levels | different_levels_b


def __build_intervals(solution: Solution):
    return (Interval(j_id, j_interval.start, j_interval.end) for j_id, j_interval in solution.job_interval_solutions.items())


def last_from(iterator):
    *_, last = iterator
    return last


def clamp(num, low, high):
    return max(low, min(num, high))
