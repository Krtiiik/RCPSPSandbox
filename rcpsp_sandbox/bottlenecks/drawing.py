import functools
import itertools
import math
from collections import namedtuple
from typing import Iterable

import numpy as np
import matplotlib.colors
import matplotlib.spines
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import utils
from bottlenecks.evaluations import EvaluationKPIs, Evaluation
from bottlenecks.improvements import left_closure
from bottlenecks.utils import compute_resource_consumption
from instances.problem_instance import ProblemInstance, compute_component_jobs, compute_resource_availability, \
    compute_resource_periodical_availability
from utils import interval_overlap_function, flatten, __is_pareto_efficient
from solver.solution import Solution


Interval = namedtuple("Interval", ("key", "start", "end"))
Deadline = namedtuple("Deadline", ("id_root_job", "time"))
PlotParameters = namedtuple("PlotParameters", ("origin", "horizon", "colormap", "dividers"))

T_StepFunction = list[tuple[int, int, int]]


COLOR_JOB = "green"
COLOR_JOB_DISABLED = "gray"
COLOR_RESOURCE_CAPACITY = "silver"
COLOR_RESOURCE_CAPACITY_REDUCED = "indianred"
COLOR_RESOURCE_CONSUMPTION = "green"
COLOR_RESOURCE_CONSUMPTION_DIMMED = "gray"
COLOR_RESOURCE_CONSUMPTION_HIGHLIGHT = "darkturquoise"
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

    def resource_capacity_reduced(self, resource_key):
        return COLOR_RESOURCE_CAPACITY_REDUCED

    def resource_consumption(self, resource_key, job_id=None):
        return COLOR_RESOURCE_CONSUMPTION if not job_id else self.__job_component_color(job_id)

    def resource_consumption_mixed(self, resource_key, job_id=None):
        return COLOR_RESOURCE_CONSUMPTION_DIMMED if not job_id else self.__job_component_color(job_id)

    def resource_consumption_highlight(self, resource_key):
        return COLOR_RESOURCE_CONSUMPTION_HIGHLIGHT

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


def plot_evaluation_solution_comparison(evaluation: Evaluation,
                                        block: bool = True,
                                        save_as: list[str] = None,
                                        dimensions: list[tuple[int, int]] = ((8, 11), (8, 11)),
                                        highlight_addition: bool = False,
                                        highlight: bool = False,
                                        ):
    def get(iterable, i): return None if iterable is None else iterable[i]
    horizon = max(max(int_sol.end for int_sol in evaluation.base_solution.job_interval_solutions.values()),
                  max(int_sol.end for int_sol in evaluation.solution.job_interval_solutions.values()))
    levels_a, levels_b = compute_shifting_interval_levels(evaluation.base_solution, evaluation.solution)
    if highlight:
        highlight_a = left_closure(evaluation.base_instance.target_job, evaluation.base_instance, evaluation.base_solution)
        highlight_b = left_closure(evaluation.base_instance.target_job, evaluation.modified_instance, evaluation.solution)
    else:
        highlight_a = highlight_b = None

    plot_solution(evaluation.base_solution, block=block, save_as=get(save_as, 0), dimensions=get(dimensions, 0), horizon=horizon, job_interval_levels=levels_a, highlight_non_periodical_consumption=highlight_addition, highlight=highlight_a)
    plot_solution(evaluation.solution, block=block, save_as=get(save_as, 1), dimensions=get(dimensions, 1), horizon=horizon, job_interval_levels=levels_b, highlight_non_periodical_consumption=highlight_addition, highlight=highlight_b)


def plot_evaluations(instance_evaluations_kpis: dict[str, list[list[EvaluationKPIs]]],
                     value_axes: tuple[str, str],
                     pareto_front: bool = False,
                     evaluations_kpis_to_annotate: Iterable[str] = None,
                     annotate_extremes: bool = False,
                     ncols: int = 2,
                     block: bool = True,
                     save_as: str = None,
                     ):
    f: plt.Figure
    axarr: np.ndarray[plt.Axes]
    nrows = math.ceil(len(instance_evaluations_kpis) / ncols)
    f, axarr = plt.subplots(nrows=nrows, ncols=ncols, height_ratios=nrows*[1])

    for instance_name, axes in zip(instance_evaluations_kpis, axarr.flatten()):
        __plot_algorithms_evaluations_kpis(instance_evaluations_kpis[instance_name], axes, title=instance_name,
                                           value_axes=value_axes, pareto_front=pareto_front, evaluations_kpis_to_annotate=evaluations_kpis_to_annotate,
                                           annotate_extremes=annotate_extremes)

    for axes in axarr.flatten()[len(instance_evaluations_kpis):]:
        axes.set_axis_off()

    f.legend(labels=[evaluations_kpis[0].evaluation.alg_string for evaluations_kpis in instance_evaluations_kpis[next(iter(instance_evaluations_kpis))]],
             loc='lower center'
             )

    f.tight_layout()
    f.set_size_inches(4*ncols, nrows*3)
    f.subplots_adjust(hspace=0.5,
                      wspace=0.3,
                      top=0.95,
                      bottom=0.15,
                      left=0.15,
                      right=0.95
                      )

    if save_as:
        plt.savefig(save_as)
        plt.close()
    else:
        plt.show(block=block)


def __plot_algorithms_evaluations_kpis(algorithms_evaluations_kpis: list[list[EvaluationKPIs]], axes: plt.Axes,
                                       value_axes: tuple[str, str],
                                       pareto_front: bool,
                                       annotate_extremes: bool,
                                       title: str = None,
                                       evaluations_kpis_to_annotate: Iterable[str] = None,
                                       ):
    value_extractors = {
        "cost": (lambda ev: ev.cost),
        "improvement": (lambda ev: ev.improvement),
        "schedule difference": (lambda ev: ev.schedule_difference),
        "duration": (lambda ev: ev.evaluation.duration)
    }

    if pareto_front:
        algorithms_evaluations_kpis = [utils.pareto_front_kpis(alg_evaluations_kpis, value_axes[0], value_axes[1])
                                       for alg_evaluations_kpis in algorithms_evaluations_kpis]

    x_extractor, y_extractor = value_extractors[value_axes[0]], value_extractors[value_axes[1]]
    evaluations_kpis_to_annotate = set(evaluations_kpis_to_annotate if evaluations_kpis_to_annotate is not None
                                       else ({
                                           min(flatten(algorithms_evaluations_kpis), key=x_extractor).evaluation.by,
                                           max(flatten(algorithms_evaluations_kpis), key=x_extractor).evaluation.by,
                                           min(flatten(algorithms_evaluations_kpis), key=y_extractor).evaluation.by,
                                           max(flatten(algorithms_evaluations_kpis), key=y_extractor).evaluation.by,
                                       } if annotate_extremes else ()))
    markers = itertools.cycle(['s', 'o', '^', 'v', '+', 'x'])

    axes.grid(which='both', axis='both', ls='--')
    for evaluations_kpis, marker in zip(algorithms_evaluations_kpis, markers):
        xs = [x_extractor(evaluation_kpi) for evaluation_kpi in evaluations_kpis]
        ys = [y_extractor(evaluation_kpi) for evaluation_kpi in evaluations_kpis]
        axes.scatter(xs, ys, marker=marker)

    def get_name(_evaluation): return f'{"".join(filter(str.isupper, _evaluation.alg_string))}-{_evaluation.settings_string}'
    annotations = [axes.text(x_extractor(e_kpis), y_extractor(e_kpis), get_name(e_kpis.evaluation), ha='center', va='center')
                   for e_kpis in flatten(algorithms_evaluations_kpis)
                   if e_kpis.evaluation.by in evaluations_kpis_to_annotate]

    axes.set_xlabel(value_axes[0].capitalize())
    axes.xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True, min_n_ticks=1))
    axes.set_xlim(xmin=0)
    axes.set_ylabel(value_axes[1].capitalize())
    axes.yaxis.set_label_coords(-0.18, 0.5)
    axes.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True, min_n_ticks=1))
    axes.set_ylim(ymin=0)
    if title is not None:
        axes.set_title(title)


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
                  highlight_non_periodical_consumption: bool = False,
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
                       split_consumption=split_consumption, highlight_consumption=highlight,
                       highlight_non_periodical_consumption=highlight_non_periodical_consumption)

    f.suptitle(solution.instance.name)
    f.tight_layout()
    f.subplots_adjust(hspace=0.1, top=0.92, bottom=0.05, left=0.1, right=0.95)
    f.set_size_inches(dimensions)
    f.set_dpi(300)

    if save_as:
        plt.savefig(save_as)
        plt.close()
    else:
        plt.show(block=block)


def __compute_height_ratios(solution, horizon):
    interval_panel_height = [3+math.ceil(1.5*__compute_max_interval_overlap(list(__build_intervals(solution))))]
    resource_panel_heights = [1+(max(c for s, e, c in compute_resource_availability(r, solution.instance, horizon)) // 5)
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
        plt.close()
    else:
        plt.show(block=block)


def plot_resources(solution: Solution,
                   split_consumption: bool = False,
                   block: bool = False,
                   save_as: str = None,
                   dimensions: tuple[int, int] = (8, 11),
                   horizon: int = None,
                   highlight_non_periodical_consumption: bool = False,
                   ):

    instance = solution.instance

    horizon = (24 * math.ceil(max(i.end for i in __build_intervals(solution)) / 24) if horizon is None else horizon)
    params = PlotParameters(0, horizon, ColorMap(instance), list(range(6, horizon, 8)))

    f: plt.Figure
    axarr: list[plt.Axes]
    resource_count = len(instance.resources)
    f, axarr = plt.subplots(resource_count, sharex="col")

    __resources_panels(solution, axarr, params,
                       split_consumption=split_consumption,
                       highlight_non_periodical_consumption=highlight_non_periodical_consumption)

    f.tight_layout()
    f.subplots_adjust(hspace=0.1, top=0.95, bottom=0.05, left=0.1, right=0.95)
    f.set_size_inches(dimensions)
    f.set_dpi(300)

    if save_as:
        plt.savefig(save_as)
        plt.close()
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
                       highlight_non_periodical_consumption: bool = False,
                       ):
    instance = solution.instance
    if highlight_consumption:
        highlight_consumption = set(highlight_consumption)

    for i_resource, resource in enumerate(instance.resources):
        axes = axarr[i_resource]

        availability = compute_resource_availability(resource, instance, params.horizon)

        __plot_dividers(params.dividers, axes, params)
        if not split_consumption and highlight_non_periodical_consumption:
            periodical_availability = compute_resource_periodical_availability(resource, params.horizon)
            full_periodical_availability = interval_overlap_function(periodical_availability, first_x=0, last_x=params.horizon)
            __plot_step_function(full_periodical_availability, axes, params, color=params.colormap.resource_capacity_reduced(resource.key), fill=True)

        __plot_step_function(availability, axes, params, color=params.colormap.resource_capacity(resource.key), fill=True)

        if split_consumption:
            component_jobs = {root_job.id_job: {j.id_job for j in jobs}
                              for root_job, jobs in compute_component_jobs(instance).items()}
            if highlight_consumption:
                highlight_component_jobs = {root_job_id: jobs & highlight_consumption for root_job_id, jobs in component_jobs.items()}
                remaining_jobs = set(j.id_job for j in instance.jobs) - highlight_consumption
                consumptions = [compute_resource_consumption(solution.instance, solution, resource, selected=highlight_component_jobs[root_job_id], horizon=params.horizon)
                                for root_job_id in sorted(highlight_component_jobs)] \
                               + [compute_resource_consumption(solution.instance, solution, resource, selected=remaining_jobs, horizon=params.horizon)]
                accumulated_consumptions = map(functools.partial(interval_overlap_function, first_x=0, last_x=params.horizon),
                                               itertools.accumulate(consumptions))
                for id_root_job, consumption in reversed(list(zip(list(sorted(component_jobs)) + [None], accumulated_consumptions))):
                    __plot_step_function(consumption, axes, params, color=params.colormap.resource_consumption_mixed(resource.key, id_root_job), fill=True)
            else:
                consumptions = [compute_resource_consumption(solution.instance, solution, resource, selected=component_jobs[root_job_id], horizon=params.horizon)
                                for root_job_id in sorted(component_jobs)]
                accumulated_consumptions = map(functools.partial(interval_overlap_function, first_x=0, last_x=params.horizon),
                                               itertools.accumulate(consumptions))
                for id_root_job, consumption in reversed(list(zip(sorted(component_jobs), accumulated_consumptions))):
                    __plot_step_function(consumption, axes, params, color=params.colormap.resource_consumption(resource.key, job_id=id_root_job), fill=True)
        else:
            if highlight_consumption:
                highlighted_consumption = compute_resource_consumption(instance, solution, resource, selected=highlight_consumption, horizon=params.horizon)
                full_consumption = compute_resource_consumption(instance, solution, resource, horizon=params.horizon)
                __plot_step_function(full_consumption, axes, params, color=params.colormap.resource_consumption_mixed(resource.key), fill=True)
                __plot_step_function(highlighted_consumption, axes, params, color=params.colormap.resource_consumption(resource.key), fill=True)
            else:
                if highlight_non_periodical_consumption:
                    def compute_capped_consumption(_start, _end, _c):
                        overlapping_availability = [(_s, _e, _a) for _s, _e, _a in full_periodical_availability if
                                                    utils.intervals_overlap((_start, _end), (_s, _e))]
                        return min((_a for _s, _e, _a in overlapping_availability), default=0)

                    consumption = unitize_intervals(compute_resource_consumption(instance, solution, resource, horizon=params.horizon))
                    capped_consumption = [(s, e, min(c, compute_capped_consumption(s, e, c))) for s, e, c in consumption]
                    __plot_step_function(consumption, axes, params, color=params.colormap.resource_consumption_highlight(resource.key), fill=True)
                    __plot_step_function(capped_consumption, axes, params, color=params.colormap.resource_consumption(resource.key), fill=True)
                else:
                    consumption = compute_resource_consumption(instance, solution, resource, horizon=params.horizon)
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


def unitize_intervals(intervals):
    return [(t, t+1, c)
            for s, e, c in intervals
            for t in range(s, e, 1)]
