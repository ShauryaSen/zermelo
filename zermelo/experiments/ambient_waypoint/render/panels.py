"""What one episode looks like on the grid, and the sheets its panels go onto"""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter, MaxNLocator

from zermelo.experiments.ambient_waypoint.metrics.data import CURVES, settings
from zermelo.experiments.ambient_waypoint.render.replay import Replay
from zermelo.experiments.ambient_waypoint.render.style import Style, plain_numbers, text_width

RASTER_NAMES = ("truth", "belief", "uncertainty", "error", "acquisition")
"""What a world panel may be shaded by, and what the panels beside it are chosen from"""

LIVE = ("simple_regret", "cumulative_regret", "reconstruction_error", "posterior_uncertainty")
"""What the strip under a single episode advances through, one panel each"""

TITLES = {
    "truth": "true field",
    "belief": "posterior mean",
    "uncertainty": "posterior standard deviation",
    "error": "absolute posterior error",
    "acquisition": "acquisition score",
}
"""What a raster is called in words"""

SYMBOLS = {
    "truth": r"$f^{\star}(z)$",
    "belief": r"$\mu_t(z)$",
    "uncertainty": r"$\sigma_t(z)$",
    "error": r"$|\mu_t(z) - f^{\star}(z)|$",
    "acquisition": r"$\alpha_t(z)$",
}
"""What a raster is called in symbols"""

ACROSS = "ambient coordinate $y$"
UP = "controllable coordinate $q$"


def _range(name: str) -> str:
    """The label a colour bar carries for `name`: its words and its symbol together"""
    return f"{TITLES[name]}   {SYMBOLS[name]}"


def _colours(name: str, style: Style) -> Colormap:
    """The colour map `name` is drawn with"""
    return {
        "truth": style.field_colours,
        "belief": style.field_colours,
        "uncertainty": style.uncertainty_colours,
        "error": style.error_colours,
        "acquisition": style.acquisition_colours,
    }[name]


def _scored(replay: Replay) -> tuple[float, float, float, float]:
    """The block of cells a candidate is ever drawn from, as (left, right, low, high) in coordinates"""
    rows, columns = np.nonzero(~replay.pad)
    across_lo, _, up_lo, _ = replay.extent
    return (
        across_lo + columns.min() * replay.spacing,
        across_lo + (columns.max() + 1) * replay.spacing,
        up_lo + rows.min() * replay.spacing,
        up_lo + (rows.max() + 1) * replay.spacing,
    )


def _centres(replay: Replay) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres up the controllable axis and across the ambient one, in coordinates"""
    across_lo, across_hi, up_lo, up_hi = replay.extent
    half = replay.spacing / 2
    return np.linspace(up_lo + half, up_hi - half, replay.shape[0]), np.linspace(across_lo + half, across_hi - half, replay.shape[1])


def _place(fig: Figure, left: float, bottom: float, width: float, height: float) -> Axes:
    """One axes at `(left, bottom)` of size `(width, height)`, every argument in inches from the sheet's bottom left"""
    sheet_width, sheet_height = fig.get_size_inches()
    return fig.add_axes((left / sheet_width, bottom / sheet_height, width / sheet_width, height / sheet_height))


def _bar(fig: Figure, image: AxesImage, rect: tuple[float, float, float, float], label: str | None, style: Style) -> None:
    """A vertical colour bar for `image` on its own axes at `rect`, in inches from the sheet's bottom left"""
    bar = fig.colorbar(image, cax=_place(fig, *rect))
    bar.ax.tick_params(labelsize=style.tick_size, colors=style.faint)
    bar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    low, high = image.get_clim()
    peak = max(abs(low), abs(high))
    power = int(np.floor(np.log10(peak))) if peak > 0.0 and not 1e-2 <= peak < 1e4 else 0
    if power:  # in the label rather than a floating offset, which collides with whatever sits above the bar
        bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 10.0**power:g}"))
        scale = rf"$\times 10^{{{power}}}$"  # a bare bar still says what its numbers are in, its panel's title cannot
        label = scale if label is None else f"{label}  {scale}"
    if label:
        bar.set_label(label, fontsize=style.label_size, color=style.ink, labelpad=6)
    for written in bar.ax.get_yticklabels():
        written.set_color(style.ink)
    bar.outline.set_edgecolor(style.faint)


def raster(
    ax: Axes,
    replay: Replay,
    name: str,
    move: int,
    style: Style,
    *,
    named_axes: bool = False,
    cropped: bool = False,
    limits: tuple[float, float] | None = None,
) -> AxesImage:
    """`name` at `move`, over the counted cells alone or over the whole grid with the ring outside them washed over"""
    frame = replay.raster(name, move)  # asked for first: an arm with no rule is refused by name, not by a range
    low, high = replay.limits(name, move) if limits is None else limits
    image = ax.imshow(
        frame,
        origin="lower",  # row 0 is the lowest controllable cell, and that axis runs up the page
        extent=replay.extent,
        cmap=_colours(name, style).with_extremes(bad=style.absent),
        vmin=low,
        vmax=high,
        interpolation="nearest",
        aspect="equal",
    )
    ax.imshow(  # one flat colour, laid only where nothing counts
        np.where(replay.pad, 1.0, np.nan),
        origin="lower",
        extent=replay.extent,
        cmap=ListedColormap([style.pad_wash]),
        vmin=0.0,
        vmax=1.0,
        alpha=style.pad_alpha,
        interpolation="nearest",
        aspect="equal",
    )
    block = _scored(replay)
    if not cropped:  # cropped, the axes edge is that outline already
        ax.add_patch(
            Rectangle(
                block[::2],
                block[1] - block[0],
                block[3] - block[2],
                fill=False,
                edgecolor=style.frame_colour,
                linewidth=style.frame_width,
                zorder=2.5,
            )
        )
    ax.set_xlim(*(block[:2] if cropped else replay.extent[:2]))
    ax.set_ylim(*(block[2:] if cropped else replay.extent[2:]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=style.ticks_per_axis, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=style.ticks_per_axis, integer=True))
    ax.tick_params(labelsize=style.tick_size, colors=style.faint, length=3.0, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(style.ink)
    if named_axes:
        ax.set_xlabel(ACROSS, fontsize=style.label_size, color=style.ink, labelpad=style.axis_name_pad)
        ax.set_ylabel(UP, fontsize=style.label_size, color=style.ink, labelpad=style.axis_name_pad)
    for edge in ax.spines.values():
        edge.set_color(style.faint)
    return image


def world(
    ax: Axes,
    replay: Replay,
    move: int,
    style: Style,
    *,
    background: str = "uncertainty",
    show_plan: bool = False,
    show_belief: bool = False,
    named_axes: bool = False,
    limits: tuple[float, float] | None = None,
) -> AxesImage:
    """One move drawn whole: the chosen background, the field as arrows, the walk taken, the walk imagined, and the actor"""
    image = raster(ax, replay, background, move, style, named_axes=named_axes, limits=limits)
    up, across = _centres(replay)
    stride = max(1, min(replay.shape) // style.arrows_per_axis)
    right, high = np.meshgrid(across[::stride], up[::stride])
    reach = min(style.arrow_cells, float(stride)) * replay.spacing  # what a saturated arrow takes, capped so two cannot collide
    laid = (
        [("belief", style.belief_colour, 1.9), ("truth", style.arrow_colour, 2.1)] if show_belief else [("truth", style.arrow_colour, 2.1)]
    )
    for name, colour, height in laid:
        # saturating: one gust cannot shrink a breeze to a dot, and both fields share the one saturation
        field = replay.raster(name, 0 if name == "truth" else move)[::stride, ::stride]  # (rows', columns'), NaN through the ring
        ax.quiver(
            right,
            high,
            reach * np.tanh(field / replay.amplitude),  # the field displaces along the ambient axis alone, so an arrow lies flat
            np.zeros_like(field),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=colour,
            edgecolor=style.arrow_colour,  # the believed field is white inside this outline; the true field is filled with it
            linewidth=style.belief_edge,
            width=style.arrow_width,
            zorder=height,
        )
    if show_plan and replay.plan is not None:
        step = max(1, min(replay.shape) // style.plan_arrows_per_axis)
        steer = np.full(replay.shape, np.nan)
        steer[replay.cell[:, 0], replay.cell[:, 1]] = replay.plan.steer[move]  # (rows, columns) what the plan steers everywhere
        # a fraction of the gap to the next arrow: arrows never chain into one line
        commanded = np.sign(steer[::step, ::step]) * style.plan_span * step * replay.spacing
        wide, tall = np.meshgrid(across[::step], up[::step])
        ax.quiver(
            wide,
            tall,
            np.zeros_like(commanded),  # steering moves along the controllable axis alone, so every arrow of the plan stands upright
            commanded,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=style.plan_colour,
            alpha=style.plan_alpha,
            units="inches",  # the shaft, and the head that is a multiple of it, rather than a fraction of the sheet
            width=style.plan_width,
            headwidth=style.plan_head_width,
            headlength=style.plan_head_length,
            headaxislength=style.plan_head_length,
            zorder=2.2,  # over the field
        )
    aim = replay.plan.waypoint[move] if replay.plan is not None else np.full(2, np.nan)
    if np.isfinite(aim).all():
        assert replay.plan is not None, "only a plan sets a waypoint"
        ax.plot(*replay.plan.walk[move].T, ls="--", lw=style.imagined_width, color=style.imagined_colour, zorder=3)
        ax.scatter(
            aim[0],
            aim[1],
            s=style.waypoint_size,
            marker="X",
            color=style.waypoint_colour,
            edgecolors=style.trail_halo,  # a thin dark outline, so it reads on the light end of every colour map here
            linewidths=1.0,
            zorder=5,
        )
    opening = replay.plan.opening_moves if replay.plan is not None else 0
    walked = replay.path[: move + 1]  # (move+1, 2) the whole walk so far, the old of it faded rather than cut
    if walked.shape[0] > 1:
        segments = np.stack([walked[:-1], walked[1:]], 1)  # (move, 2, 2) one segment per move walked
        back = np.arange(segments.shape[0] - 1, -1, -1)  # 0 at the newest segment, counting back through the walk
        age = np.minimum(1.0, back / max(1.0, style.trail_span * replay.n_moves))  # 1 once a segment is a whole span old
        alpha = (1.0 - age) ** style.trail_gamma * (1.0 - style.trail_floor) + style.trail_floor
        taper = style.trail_width * (style.trail_taper + (1.0 - style.trail_taper) * (1.0 - age))
        shade = np.where(np.arange(segments.shape[0]) + 1 <= opening, style.opening_colour, style.trail_colour)
        laid = list(segments)
        ax.add_collection(LineCollection(laid, colors=style.trail_halo, alpha=alpha * style.halo_alpha, linewidths=taper + 2.0, zorder=4))
        ax.add_collection(LineCollection(laid, colors=shade, alpha=alpha, linewidths=taper, zorder=4.1))
    here = replay.path[move]
    steered = replay.control[min(move, replay.n_moves - 1)]
    steps = (
        (replay.drift[move], style.drift_colour),  # true length: the magnitude is the whole of what it says
        (np.sign(steered) * style.control_cells * replay.spacing, style.control_colour),  # one step or none: only the sign is drawn
    )
    for vector, colour in steps:  # annotated rather than quivered: the head keeps its size however short the step is
        ax.annotate(
            "",
            xy=here + vector,
            xytext=here,
            zorder=7,  # above the actor's marker, which a one-cell step would otherwise sit inside
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=style.glyph_width, mutation_scale=style.glyph_head, shrinkA=0, shrinkB=0),
        )
    ax.scatter(here[0], here[1], s=style.actor_size, color=style.actor_colour, edgecolors="white", linewidths=1.4, zorder=6)
    return image


def _arrow(colour: str, label: str, style: Style, *, alpha: float = 1.0, edge: str | None = None) -> Line2D:
    """One key mark drawn as an arrow: a shaft in `colour` with a head at its end"""
    return Line2D(
        [], [], color=colour, lw=3.2, alpha=alpha, marker=">", ms=style.key_head, markevery=[-1], mec=edge or colour, mew=1.2, label=label
    )


def _walked(colour: str, label: str, style: Style) -> Line2D:
    """One key mark for a stretch of the walk: a line in `colour` on the same dark stroke the panel draws it on"""
    stroke = [withStroke(linewidth=style.trail_width + 2.0, foreground=style.trail_halo)]
    return Line2D([], [], color=colour, lw=style.trail_width, path_effects=stroke, label=label)


def keys(style: Style, *, plan: bool, show_belief: bool = False) -> list[Line2D | Patch]:
    """The key to a world panel, naming every mark the sheet carries"""
    marks: list[Line2D | Patch] = [_arrow(style.arrow_colour, r"drift field $f^{\star}(z)$", style)]
    if show_belief:
        marks.append(_arrow(style.belief_colour, r"believed field $\mu_t(z)$", style, edge=style.arrow_colour))
    marks += [
        Line2D([], [], color=style.actor_colour, marker="o", ls="", mec="white", mew=1.4, ms=11, label="actor $z_t$"),
        _arrow(style.drift_colour, "drift step", style),
        _arrow(style.control_colour, "control step", style),
        _walked(style.trail_colour, "trajectory $z_{0:t}$", style),
    ]
    if plan:  # an arm running no rule has no opening, no waypoint and no route
        marks += [
            _walked(style.opening_colour, "initial random rollout", style),
            Line2D([], [], color=style.imagined_colour, lw=2.6, ls="--", label="planned rollout to waypoint"),
            Line2D(
                [],
                [],
                color=style.waypoint_colour,
                marker="X",
                ls="",
                mec=style.trail_halo,
                mew=1.0,
                ms=13,
                label=r"waypoint $z^{\star}_t$",
            ),
            _arrow(style.plan_colour, r"policy $\pi_t$, per cell", style, alpha=0.8),
            Line2D([], [], color=style.faint, lw=1.2, ls=":", label="end of the shared opening"),
        ]
    marks.append(Patch(facecolor=style.pad_wash, alpha=style.pad_alpha, edgecolor=style.faint, lw=0.8, label="pad, no candidate scored"))
    return marks


def _plans(replay: Replay, move: int) -> int:
    """Plans in force by `move`, none where the arm runs no rule"""
    return 0 if replay.plan is None else int((replay.plan.replans <= move).sum())


def _spent(replay: Replay, move: int) -> str:
    """What one lane has spent by `move`: plans taken, and minutes inside the planner"""
    return f"replans: {_plans(replay, move)}      {float(replay.seconds[:move].sum()) / 60:.1f} min planning"


def caption(replay: Replay, move: int) -> str:
    """One line of state: which arm, how far in, how many plans, and what it has cost so far"""
    spent, held = float(replay.seconds[:move].sum()), float(replay.bytes_held[max(move - 1, 0)]) / 1e9
    return (
        f"{replay.name}      $t = {move}$ of ${replay.n_moves}$      replans: {_plans(replay, move)}      "
        f"planning time: {spent / 60:.1f} min      memory: {held:.2f} GB"
    )


def _written(fig: Figure, title: str, longest: str, style: Style) -> float:
    """Width in inches a sheet needs for `title` set above the line `longest`"""
    return 2 * style.margin + max(text_width(fig, title, style.title_size), text_width(fig, longest, style.subtitle_size))


def _even(inches: float, dpi: float) -> float:
    """`inches` rounded up to an even whole number of pixels at `dpi`"""
    return 2.0 * float(np.ceil(inches * dpi / 2.0)) / dpi


def _footer(fig: Figure, marks: list[Line2D | Patch], style: Style, sheet_width: float) -> tuple[int, float]:
    """How many columns the key takes on a sheet `sheet_width` wide, and the height in inches its rows then need"""
    widest = max(text_width(fig, str(mark.get_label()), style.key_size) for mark in marks)
    column = widest + (style.key_handle + style.key_spacing) * style.key_size / 72.0  # a mark, its name, and the space after it
    fits = max(1, int((sheet_width - 2 * style.margin) // column))
    across = int(np.ceil(len(marks) / np.ceil(len(marks) / fits)))  # evened out: the last row is never one lone mark
    return across, np.ceil(len(marks) / across) * style.key_line * style.key_size / 72.0


def _dress(fig: Figure, title: str, state: str, style: Style, marks: list[Line2D | Patch], across: int) -> None:
    """The title, the line of state under it, and the key, in the bands the layout reserved for them"""
    sheet_width, sheet_height = fig.get_size_inches()
    left = style.margin / sheet_width
    fig.text(left, 1.0 - (style.margin + 0.34) / sheet_height, title, fontsize=style.title_size, color=style.ink, va="baseline")
    fig.text(left, 1.0 - (style.margin + 0.72) / sheet_height, state, fontsize=style.subtitle_size, color=style.faint, va="baseline")
    fig.legend(
        handles=marks,
        loc="lower left",
        bbox_to_anchor=(left, style.margin / sheet_height),
        ncol=across,
        fontsize=style.key_size,
        labelcolor=style.ink,
        frameon=False,
        handlelength=style.key_handle,
        columnspacing=style.key_spacing,
    )


def progress(ax: Axes, replay: Replay, name: str, move: int, style: Style) -> None:
    """One curve of this episode against move, the whole of it faint and the part already run drawn over that"""
    curve = replay.curves[name]
    steps = np.arange(1, curve.size + 1)
    run = min(max(move, 1), curve.size)
    ax.plot(steps, curve, color=style.faint, lw=1.0, alpha=0.4, zorder=2)  # the whole curve: no frame rescales an axis
    ax.plot(steps[:run], curve[:run], color=style.ink, lw=style.curve_width, zorder=3)
    ax.plot(steps[run - 1], curve[run - 1], "o", ms=style.live_dot, color=style.ink, zorder=4)
    ax.set_yscale(CURVES[name][1])
    if CURVES[name][1] == "log":
        plain_numbers(ax)
    ax.set_xlim(0, curve.size)
    if replay.plan is not None:  # the opening, walked alike by every arm
        ax.axvline(replay.plan.opening_moves, ls=":", lw=1.2, color=style.faint, alpha=0.8, zorder=1)
    ax.set_xlabel("move $t$", fontsize=style.panel_title_size, color=style.ink, labelpad=style.axis_name_pad)
    ax.set_ylabel(CURVES[name][0], fontsize=style.panel_title_size, color=style.ink, labelpad=style.axis_name_pad)
    ax.tick_params(labelsize=style.tick_size, colors=style.faint, labelcolor=style.ink)
    ax.grid(True, color=style.faint, alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)
    for edge, shown in (("top", False), ("right", False), ("left", True), ("bottom", True)):
        ax.spines[edge].set_visible(shown)
        ax.spines[edge].set_color(style.faint)


def detail(
    fig: Figure,
    replay: Replay,
    move: int,
    style: Style,
    *,
    background: str = "uncertainty",
    show_plan: bool = False,
    show_belief: bool = False,
) -> None:
    """One move at full size: the world, every raster it is not already shaded by, and the curves so far under both"""
    aspect = (replay.extent[1] - replay.extent[0]) / (replay.extent[3] - replay.extent[2])
    block = _scored(replay)
    counted = (block[1] - block[0]) / (block[3] - block[2])  # a mini panel is cropped to the counted cells and has their aspect
    beside = [name for name in RASTER_NAMES if name != background and (name != "acquisition" or replay.plan is not None)][:4]
    named = style.bar_gap + style.bar_thickness + style.bar_ticks + style.bar_name
    # two rows of small panels stand beside the world and come to its height; only the bottom row of them is numbered
    tall = (style.world_height - style.gap - 2 * style.title_gap - style.tick_gap) / 2
    wide, world_wide = tall * counted, style.world_height * aspect
    left_column = style.gap + style.tick_gap + wide + named  # numbered up its own left edge
    right_column = style.gap + wide + named  # reading the numbers of the column beside it

    marks = keys(style, plan=replay.plan is not None, show_belief=show_belief)
    world_left = style.margin + style.axis_gap
    content = world_wide + named + left_column + right_column
    longest = caption(replay, replay.n_moves)  # the widest line this episode will ever carry: every frame is one width
    sheet_width = _even(max(world_left + content + style.margin, _written(fig, replay.label, longest, style)), float(fig.dpi))
    across, footer = _footer(fig, marks, style, sheet_width)
    strip = style.axis_gap + style.strip_height + style.gap  # the row of curves under everything, over its own axis names
    floor = style.margin + footer + strip + style.axis_gap
    fig.set_size_inches(sheet_width, _even(floor + style.world_height + style.header + style.margin, float(fig.dpi)))

    main = _place(fig, world_left, floor, world_wide, style.world_height)
    image = world(main, replay, move, style, background=background, show_plan=show_plan, show_belief=show_belief, named_axes=True)
    _bar(fig, image, (world_left + world_wide + style.bar_gap, floor, style.bar_thickness, style.world_height), _range(background), style)
    starts = (world_left + world_wide + named + style.gap + style.tick_gap, world_left + world_wide + named + left_column + style.gap)
    for slot, name in enumerate(beside):
        row, over = slot // 2, slot % 2
        short = 0.5 * right_column if len(beside) - 2 * row == 1 else 0.0  # a row holding one panel is centred, leaving no hole
        left = starts[over] + short
        bottom = floor + style.tick_gap + (1 - row) * (tall + style.title_gap + style.gap)
        ax = _place(fig, left, bottom, wide, tall)
        image = raster(ax, replay, name, move, style, cropped=True)
        _bar(fig, image, (left + wide + style.bar_gap, bottom, style.bar_thickness, tall), SYMBOLS[name], style)
        ax.tick_params(labelleft=over == 0, labelbottom=row == 1)  # one grid throughout: one column and one row carry the numbers
        ax.set_title(TITLES[name], fontsize=style.panel_title_size, color=style.ink, pad=5)
    span = sheet_width - world_left - style.margin
    each = (span - (len(LIVE) - 1) * style.strip_gap) / len(LIVE)  # every panel of the strip carries its own name up its left
    for slot, name in enumerate(LIVE):
        left = world_left + slot * (each + style.strip_gap)
        progress(_place(fig, left, style.margin + footer + style.axis_gap, each, style.strip_height), replay, name, move, style)
    _dress(fig, replay.label, caption(replay, move), style, marks, across)


def survey(replays: Sequence[Replay], move: int) -> str:
    """One line of state for a sheet of many episodes: how many lanes, what they hold fixed, and how far in"""
    rules = {replay.label for replay in replays}
    seeds = {settings(replay.name).get("seed", "?") for replay in replays}
    horizon = max(replay.n_moves for replay in replays)
    if len(rules) == 1:
        held = rules.pop()
    elif len(seeds) == 1:
        held = f"seed {seeds.pop()}"
    else:
        held = f"{len(rules)} rules over {len(seeds)} seeds"
    return f"{len(replays)} cells      {held}      $t = {min(move, horizon)}$ of ${horizon}$"


def _lane(replay: Replay, varies: Sequence[str]) -> str:
    """What tells one lane from the others: its rule where the arms differ, its seed where the seeds do, both where both"""
    told = {"arm": replay.label, "seed": f"seed {settings(replay.name).get('seed', '?')}"}
    return ", ".join(told[key] for key in varies) if varies else replay.label


def _sheet(
    fig: Figure,
    lanes: Sequence[tuple[Replay, int, str]],
    title: str,
    state: str,
    longest: str,
    style: Style,
    *,
    background: str = "uncertainty",
    show_plan: bool = False,
    show_belief: bool = False,
) -> None:
    """A near-square grid of world panels, one to a lane of `(episode, move, panel title)`, all on one colour range"""
    first = lanes[0][0]
    aspect = (first.extent[1] - first.extent[0]) / (first.extent[3] - first.extent[2])
    wide = style.tile_height * aspect
    columns = max(1, min(len(lanes), int(np.ceil(np.sqrt(len(lanes) / aspect)))))  # near square, once a panel's own aspect is in
    rows = int(np.ceil(len(lanes) / columns))
    across = wide + style.gap  # one column to the next, the numbers living only up the leftmost of them
    lane = style.title_gap + style.state_gap  # under every panel: its own name, and the line of what it spent
    up = style.tile_height + lane + style.gap  # one row to the next
    named = style.bar_gap + style.bar_thickness + style.bar_ticks + style.bar_name
    ranges = [replay.limits(background, move) for replay, move, _ in lanes]  # one range over every lane, which one bar then carries
    limits = (min(low for low, _ in ranges), max(high for _, high in ranges))

    marks = keys(style, plan=any(replay.plan is not None for replay, _, _ in lanes), show_belief=show_belief)
    left = style.margin + style.axis_gap
    room = left + (columns - 1) * across + wide + named + style.margin
    sheet_width = _even(max(room, _written(fig, title, longest, style)), float(fig.dpi))
    ncol, footer = _footer(fig, marks, style, sheet_width)
    floor = style.margin + footer + lane  # the bottom row carries its own two lines under it like every other row
    # the numbers run across the top, leaving the foot of every panel to its own cost
    sheet_height = _even(floor + (rows - 1) * up + style.tile_height + style.axis_gap + style.header + style.margin, float(fig.dpi))
    fig.set_size_inches(sheet_width, sheet_height)

    image = None
    for slot, (replay, move, name) in enumerate(lanes):
        row, over = slot // columns, slot % columns
        held = min(columns, len(lanes) - row * columns)  # panels this row holds, fewer than a column on the last row
        short = (columns - held) * across / 2  # a row holding fewer is centred, leaving no hole
        foot = floor + (rows - 1 - row) * up
        ax = _place(fig, left + short + over * across, foot, wide, style.tile_height)
        labelled = over == 0 and row == 0  # one panel carries the axis names, on the two edges already carrying the numbers
        image = world(
            ax, replay, move, style, background=background, show_plan=show_plan, show_belief=show_belief, named_axes=labelled, limits=limits
        )
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(labelleft=over == 0, labeltop=row == 0, labelcolor=style.ink)  # one column and one row carry the one grid
        centre = (left + short + over * across + wide / 2) / sheet_width
        for written, band, size, colour in (  # the name under the panel and the cost under that, each in a band of its own
            (name, style.title_gap, style.panel_title_size, style.ink),
            (_spent(replay, move), lane, style.tick_size, style.faint),
        ):
            place = (foot - band + 0.25 * size / 72.0) / sheet_height  # a baseline one descender up from the foot of its band
            fig.text(centre, place, written, fontsize=size, color=colour, ha="center", va="baseline")
    assert image is not None, "a sheet is drawn from at least one lane"
    spans = ((rows - 1) * up + style.tile_height) * style.bar_shrink  # one bar for every panel, centred on the block of them
    bottom = floor + ((rows - 1) * up + style.tile_height - spans) / 2
    _bar(fig, image, (left + (columns - 1) * across + wide + style.bar_gap, bottom, style.bar_thickness, spans), _range(background), style)
    _dress(fig, title, state, style, marks, ncol)


def contact(
    fig: Figure,
    replay: Replay,
    moves: np.ndarray,
    style: Style,
    *,
    background: str = "uncertainty",
    show_plan: bool = False,
    show_belief: bool = False,
) -> None:
    """The same world panel at each of `moves`"""
    lanes = [(replay, int(move), f"$t = {int(move)}$") for move in moves]
    _sheet(
        fig,
        lanes,
        replay.label,
        caption(replay, int(moves[-1])),
        caption(replay, replay.n_moves),
        style,
        background=background,
        show_plan=show_plan,
        show_belief=show_belief,
    )


def compare(
    fig: Figure,
    replays: Sequence[Replay],
    move: int,
    title: str,
    style: Style,
    *,
    background: str = "uncertainty",
    show_plan: bool = False,
    show_belief: bool = False,
) -> None:
    """One move of every episode side by side on one clock, a lane whose episode has ended holding its last frame"""
    if background == "acquisition" and len({replay.label for replay in replays}) > 1:
        raise ValueError("a score is in the units of the rule that computed it, so lanes running different rules share no range")
    varies = [key for key in ("arm", "seed") if len({settings(replay.name).get(key) for replay in replays}) > 1]
    lanes = [(replay, min(move, replay.n_moves), _lane(replay, varies)) for replay in replays]
    horizon = max(replay.n_moves for replay in replays)
    _sheet(
        fig,
        lanes,
        title,
        survey(replays, move),
        survey(replays, horizon),
        style,
        background=background,
        show_plan=show_plan,
        show_belief=show_belief,
    )
