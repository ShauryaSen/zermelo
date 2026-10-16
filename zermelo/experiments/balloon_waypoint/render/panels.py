"""What one episode looks like over the grid, and the sheets its panels go onto"""

import functools
from collections.abc import Sequence
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter, MaxNLocator

from zermelo.experiments.balloon_waypoint.metrics.data import CURVES, settings
from zermelo.experiments.balloon_waypoint.render.replay import Replay
from zermelo.experiments.balloon_waypoint.render.style import Style, plain_numbers, text_width

RASTER_NAMES = ("truth", "belief", "uncertainty", "error", "acquisition")
"""What the map may be shaded by, and what the panels beside it are chosen from"""

LIVE = ("simple_regret", "cumulative_regret", "reconstruction_error", "posterior_uncertainty")
"""What the strip under a single episode advances through, one panel each"""

TITLES = {
    "truth": "true wind speed",
    "belief": "believed wind speed",
    "uncertainty": "posterior standard deviation",
    "error": "absolute posterior error",
    "acquisition": "acquisition score",
}
"""What a raster is called in words"""

SYMBOLS = {
    "truth": r"$\|f^\star(x)\|$",
    "belief": r"$\|\mu_t(x)\|$",
    "uncertainty": r"$s_t(x)$",
    "error": r"$|\, \|\mu_t(x)\| - \|f^\star(x)\| \,|$",
    "acquisition": r"$\alpha_t(x)$",
}
"""What a raster is called in symbols"""

ACROSS = "longitude"
UP = "latitude"


def _range(name: str) -> str:
    """The label a colour bar carries for `name`: its words and its symbol together"""
    return f"{TITLES[name]}  {SYMBOLS[name]}"


def _colours(name: str, style: Style) -> Colormap:
    """The colour map `name` is drawn with"""
    return {
        "truth": style.speed_colours,
        "belief": style.speed_colours,
        "uncertainty": style.uncertainty_colours,
        "error": style.error_colours,
        "acquisition": style.acquisition_colours,
    }[name]


def _projection(replay: Replay) -> ccrs.Projection:
    """Plate carree centred on the grid, so a box across the dateline (120-240E, say) draws whole rather than as two strips"""
    return ccrs.PlateCarree(central_longitude=(replay.extent[0] + replay.extent[1]) / 2)


def _frame(ax: Axes, replay: Replay) -> dict[str, Any]:
    """What a drawing in degrees needs on a map (a transform) and on a plain axes (nothing)"""
    return {"transform": ccrs.PlateCarree()} if isinstance(ax, GeoAxes) else {}


def _place(fig: Figure, left: float, bottom: float, width: float, height: float, **extra: Any) -> Axes:
    """One axes at `(left, bottom)` of size `(width, height)`, every argument in inches from the sheet's bottom left"""
    sheet_width, sheet_height = fig.get_size_inches()
    return fig.add_axes((left / sheet_width, bottom / sheet_height, width / sheet_width, height / sheet_height), **extra)


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
        scale = rf"$\times 10^{{{power}}}$"
        label = scale if label is None else f"{label}  {scale}"
    if label:
        bar.set_label(label, fontsize=style.label_size, color=style.ink, labelpad=6)
    for written in bar.ax.get_yticklabels():
        written.set_color(style.ink)
    bar.outline.set_edgecolor(style.faint)


def _geography(ax: GeoAxes, replay: Replay, style: Style) -> None:
    """The coastlines and the box a map is read against"""
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor=style.ink, alpha=0.55, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.35, edgecolor=style.faint, alpha=0.4, zorder=5)
    ax.set_extent(replay.extent, crs=ccrs.PlateCarree())


def raster(ax: Axes, replay: Replay, name: str, move: int, style: Style, *, cropped: bool = False) -> AxesImage:
    """`name` at `move`, over the grid, the margin outside the candidates washed over"""
    low, high = replay.limits(name, move)
    image = ax.imshow(
        replay.raster(name, move),
        origin="lower" if replay.extent[3] > replay.extent[2] else "upper",
        extent=replay.extent,
        cmap=_colours(name, style),
        vmin=low,
        vmax=high,
        interpolation="nearest",
        aspect="auto",  # the axes box is already cut to the grid's aspect: letterboxing again wastes it
        zorder=2,
        **_frame(ax, replay),
    )
    if not cropped:
        ax.imshow(
            np.where(replay.outside, 1.0, np.nan),
            origin=image.origin,
            extent=replay.extent,
            cmap=ListedColormap([style.pad_wash]),
            vmin=0.0,
            vmax=1.0,
            alpha=style.pad_alpha,
            interpolation="nearest",
            aspect="auto",
            zorder=3,
            **_frame(ax, replay),
        )
    if isinstance(ax, GeoAxes):
        ax.set_extent(replay.extent, crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(replay.extent[0], replay.extent[1])
        ax.set_ylim(min(replay.extent[2], replay.extent[3]), max(replay.extent[2], replay.extent[3]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=style.ticks_per_axis))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=style.ticks_per_axis))
    ax.tick_params(labelsize=style.tick_size, colors=style.faint, labelcolor=style.ink)
    for edge in ax.spines.values():
        edge.set_color(style.faint)
    return image


def ladder(ax: Axes, replay: Replay, move: int, style: Style) -> None:
    """The altitudes as rungs, the one being flown (or nearest) filled, each named in km"""
    here = int(round(float(replay.flown[min(move, replay.flown.size - 1)])))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, len(replay.altitude_km) - 0.4)
    for level, km in enumerate(replay.altitude_km):
        filled = level == here
        ax.add_patch(
            Rectangle(
                (0.04, level - 0.38),
                0.92,
                0.76,
                facecolor=style.actor_colour if filled else style.paper,
                edgecolor=style.ink if filled else style.faint,
                lw=2.2 if filled else 1.0,
                zorder=3,
            )
        )
        ax.text(
            0.5,
            level,
            f"{km:.1f}",
            ha="center",
            va="center",
            fontsize=style.tick_size,
            color=style.paper if filled else style.faint,
            zorder=4,
        )
    ax.set_title("km", fontsize=style.panel_title_size, color=style.ink, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for edge in ax.spines.values():
        edge.set_visible(False)


def levels(axes: Sequence[Axes], replay: Replay, move: int, style: Style) -> None:
    """Every altitude's own speed field across, the cells stood on it marked, the one being flown (or nearest) outlined"""
    here = int(round(float(replay.flown[min(move, replay.flown.size - 1)])))
    span = float(np.max(replay.speed))
    fastest = np.unravel_index(int(np.argmax(replay.speed)), replay.speed.shape)  # the one fastest point of the volume
    lon = np.linspace(replay.extent[0], replay.extent[1], replay.shape[1])
    lat = np.linspace(replay.extent[2], replay.extent[3], replay.shape[0])
    walked, flown = replay.path[: move + 1], replay.flown[: move + 1]
    for level, ax in enumerate(axes):
        ax.imshow(
            replay.speed[level],
            origin="lower" if replay.extent[3] > replay.extent[2] else "upper",
            extent=replay.extent,
            cmap=style.speed_colours,
            vmin=0.0,
            vmax=span,
            interpolation="nearest",
            aspect="auto",
            zorder=2,
        )
        stood = walked[np.round(flown) == level]
        if stood.size:
            ax.plot(stood[:, 1], stood[:, 0], "o", ms=style.visited_dot, mew=0.0, alpha=0.75, color=style.trail_colour, zorder=3)
        if level == fastest[0]:
            ax.plot(lon[fastest[2]], lat[fastest[1]], "*", ms=style.star_size, mew=1.0, color=style.paper, mec=style.ink, zorder=4)
        ax.set_xlim(replay.extent[0], replay.extent[1])
        ax.set_ylim(min(replay.extent[2], replay.extent[3]), max(replay.extent[2], replay.extent[3]))
        ax.set_title(
            f"{replay.altitude_km[level]:.1f} km   {len(stood)} moves",
            fontsize=style.panel_title_size,
            color=style.ink if level == here else style.faint,
            pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for edge in ax.spines.values():
            edge.set_color(style.ink if level == here else style.faint)
            edge.set_linewidth(2.6 if level == here else 0.8)


def track(ax: Axes, replay: Replay, move: int, style: Style) -> None:
    """The altitude flown against move, up to `move`"""
    steps = np.arange(replay.flown.size)
    run = min(max(move, 1), replay.flown.size)
    smooth = bool(np.any(replay.flown != np.round(replay.flown)))  # a continuous balloon draws a line, a grid balloon stairs
    line = ax.plot if smooth else functools.partial(ax.step, where="post")
    line(steps, replay.flown, color=style.faint, lw=1.0, alpha=0.4, zorder=2)
    line(steps[:run], replay.flown[:run], color=style.ink, lw=style.curve_width, zorder=3)
    ax.plot(steps[run - 1], replay.flown[run - 1], "o", ms=style.live_dot, color=style.actor_colour, zorder=4)
    ax.set_yticks(range(len(replay.altitude_km)))
    ax.set_yticklabels([f"{km:.1f}" for km in replay.altitude_km])
    ax.set_ylim(-0.5, len(replay.altitude_km) - 0.5)
    ax.set_xlim(0, replay.flown.size)
    if replay.plan is not None:
        ax.axvline(replay.plan.opening_moves, ls=":", lw=1.2, color=style.faint, alpha=0.8, zorder=1)
    ax.set_xlabel("move $t$", fontsize=style.panel_title_size, color=style.ink, labelpad=style.axis_name_pad)
    ax.set_ylabel("altitude, km", fontsize=style.panel_title_size, color=style.ink, labelpad=style.axis_name_pad)
    ax.tick_params(labelsize=style.tick_size, colors=style.faint, labelcolor=style.ink)
    ax.grid(True, color=style.faint, alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)
    for edge, shown in (("top", False), ("right", False), ("left", True), ("bottom", True)):
        ax.spines[edge].set_visible(shown)
        ax.spines[edge].set_color(style.faint)


def world(
    ax: GeoAxes,
    replay: Replay,
    move: int,
    style: Style,
    *,
    background: str = "uncertainty",
    show_plan: bool = False,
    named_axes: bool = False,
) -> AxesImage:
    """One move drawn whole: the chosen shading, the wind at the altitude being flown, the track, and the balloon"""
    image = raster(ax, replay, background, move, style)
    _geography(ax, replay, style)
    here = int(replay.flown[min(move, replay.flown.size - 1)])
    lon = np.linspace(replay.extent[0], replay.extent[1], replay.shape[1])
    lat = np.linspace(replay.extent[2], replay.extent[3], replay.shape[0])
    # one arrow per cell; length carries speed, the fastest wind in the volume drawing `arrow_span` of the panel wide
    ax.quiver(
        lon,
        lat,
        replay.wind[here, :, :, 0],
        replay.wind[here, :, :, 1],
        color=style.arrow_colour,
        width=style.arrow_width,
        headwidth=style.arrow_head,
        headlength=style.arrow_head,
        headaxislength=style.arrow_head * 0.85,
        scale=float(np.max(replay.speed)) / style.arrow_span,
        scale_units="width",
        zorder=6,
        transform=ccrs.PlateCarree(),
    )
    fastest = np.unravel_index(int(np.argmax(replay.speed)), replay.speed.shape)  # the one fastest point of the volume
    if fastest[0] == here:
        ax.plot(
            lon[fastest[2]],
            lat[fastest[1]],
            "*",
            ms=style.star_size,
            mew=1.0,
            color=style.paper,
            mec=style.ink,
            zorder=11,
            transform=ccrs.PlateCarree(),
        )
    walked = replay.path[: move + 1]
    stroke = [withStroke(linewidth=style.trail_width + 2.0, foreground=style.trail_halo)]
    if walked.shape[0] > 1:
        ax.plot(
            walked[:, 1],
            walked[:, 0],
            color=style.trail_colour,
            lw=style.trail_width,
            path_effects=stroke,
            zorder=7,
            transform=ccrs.PlateCarree(),
        )
    if show_plan and replay.plan is not None:
        aimed, route = replay.plan.waypoint[move], replay.plan.walk[move]
        if np.isfinite(route).all():
            ax.plot(route[:, 1], route[:, 0], ls="--", lw=2.2, color=style.imagined_colour, zorder=8, transform=ccrs.PlateCarree())
        if np.isfinite(aimed).all():
            ax.plot(
                aimed[1],
                aimed[0],
                marker="X",
                ms=13,
                ls="",
                color=style.waypoint_colour,
                mec=style.trail_halo,
                mew=1.0,
                zorder=9,
                transform=ccrs.PlateCarree(),
            )
    ax.plot(
        walked[-1, 1], walked[-1, 0], "o", ms=11, color=style.actor_colour, mec="white", mew=1.4, zorder=10, transform=ccrs.PlateCarree()
    )
    if named_axes:
        ax.set_xlabel(ACROSS, fontsize=style.label_size, color=style.ink, labelpad=style.axis_name_pad)
        ax.set_ylabel(UP, fontsize=style.label_size, color=style.ink, labelpad=style.axis_name_pad)
    return image


def _walked(colour: str, label: str, style: Style) -> Line2D:
    """One key mark for a stretch of the track: a line in `colour` on the same dark stroke the panel draws it on"""
    stroke = [withStroke(linewidth=style.trail_width + 2.0, foreground=style.trail_halo)]
    return Line2D([], [], color=colour, lw=style.trail_width, path_effects=stroke, label=label)


def keys(style: Style, *, plan: bool) -> list[Line2D | Patch]:
    """The key to a map panel, naming every mark the sheet carries"""
    marks: list[Line2D | Patch] = [
        Line2D(
            [],
            [],
            color=style.arrow_colour,
            lw=3.2,
            marker=">",
            ms=style.key_head,
            markevery=[-1],
            path_effects=[withStroke(linewidth=5.0, foreground=style.trail_halo)],
            label="wind",
        ),
        Line2D([], [], color=style.actor_colour, marker="o", ls="", mec="white", mew=1.4, ms=11, label="balloon"),
        _walked(style.trail_colour, "trajectory", style),
        Line2D(
            [],
            [],
            color=style.paper,
            marker="*",
            ls="",
            mec=style.ink,
            mew=1.0,
            ms=style.star_size * 0.8,
            label="fastest wind in the volume",
        ),
    ]
    if plan:  # an arm running no rule has no opening, no waypoint and no route
        marks += [
            Line2D([], [], color=style.imagined_colour, lw=2.6, ls="--", label="planned route to waypoint"),
            Line2D([], [], color=style.waypoint_colour, marker="X", ls="", mec=style.trail_halo, mew=1.0, ms=13, label="waypoint"),
            Line2D([], [], color=style.faint, lw=1.2, ls=":", label="end of the shared opening"),
        ]
    marks.append(Patch(facecolor=style.pad_wash, alpha=style.pad_alpha, edgecolor=style.faint, lw=0.8, label="margin, no candidate scored"))
    return marks


def _plans(replay: Replay, move: int) -> int:
    """Plans in force by `move`, none where the arm runs no rule"""
    return 0 if replay.plan is None else int((replay.plan.replans <= move).sum())


def caption(replay: Replay, move: int) -> str:
    """One line of state: which arm, how far in, how high, what it has left, and what it has cost"""
    spent = float(replay.seconds[:move].sum())
    here = int(replay.flown[min(move, replay.flown.size - 1)])
    left = int(replay.balloon_resource[min(move, replay.balloon_resource.size - 1)])
    return (
        f"{replay.name}      $t = {move}$ of ${replay.n_moves}$      "
        f"{replay.altitude_km[here]:.1f} km      resource: {left}      "
        f"replans: {_plans(replay, move)}      planning time: {spent / 60:.1f} min"
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
    column = widest + (style.key_handle + style.key_spacing) * style.key_size / 72.0
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


def detail(fig: Figure, replay: Replay, move: int, style: Style, *, background: str = "uncertainty", show_plan: bool = False) -> None:
    """One move at full size: the map, the ladder beside it, every raster it is not shaded by, the levels, and the curves so far"""
    beside = [name for name in RASTER_NAMES if name != background and (name != "acquisition" or replay.plan is not None)][:4]
    named = style.bar_gap + style.bar_thickness + style.bar_ticks + style.bar_name
    map_wide = style.world_height / replay.aspect * (replay.extent[1] - replay.extent[0]) / (replay.extent[3] - replay.extent[2])
    rung = 0.72  # the ladder is a narrow column of rungs beside the map
    tall = (style.world_height - style.gap - 2 * style.title_gap - style.tick_gap) / 2
    wide = tall * (replay.extent[1] - replay.extent[0]) / (replay.extent[3] - replay.extent[2]) / replay.aspect
    column = style.gap + wide + named

    marks = keys(style, plan=replay.plan is not None)
    left = style.margin + style.axis_gap
    content = map_wide + named + 5 * style.gap + rung + style.tick_gap + 2 * column
    longest = caption(replay, replay.n_moves)  # the widest line this episode will ever carry: every frame is one width
    sheet_width = _even(max(left + content + style.margin, _written(fig, replay.label, longest, style)), float(fig.dpi))
    across, footer = _footer(fig, marks, style, sheet_width)
    strip = style.axis_gap + style.strip_height + style.gap
    levels_tall = style.world_height * 0.34
    floor = style.margin + footer + strip + style.axis_gap + levels_tall + 3 * style.gap
    fig.set_size_inches(sheet_width, _even(floor + style.world_height + style.header + style.margin, float(fig.dpi)))

    main = _place(fig, left, floor, map_wide, style.world_height, projection=_projection(replay))
    image = world(main, replay, move, style, background=background, show_plan=show_plan, named_axes=True)
    _bar(fig, image, (left + map_wide + style.bar_gap, floor, style.bar_thickness, style.world_height), _range(background), style)
    ladder(_place(fig, left + map_wide + named + 3 * style.gap, floor, rung, style.world_height), replay, move, style)

    block = left + map_wide + named + rung + 5 * style.gap + style.tick_gap  # room for the left panel's own numbers
    starts = (block, block + column)
    for slot, name in enumerate(beside):
        row, over = slot // 2, slot % 2
        panel = _place(fig, starts[over], floor + style.tick_gap + (1 - row) * (tall + style.title_gap + style.gap), wide, tall)
        small = raster(panel, replay, name, move, style, cropped=True)
        _bar(
            fig,
            small,
            (starts[over] + wide + style.bar_gap, panel.get_position().y0 * fig.get_size_inches()[1], style.bar_thickness, tall),
            SYMBOLS[name],
            style,
        )
        panel.tick_params(labelleft=over == 0, labelbottom=row == 1)
        panel.set_title(TITLES[name], fontsize=style.panel_title_size, color=style.ink, pad=5)

    span = sheet_width - left - style.margin
    each_level = (span - (len(replay.altitude_km) - 1) * style.strip_gap) / len(replay.altitude_km)
    bottom = style.margin + footer + strip + style.axis_gap
    levels(
        [
            _place(fig, left + slot * (each_level + style.strip_gap), bottom, each_level, levels_tall)
            for slot in range(len(replay.altitude_km))
        ],
        replay,
        move,
        style,
    )
    lanes = (*LIVE, "altitude")
    each = (span - (len(lanes) - 1) * style.strip_gap) / len(lanes)
    for slot, name in enumerate(lanes):
        panel = _place(fig, left + slot * (each + style.strip_gap), style.margin + footer + style.axis_gap, each, style.strip_height)
        if name == "altitude":
            track(panel, replay, move, style)
        else:
            progress(panel, replay, name, move, style)
    _dress(fig, replay.label, caption(replay, move), style, marks, across)


def survey(replays: Sequence[Replay], move: int) -> str:
    """One line of state for a sheet of many episodes: how many lanes, what they hold fixed, and how far in"""
    rules = {replay.label for replay in replays}
    seeds = {settings(replay.name).get("seed") for replay in replays}
    held = f"{len(rules)} rule{'s' if len(rules) != 1 else ''}" if len(rules) > 1 else next(iter(rules))
    return f"{len(replays)} episodes      {held}      {len(seeds)} seed{'s' if len(seeds) != 1 else ''}      $t = {move}$"


def _lane(replay: Replay, varies: Sequence[str]) -> str:
    """What tells one lane from the others: its rule where the arms differ, its seed where the seeds do, both where both"""
    own = settings(replay.name)
    return "      ".join([replay.label] if "arm" in varies else []) + ("" if "seed" not in varies else f"  seed {own.get('seed')}")


def compare(fig: Figure, replays: Sequence[Replay], move: int, title: str, style: Style, *, background: str = "uncertainty") -> None:
    """One move of every episode side by side on one clock, a lane whose episode has ended holding its last frame"""
    if background == "acquisition" and len({replay.label for replay in replays}) > 1:
        raise ValueError("a score is in the units of the rule that computed it, so lanes running different rules share no range")
    varies = [key for key in ("arm", "seed") if len({settings(replay.name).get(key) for replay in replays}) > 1]
    across = int(np.ceil(np.sqrt(len(replays))))
    down = int(np.ceil(len(replays) / across))
    marks = keys(style, plan=any(replay.plan is not None for replay in replays))
    one = replays[0]
    tall = style.world_height * 0.62
    wide = tall / one.aspect * (one.extent[1] - one.extent[0]) / (one.extent[3] - one.extent[2])
    sheet_width = _even(2 * style.margin + across * (wide + style.gap), float(fig.dpi))
    columns, footer = _footer(fig, marks, style, sheet_width)
    fig.set_size_inches(
        sheet_width,
        _even(style.margin + footer + down * (tall + style.title_gap + style.gap) + style.header + style.margin, float(fig.dpi)),
    )
    for slot, replay in enumerate(replays):
        row, over = slot // across, slot % across
        panel = _place(
            fig,
            style.margin + over * (wide + style.gap),
            style.margin + footer + (down - 1 - row) * (tall + style.title_gap + style.gap),
            wide,
            tall,
            projection=_projection(replay),
        )
        world(panel, replay, min(move, replay.n_moves), style, background=background, show_plan=True)
        panel.set_title(_lane(replay, varies), fontsize=style.panel_title_size, color=style.ink, pad=4)
    _dress(fig, title, survey(replays, move), style, marks, columns)


def contact(fig: Figure, replay: Replay, moves: np.ndarray, style: Style, *, background: str = "uncertainty") -> None:
    """The same map panel at each of `moves`"""
    marks = keys(style, plan=replay.plan is not None)
    across = len(moves)
    tall = style.world_height * 0.62
    wide = tall / replay.aspect * (replay.extent[1] - replay.extent[0]) / (replay.extent[3] - replay.extent[2])
    sheet_width = _even(2 * style.margin + across * (wide + style.gap), float(fig.dpi))
    columns, footer = _footer(fig, marks, style, sheet_width)
    fig.set_size_inches(
        sheet_width, _even(style.margin + footer + tall + style.title_gap + style.gap + style.header + style.margin, float(fig.dpi))
    )
    for slot, move in enumerate(moves):
        panel = _place(fig, style.margin + slot * (wide + style.gap), style.margin + footer, wide, tall, projection=_projection(replay))
        world(panel, replay, int(move), style, background=background, show_plan=True)
        panel.set_title(f"$t = {int(move)}$", fontsize=style.panel_title_size, color=style.ink, pad=4)
    _dress(fig, replay.label, caption(replay, int(moves[-1])), style, marks, columns)
