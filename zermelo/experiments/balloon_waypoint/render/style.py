"""Every number a drawing is made with"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import cmocean
import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter


def truncated(name: str, cap: float) -> Colormap:
    """The colour map `name` restricted to [0, `cap`] and stretched back over [0, 1]"""
    return LinearSegmentedColormap.from_list(f"{name}_{cap}", mpl.colormaps[name](np.linspace(0.0, cap, 256)))


def text_width(figure: Figure, string: str, size: float) -> float:
    """Width in inches of `string` set at `size` points"""
    written = figure.text(0.0, 0.0, string, fontsize=size)
    width = written.get_window_extent().width / float(figure.dpi)
    written.remove()
    return width


def plain_numbers(panel: Axes) -> None:
    """A log axis's vertical ticks as plain numbers where its range spans less than a decade"""
    low, high = panel.get_ylim()
    if high < 10.0 * low:  # inside one decade every tick is a minor one, and the default writes those as a power
        panel.yaxis.set_major_formatter(ScalarFormatter())
        panel.yaxis.set_minor_formatter(ScalarFormatter())


@dataclass(frozen=True)
class Style:
    """What a drawing looks like"""

    # --- the sheet, in inches ------------------------------------------------------------------------------------
    # a panel is drawn at one cell to one cell: these are the panel heights, and a width is that height
    # times the grid's own aspect ratio
    world_height: float = 7.2
    """Height of the panel a whole move is drawn in"""

    tile_height: float = 2.9
    """Height of one panel on a sheet of many moves"""

    gap: float = 0.16
    margin: float = 0.24
    bar_thickness: float = 0.17
    bar_gap: float = 0.12
    bar_ticks: float = 0.62
    """Room beside a colour bar for the numbers against it"""

    bar_name: float = 0.48
    """Further room beside a colour bar that carries its own name"""

    bar_shrink: float = 0.55
    """Fraction of the block of panels one shared colour bar spans, centred on it"""
    tick_gap: float = 0.56
    """Left and bottom room for tick labels alone"""

    axis_gap: float = 0.98
    """Left and bottom room for tick labels with a named axis under them"""

    axis_name_pad: float = 8.0
    """Points between an axis's numbers and its name"""

    title_gap: float = 0.40
    """Room for a panel's own title"""

    state_gap: float = 0.26
    """Further room under a panel for the line of what its episode has spent"""

    header: float = 1.30
    """Room at the top for the figure title and the line of state under it"""

    strip_height: float = 2.0
    """Height of one panel of the strip of curves under a single episode"""

    strip_gap: float = 1.55
    """Room between two panels of that strip, for the right one's numbers and its name"""

    live_dot: float = 9.0
    """Size of the mark the strip puts on a curve at the move being drawn"""

    key_line: float = 1.7
    """Height of one row of the key, in multiples of the key's own type size"""

    dpi: int = 220
    """Dots per inch a sheet is drawn at when nothing names a resolution"""

    resolutions: Mapping[str, int] = field(default_factory=lambda: {"high": 200, "medium": 140, "low": 90})
    """Dots per inch each named resolution draws at"""

    paper: str = "#ffffff"
    ink: str = "#12161b"
    faint: str = "#5d666f"

    # --- type ----------------------------------------------------------------------------------------------------
    title_size: float = 32.0
    subtitle_size: float = 20.0
    label_size: float = 22.0
    panel_title_size: float = 21.0
    tick_size: float = 17.0
    key_size: float = 19.0
    key_head: float = 13.0
    """Point size of the arrowhead on a key mark"""
    key_handle: float = 2.2
    """Length of a key mark, in multiples of the key's own type size"""
    key_spacing: float = 1.6
    """Space between key columns, in multiples of the key's own type size"""

    # --- the rasters ---------------------------------------------------------------------------------------------
    field_colours: Colormap = field(default_factory=lambda: mpl.colormaps["RdBu_r"])
    """Signed quantities: the true field and the belief's mean of it"""

    uncertainty_colours: Colormap = field(default_factory=lambda: truncated("plasma", 0.85))
    error_colours: Colormap = field(default_factory=lambda: truncated("Reds", 0.9))
    acquisition_colours: Colormap = field(default_factory=lambda: mpl.colormaps["viridis"])

    speed_colours: Colormap = field(default_factory=lambda: truncated("YlGnBu", 0.92))
    """What a wind magnitude is shaded on: pale at calm through teal to deep blue at the fastest, so a dark
    arrow reads over the whole range"""

    arrow_span: float = 0.065
    """Length of an arrow at the field's fastest wind, as a fraction of the panel's width"""

    absent: str = "#f1f3f5"
    """What a cell with no value at all is drawn as"""

    pad_wash: str = "#20242b"
    pad_alpha: float = 0.11
    """How dark the ring outside the counted cells goes"""

    ticks_per_axis: int = 5
    """Upper bound on labelled ticks per axis, the locator picking round coordinates under it"""

    frame_colour: str = "#2f3640"
    frame_width: float = 1.1
    """Width of the outline drawn around the cells that are scored"""

    # --- the field, drawn as arrows ------------------------------------------------------------------------------
    arrows_per_axis: int = 17
    """Arrows across the counted cells, the grid strided to land near this many"""

    arrow_cells: float = 2.7
    """Length in cells of a field arrow at a saturating magnitude"""

    arrow_colour: str = "#101418"
    arrow_width: float = 0.0022
    arrow_head: float = 2.6
    """Head length of an arrow, in multiples of its own shaft width"""
    belief_colour: str = "#ffffff"
    belief_edge: float = 0.7
    """Width of the outline the believed field's white arrows carry"""

    # --- the plan, drawn as arrows -------------------------------------------------------------------------------
    plan_arrows_per_axis: int = 13
    plan_span: float = 0.75
    """Length of a policy arrow as a fraction of the gap to the next. Arrow length carries no data -- every cell is steered by one
    control step or none -- so this only sets spacing"""

    plan_colour: str = "#1f9e6e"
    plan_alpha: float = 0.6
    plan_width: float = 0.030
    """Shaft width of a policy arrow, in inches"""

    plan_head_width: float = 2.4
    plan_head_length: float = 2.4
    """Head of a policy arrow, in multiples of its own shaft width"""

    # --- the walk taken, and the walk imagined -------------------------------------------------------------------
    # a segment `a` of the way back through the walk so far is drawn at
    # alpha = floor + (1 - floor) * (1 - a) ** gamma, width = trail_width * (taper + (1 - taper) * (1 - a))
    trail_span: float = 0.12
    """How far back the walk stays visible, as a fraction of the whole episode"""

    trail_gamma: float = 2.6
    trail_floor: float = 0.0
    trail_taper: float = 0.30
    trail_width: float = 3.6
    trail_colour: str = "#ff4fa6"

    visited_dot: float = 5.0
    """Size of a mark on the strip of altitudes, one per cell stood on at that altitude"""

    star_size: float = 22.0
    """Size of the mark on the fastest point of the whole volume"""
    trail_halo: str = "#12161c"
    halo_alpha: float = 0.70
    """Opacity of the dark outline the walk is drawn on"""

    opening_colour: str = "#9fb0c2"
    """The stretch walked before any plan existed"""

    imagined_colour: str = "#19d3f3"
    imagined_width: float = 2.2

    # --- the actor -----------------------------------------------------------------------------------------------
    actor_size: float = 110.0
    actor_colour: str = "#111418"
    drift_colour: str = "#0b6fa4"
    control_colour: str = "#d94f04"
    glyph_width: float = 4.8
    glyph_head: float = 20.0
    drift_cells: float = 5.0
    """Length in cells of a drift-step arrow at a saturating magnitude"""

    control_cells: float = 3.5
    """Length in cells of the control-step arrow. The drift arrow beside it is drawn at true length"""

    waypoint_colour: str = "#e8112d"
    waypoint_size: float = 190.0
    """Size of the filled cross a waypoint is drawn as"""

    # --- which moves get drawn -----------------------------------------------------------------------------------
    stills: int = 12
    """Panels on a contact sheet when no explicit moves are asked for"""

    # --- the curves a sweep is read by ---------------------------------------------------------------------------
    arm_colours: tuple[str, ...] = ("#12161b", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#8a6d3b")
    """One colour per arm in the order the arms sort, cycled where a sweep runs more arms than colours"""

    curve_width: float = 2.2
    band_alpha: float = 0.18
    """Opacity of the band of an arm's spread over seeds, drawn behind its line"""

    curve_marks: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
    """One mark per arm in the order the arms sort, cycled where a sweep runs more arms than marks"""

    curve_mark_size: float = 7.0
    curve_marks_drawn: int = 8
    """Marks spaced along a line, so a curve of many moves carries the mark its key names"""

    panel_width: float = 6.4
    panel_height: float = 4.2
    """One curve panel, in inches"""

    curve_key_size: float = 13.0
    """Type size of the key under a curve panel"""
