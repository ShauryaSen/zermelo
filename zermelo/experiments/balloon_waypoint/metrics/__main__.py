"""One sweep run drawn and tabulated: `python -m zermelo.experiments.balloon_waypoint.metrics <sweep name or its directory>`"""

import sys
from pathlib import Path

import numpy as np

from zermelo.experiments.balloon_waypoint.metrics import figures, summary
from zermelo.experiments.balloon_waypoint.metrics.data import (
    CURVES,
    DIAGNOSTICS,
    channels,
    finished_cells,
    labels,
    latest_launch,
    opening_moves,
    tables,
)
from zermelo.experiments.balloon_waypoint.render.style import Style
from zermelo.problems.balloon.world import load_wind

SWEEPS = Path("results") / __package__.split(".")[-2]
"""Where a sweep named rather than pointed at is looked for: this experiment's own results"""

asked = Path(sys.argv[1])
where = asked if asked.exists() else SWEEPS / asked
if not where.is_dir():
    sys.exit(f"no sweep or launch directory at {asked}, and none at {SWEEPS / asked}")
sweep = latest_launch(where)
cells = finished_cells(sweep)  # names and configurations only, the arrays read one cell at a time
curves, legs, spent = tables(cells)
style = Style()
names = labels(sorted(curves["arm"].unique()))
colours = {arm: style.arm_colours[slot % len(style.arm_colours)] for slot, arm in enumerate(sorted(names))}
title = sweep.parent.name.replace("_", " ")
# an arm with no rule has no planner, so the planner is read off one that does
described = next((config for *_, config in cells if config.get("method") is not None), cells[0][3])
method = described.get("method")
moves = int(curves["step"].max())
budget = moves if method is None else int(method["planner"]["max_steps"])
opening = opening_moves(described, moves)

figures.draw_curves(curves, CURVES, colours, names, opening, style, title, sweep / "curves.png")
figures.draw_curves(curves, DIAGNOSTICS, colours, names, opening, style, title, sweep / "diagnostics.png")
figures.draw_cost(spent, curves, colours, names, style, title, sweep / "cost.png")

record = load_wind(Path(described["problem"]["wind_path"]))
grid = record.grid

walked = 0
for arm, seed, cell, _ in cells:
    own = legs[(legs["arm"] == arm) & (legs["seed"] == seed)]
    if not own.empty:  # an arm running no rule aims at nothing and finishes no waypoint leg
        figures.draw_legs(own, budget, colours[arm], style, f"{names[arm]}, seed {seed}", cell / "legs.png")
        walked += 1
print(f"wrote curves.png, diagnostics.png and cost.png under {sweep}, and legs.png under {walked} cells")
states = record.n_alt * grid.n_lat * grid.n_lon
candidates = int(np.asarray(channels(cells[0][2], ("objective_state/truth",))["objective_state/truth"]).shape[-1])
print(f"wrote the tables under {summary.write(sweep, curves, legs, spent, described, names, states, candidates)}")
