"""One line a noise level: the controller read across a sweep of forecast error

    pixi run python -m zermelo.experiments.balloon_waypoint.metrics.ladder mpc_noise_s0 mpc_noise_s12 ...

`draw` reports one sweep. This reads several and puts them beside each other, which is the whole result of the
noise experiment: not how the controller does at one error, but how what it achieves moves as the forecast
it is given decays. Sigma is taken from each sweep's own problem config rather than from its name.

The columns are the ones ``§8`` of `docs/noise-sweep.md` argues for, in that order of trust: the share of the
opening gap a leg closed, then arrival, then the diagnostics that say whether the legs were worth scoring.
Each is a mean over seeds with the standard error of that mean, since what is being asked of the table is
whether neighbouring levels differ.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from zermelo.experiments.balloon_waypoint.metrics.data import finished_cells, latest_launch, tables

SWEEPS = Path("results") / __package__.split(".")[-2]

COLUMNS = (
    ("gap closed", "closed", "{:+.3f}"),
    ("arrived", "arrived", "{:.3f}"),
    ("legs flown", "flown", "{:.1f}"),
    ("out of reach", "unreachable", "{:.3f}"),
    ("opening gap km", "opening_gap", "{:.0f}"),
)
"""What to print, the column of the per-leg table it comes from, and how to write it"""


def _sigma(config: dict) -> float:
    """The forecast error one sweep was run at (m/s)"""
    return float(config["problem"]["error_scale"])


def _row(where: Path) -> dict[str, object]:
    """One sweep's legs, reduced to a line: a mean over seeds and the standard error of that mean"""
    sweep = latest_launch(where)
    cells = finished_cells(sweep)
    _, legs, _ = tables(cells)
    flown = legs[legs["flown"] > 0]
    out: dict[str, object] = {"sigma": _sigma(cells[0][3]), "seeds": int(legs["seed"].nunique())}
    for name, column, form in COLUMNS:
        # a leg that flew nowhere is not asked what it closed; the counts are over every leg
        table = legs if column in ("unreachable", "opening_gap") else flown
        if column == "flown":
            per_seed = flown.groupby("seed").size() if not flown.empty else pd.Series(dtype=float)
        else:
            per_seed = table.groupby("seed")[column].mean()
        if per_seed.empty:
            out[name] = "-"
            continue
        spread = float(per_seed.std(ddof=1) / np.sqrt(per_seed.size)) if per_seed.size > 1 else 0.0
        out[name] = f"{form.format(float(per_seed.mean()))} ± {form.format(spread).lstrip('+')}"
    return out


def main(asked: list[str]) -> None:
    rows = []
    for name in asked:
        where = Path(name) if Path(name).exists() else SWEEPS / name
        if not where.is_dir():
            print(f"skipped {name}: no sweep there", file=sys.stderr)
            continue
        try:
            rows.append(_row(where))
        except ValueError as unfinished:  # no cell has written a record yet
            print(f"skipped {name}: {unfinished}", file=sys.stderr)
    if not rows:
        sys.exit("no finished sweep among those named")
    table = pd.DataFrame(sorted(rows, key=lambda row: row["sigma"]))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
