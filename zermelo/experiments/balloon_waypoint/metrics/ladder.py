"""One line a noise level: the controller read across a sweep of forecast error

    pixi run python -m zermelo.experiments.balloon_waypoint.metrics.ladder mpc_noise_s0 mpc_noise_s12 ...
    pixi run python -m zermelo.experiments.balloon_waypoint.metrics.ladder --plot ladder.png winds_noise_s0 ...

A sweep with several arms gives a line per arm, so `winds_noise_s*` draws the score against noise once for
each `n_winds`. With `--plot` the same table is written as a figure.

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


def _measure(legs: pd.DataFrame, column: str) -> tuple[float, float]:
    """A column's mean over seeds and the standard error of that mean, nan where no seed offers one"""
    flown = legs[legs["flown"] > 0]
    # a leg that flew nowhere is not asked what it closed; the counts are over every leg
    if column == "flown":
        per_seed = flown.groupby("seed").size() if not flown.empty else pd.Series(dtype=float)
    else:
        per_seed = (legs if column in ("unreachable", "opening_gap") else flown).groupby("seed")[column].mean()
    if per_seed.empty:
        return float("nan"), float("nan")
    spread = float(per_seed.std(ddof=1) / np.sqrt(per_seed.size)) if per_seed.size > 1 else 0.0
    return float(per_seed.mean()), spread


def _rows(where: Path) -> list[dict[str, object]]:
    """One line per arm of one sweep: each measure as a mean over seeds, with its standard error"""
    cells = finished_cells(latest_launch(where))
    _, legs, _ = tables(cells)
    out = []
    for arm in sorted(legs["arm"].unique()):
        mine = legs[legs["arm"] == arm]
        row: dict[str, object] = {"sigma": _sigma(cells[0][3]), "arm": _named(arm), "seeds": int(mine["seed"].nunique())}
        for name, column, form in COLUMNS:
            mean, spread = _measure(mine, column)
            row[name] = "-" if np.isnan(mean) else f"{form.format(mean)} ± {form.format(spread).lstrip('+')}"
            row[f"_{column}"], row[f"_{column}_se"] = mean, spread  # kept numeric for the figure
        out.append(row)
    return out


def _named(arm: str) -> str:
    """An arm's own name, without the cell directory's `arm=` and the sweep's own prefix"""
    token = dict(pair.split("=", 1) for pair in arm.split(",") if "=" in pair).get("arm", arm)
    return token.rsplit("_", 1)[-1]


PLOTTED = (
    ("arrived", "share of legs that arrived"),
    ("closed", "share of the opening gap closed"),
)
"""The panels drawn against noise. Arrival first: it is bounded in [0, 1], where the gap closed has a heavy
left tail -- a leg that wanders off scores arbitrarily negative -- and measured standard errors on the
ladder were 0.04 to 0.06 for arrival against 0.07 to 0.21 for the gap closed"""


def _plot(table: pd.DataFrame, into: Path) -> None:
    """Each measure against noise, a line per arm, with the standard error of each mean as a bar"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from zermelo.experiments.balloon_waypoint.render.style import Style

    style = Style()
    figure, panels = plt.subplots(1, len(PLOTTED), figsize=(5.2 * len(PLOTTED), 4.0), dpi=style.dpi, squeeze=False)
    for panel, (column, label) in zip(panels[0], PLOTTED, strict=True):
        for slot, arm in enumerate(sorted(table["arm"].unique())):
            mine = table[table["arm"] == arm].sort_values("sigma")
            panel.errorbar(
                mine["sigma"],
                mine[f"_{column}"],
                yerr=mine[f"_{column}_se"],
                label=arm,
                color=style.arm_colours[slot % len(style.arm_colours)],
                marker="o",
                markersize=4,
                capsize=3,
                linewidth=1.4,
            )
        if column == "closed":  # closing nothing: ending as far off as it set out
            panel.axhline(0.0, color="#999999", linewidth=0.8, zorder=0)
        panel.set_xlabel("forecast error $\\sigma$ (m/s)")
        panel.set_ylabel(label)
        panel.spines[["top", "right"]].set_visible(False)
        if table["arm"].nunique() > 1:
            panel.legend(frameon=False, title="n_winds" if all(a.startswith("n") for a in table["arm"]) else None)
    figure.tight_layout()
    figure.savefig(into)
    print(f"wrote {into}")


def main(asked: list[str]) -> None:
    into = None
    if asked and asked[0] == "--plot":
        into, asked = Path(asked[1]), asked[2:]
    rows = []
    for name in asked:
        where = Path(name) if Path(name).exists() else SWEEPS / name
        if not where.is_dir():
            print(f"skipped {name}: no sweep there", file=sys.stderr)
            continue
        try:
            rows.extend(_rows(where))
        except ValueError as unfinished:  # no cell has written a record yet
            print(f"skipped {name}: {unfinished}", file=sys.stderr)
    if not rows:
        sys.exit("no finished sweep among those named")
    table = pd.DataFrame(sorted(rows, key=lambda row: (row["arm"], row["sigma"])))
    print(table[[name for name in table.columns if not name.startswith("_")]].to_string(index=False))
    if into is not None:
        _plot(table, into)


if __name__ == "__main__":
    main(sys.argv[1:])
