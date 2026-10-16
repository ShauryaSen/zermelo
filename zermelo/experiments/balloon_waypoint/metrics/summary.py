"""The tables one sweep run is reported by, as LaTeX fragments and the document tectonic compiles them into"""

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zermelo.experiments.balloon_waypoint.metrics.data import LEVEL_SET_THRESHOLD, LOG_VARIANCE_FLOOR, opening_moves

NOTATION = (
    (r"$Z$, $N$", r"states of the grid and its altitudes; $N = |Z|$"),
    (r"$X \subset Z$, $C$", r"the candidates a claim is made at; $C = |X|$"),
    (r"$W(x)$", r"the wind at $x$, as $(u, v)$ in m/s"),
    (r"$\|u\|$", r"$(u_1^2 + u_2^2)^{1/2}$"),
    (r"$T$, $t$, $S$", r"moves one episode makes, $t = 1 \ldots T$; $S$ seeds per arm"),
    (r"$T_0$", r"$L \times (\text{opening legs})$, walked at random before any waypoint exists"),
    (r"$z_t$, $p_t$", r"the cell stood on and the altitude flown at after move $t$"),
    (r"$g(x)$", r"the scalar the target reads off $W$ at $x$"),
    (r"$\mu_t(x)$, $v_t(x)$", rf"the mean and the log-variances claimed for $W(x)$ after move $t$; $v_t \geq {LOG_VARIANCE_FLOOR:g}$"),
    (r"$s_t(x)$", r"$\big(\sum_{i \leq 2} e^{v_t(x)_i}\big)^{1/2}$"),
    (r"$\ell(a)$", rf"$\mathbf{{1}}[a > {LEVEL_SET_THRESHOLD:g}]$, at a speed in m/s"),
    (r"$g^\star$", r"$\max_{x \in X} g(x)$"),
    (r"$b_t$", r"$\max_{u \leq t} \|W(z_u, p_u)\|$"),
    (r"$L$, $\rho$", r"the planner's truncation; how close to a waypoint counts as arrival"),
    (r"$J$, $j$", r"legs that finish after $T_0$, one stretch between waypoints taking force; $j = 1 \ldots J$"),
    (r"$k_j$, $x^\star_j$", r"the steps leg $j$ walked; the waypoint it aimed at"),
    (r"$H^\pi(z, x)$", r"steps from $z$ to $x$ under $\pi$, as the planner's own table, capped at $L$"),
    (r"$H_j$", r"$H^\pi(z, x^\star_j)$ read when leg $j$ chose its waypoint"),
    (r"$w_t(x)$", r"what $x$ was worth at move $t$, before travel was charged for reaching it"),
    (r"$\Delta_t$", r"seconds inside the method's own call at move $t$"),
    (r"$t_{\mathrm{draw}}$, $t_{\mathrm{last}}$", r"wall clock at the draw of the instance and at the last move"),
    (r"$\mathrm{RSS}_t$", r"resident bytes of the process at move $t$, which several episodes may share"),
)
"""Every symbol the tables below use, as the symbol against its definition"""

OUTCOME = (
    ("simple_regret", "simple regret", r"R_T = g^\star - b_T", 2, "min"),
    ("cumulative_regret", "cumulative regret", r"\sum_{t = 1}^{T} (g^\star - b_t)", 0, "min"),
    ("reconstruction_error", "reconstruction error", r"\sqrt{\frac{1}{C} \sum_{x \in X} \big(\|\mu_T(x)\| - g(x)\big)^2}", 3, "min"),
    ("posterior_uncertainty", "posterior uncertainty", r"\frac{1}{C} \sum_{x \in X} s_T(x)", 3, "min"),
    ("level_set_error", "level-set error", r"\frac{1}{C} \big|\{ x \in X : \ell(g(x)) \neq \ell(\|\mu_T(x)\|) \}\big|", 4, "min"),
)
"""What an arm achieved, as its column of the per-move table, its name, its formula, its digits, and which end is better"""

DIAGNOSTIC = (
    ("legs", "legs", r"J", 0, "none"),
    ("arrival_rate", "arrived", r"\frac{1}{J} \sum_{j \leq J} \mathbf{1}[k_j < L]", 3, "max"),
    ("walked", "steps walked", r"\frac{1}{J} \sum_{j \leq J} k_j", 1, "none"),
    ("predicted", "steps predicted", r"\frac{1}{J} \sum_{j \leq J} H_j", 1, "none"),
    ("altitude_changes", "altitude changes", r"\sum_{t = 1}^{T} |p_t - p_{t - 1}|", 0, "none"),
    ("fastest_wind_flown", "fastest wind flown", r"b_T", 1, "max"),
    ("zero_value_candidates", "worth nothing", r"\frac{1}{T - T_0} \sum_{t > T_0} \frac{|\{ x \in X : w_t(x) = 0 \}|}{C}", 3, "none"),
    ("reachable_candidates", "reachable", r"\frac{1}{T - T_0} \sum_{t > T_0} \frac{|\{ x \in X : H^\pi(z_t, x) < L \}|}{C}", 3, "none"),
    ("cells_visited", "cells visited", r"|\{ z_1, \ldots, z_T \}|", 0, "none"),
)
"""What an arm did to get there, in the same five parts"""

COST = (
    ("seconds_per_move", "seconds per move", r"\frac{1}{T} \sum_{t = 1}^{T} \Delta_t", 3, "min"),
    ("seconds", "seconds whole", r"t_{\mathrm{last}} - t_{\mathrm{draw}}", 0, "min"),
    ("peak_bytes", "peak GB", r"\max_{t \leq T} \mathrm{RSS}_t \, / \, 10^9", 2, "min"),
)
"""What an arm cost, in the same five parts"""

DOCUMENT = r"""\documentclass[11pt]{article}
\usepackage[a4paper,landscape,margin=1cm]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{float}
\floatstyle{plaintop}
\restylefloat{table}
\floatplacement{table}{H}
\pagestyle{empty}

\begin{document}
\begin{center}
{\large\bfseries %(sweep)s} \\[2pt]
launch %(launch)s: %(arms)d arms, %(seeds)d seeds, %(moves)d moves.
\end{center}

\section*{Notation}
\input{notation}

\medskip
\noindent At this launch: $N = %(cells)d$, $C = %(candidates)d$, $d = %(axes)d$, $T = %(moves)d$, $S = %(seeds)d$,
$L = %(budget)d$, $\rho = %(radius)g$, $T_0 = %(opening)d$.

\input{outcome}
\input{diagnostics}
\input{cost}
\end{document}
"""
"""The standalone document, taking one sweep run's own identity and the values of the symbols above"""


def _escaped(label: str) -> str:
    """`label` with every character LaTeX reads as syntax backslashed"""
    return "".join("\\" + character if character in "&%$#_{}" else character for character in label)


def _cell(values: pd.Series, digits: int, best: bool) -> str:
    """The mean over one arm's seeds, bold where best and `--` where there is none, over `+- deviation` on a second line"""
    mean, deviation = float(values.mean()), float(values.std())
    written = "--" if not np.isfinite(mean) else (rf"$\mathbf{{{mean:.{digits}f}}}$" if best else f"${mean:.{digits}f}$")
    under = rf"{{\scriptsize $\pm {deviation:.{digits}f}$}}" if np.isfinite(mean) and np.isfinite(deviation) else r"{\scriptsize\strut}"
    # [t] sets the first of the two lines on the outer row's baseline
    return r"\begin{tabular}[t]{@{}c@{}}" + written + r" \\ " + under + r"\end{tabular}"


def _table(
    title: str, caption: str, columns: Sequence[tuple[str, str, str, int, str]], per_cell: pd.DataFrame, names: dict[str, str]
) -> str:
    """One `table` fragment over `per_cell`, a row an arm and a column its heading above the formula it is formed by"""
    grouped = per_cell.groupby("arm")
    mean = grouped[[column for column, *_ in columns]].mean()  # (arms, columns), what a bold is decided on
    best: dict[str, str] = {}
    for column, _, _, _, direction in columns:
        finite = mean[column][np.isfinite(mean[column])]  # only a finite mean is compared
        best[column] = "" if direction == "none" or finite.empty else (finite.idxmin() if direction == "min" else finite.idxmax())
    rows = [
        " & ".join(
            [_escaped(names.get(str(arm), str(arm)))]
            + [_cell(held[column], digits, arm == best[column]) for column, _, _, digits, _ in columns]
        )
        for arm, held in grouped
    ]
    return "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            rf"\caption{{\textbf{{{title}.}} {caption}, as a mean over seeds above its standard deviation, best in bold.}}",
            rf"\label{{tab:{title.lower()}}}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{l" + "c" * len(columns) + "}",
            r"\toprule",
            " & ".join(["arm", *(heading for _, heading, _, _, _ in columns)]) + r" \\",
            " & ".join(["", *(rf"{{\scriptsize ${equation}$}}" for _, _, equation, _, _ in columns)]) + r" \\",
            r"\midrule",
            *(row + r" \\" for row in rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def write(
    launch: Path,
    curves: pd.DataFrame,
    legs: pd.DataFrame,
    spent: pd.DataFrame,
    config: dict[str, Any],
    names: dict[str, str],
    states: int,
    candidates: int,
) -> Path:
    """The LaTeX fragments, the document that inputs them and its PDF, written under `launch / "summary"` and given back as it

    `states` is every (lat, lon, altitude) the balloon may stand on, `candidates` the subset the objective scores.
    """
    method = config.get("method")
    moves = int(curves["step"].max())
    planner = {"max_steps": moves, "radius": float("nan")} if method is None else method["planner"]
    ended = curves.sort_values("step").groupby(["arm", "seed"]).last().reset_index()
    held = curves.groupby(["arm", "seed"])[["zero_value_candidates", "reachable_candidates"]].mean().reset_index()
    counted = (
        legs.groupby(["arm", "seed"]).agg(legs=("walked", "size"), walked=("walked", "mean"), predicted=("predicted", "mean")).reset_index()
    )
    counted["arrival_rate"] = legs.groupby(["arm", "seed"])["arrived"].mean().to_numpy()
    diagnostics = (
        ended[["arm", "seed", "cells_visited", "altitude_changes", "fastest_wind_flown"]]
        .merge(held, on=["arm", "seed"], how="left")
        .merge(counted, on=["arm", "seed"], how="left")
    )
    directory = launch / "summary"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "outcome.tex").write_text(_table("Outcome", r"What each arm had reached and claimed by move $T$", OUTCOME, ended, names))
    (directory / "diagnostics.tex").write_text(_table("Diagnostics", "What each arm did to get there", DIAGNOSTIC, diagnostics, names))
    (directory / "cost.tex").write_text(
        _table("Cost", "What each arm cost", COST, spent.assign(peak_bytes=spent["peak_bytes"] / 1e9), names)
    )
    (directory / "notation.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{@{}ll@{}}",
                r"\toprule",
                *(rf"{symbol} & {definition} \\" for symbol, definition in NOTATION),
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        )
    )
    (directory / "summary.tex").write_text(
        DOCUMENT
        % {
            "sweep": _escaped(launch.parent.name.replace("_", " ")),
            "launch": _escaped(launch.name),
            "arms": curves["arm"].nunique(),
            "seeds": curves["seed"].nunique(),
            "moves": moves,
            "cells": states,  # every (lat, lon, altitude) the balloon may stand on
            "candidates": candidates,  # the subset of those the objective scores
            "axes": 3,  # lat, lon and altitude
            "budget": int(planner["max_steps"]),
            "radius": float(planner["radius"]),
            "opening": opening_moves(config, moves),
        }
    )
    # on PATH, then beside the running interpreter, where pixi puts it
    binary = shutil.which("tectonic") or shutil.which("tectonic", path=str(Path(sys.executable).parent))
    if binary is None:
        print(f"no tectonic on PATH, so {directory / 'summary.pdf'} was not compiled")
        return directory
    compiled = subprocess.run(
        [binary, "-X", "compile", "--outdir", str(directory), "--keep-logs", str(directory / "summary.tex")],
        capture_output=True,
        text=True,
        check=False,
    )
    if compiled.returncode:
        print(f"tectonic exited {compiled.returncode}, so {directory / 'summary.pdf'} may be stale; see {directory / 'summary.log'}")
    return directory
