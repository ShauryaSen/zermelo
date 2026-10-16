"""Every number one sweep run is read by, off its records

W(lat, lon, p) = (u, v)     the wind (m/s)
g                           the scalar the target reads off W at one candidate
claim                       (mu_u, mu_v, log var_u, log var_v) at every candidate
"""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG_VARIANCE_FLOOR = -12.0
"""Smallest log-variance a claim is read at"""

LEVEL_SET_THRESHOLD = 30.0
"""Wind speed (m/s) the level-set error is measured at"""

RULES = {
    "random": "uniform random actions",
    "random_act": "uniform random actions",
    "random_search": "uniform random waypoint",
    "mean_mpc": "MPC on the posterior mean",
    "sampled_mpc": "MPC averaged over 8 belief draws",
    "max_variance": "maximum posterior variance",
    "posterior_spread": "maximum posterior variance",
    "max_magnitude": "largest believed speed",
    "ucb": "upper confidence bound",
    "mc_ei": "expected improvement, Monte Carlo",
    "ei": "expected improvement, closed form",
    "thompson": "Thompson sampling",
    "eui": "expected utility of information",
    "predictive_confidence": "predictive confidence",
    "mean": "posterior-mean plan",
    "sampled": "sampled-field plan",
}
"""What to call the rule an arm runs, keyed by a token of the arm's own name"""

CURVES = {
    "simple_regret": ("simple regret", "linear"),
    "cumulative_regret": ("cumulative regret", "linear"),
    "reconstruction_error": ("reconstruction error (m/s)", "linear"),
    "posterior_uncertainty": ("posterior uncertainty (m/s)", "linear"),
    "level_set_error": (f"level-set error (> {LEVEL_SET_THRESHOLD:g} m/s)", "linear"),
}
"""What an arm achieved, a value per move, as the words naming it and the scale it is drawn on"""

DIAGNOSTICS = {
    "arrival_rate": ("share of legs that arrived", "linear"),
    "cells_visited": ("distinct cells visited", "linear"),
    "altitude_changes": ("altitude changes made", "linear"),
    "fastest_wind_flown": ("fastest wind flown in (m/s)", "linear"),
    "zero_value_candidates": ("candidates worth nothing", "linear"),
    "reachable_candidates": ("candidates reachable", "linear"),
}
"""What an arm did to get there, in the same two parts"""

CURVE_CHANNELS = ("objective_state/oracle", "objective_state/incumbent", "objective_state/truth", "objective_state/claim")
"""What the curves are computed from, as the names a record holds them under"""

TABLE_CHANNELS = CURVE_CHANNELS + (
    "state/position",
    "state/altitude",
    "agent_state/steps_since_waypoint",
    "agent_state/scores.predicted_step_cost",
    "agent_state/scores.frac_zero_value_candidates",
    "agent_state/scores.frac_reachable_candidates",
    "reward",
    "time_per_decision",
    "time_per_episode",
    "peak_rss_per_process",
)


def channels(cell: Path, wanted: Sequence[str]) -> dict[str, Any]:
    """The arrays `wanted` names from one cell's record (a name the record does not hold is absent)"""
    with np.load(cell / "record.npz") as npz:
        return {name: npz[name] for name in wanted if name in npz}


def settings(name: str) -> dict[str, str]:
    """The pairs a cell directory's name carries, as `key=value` joined by commas"""
    return dict(pair.split("=", 1) for pair in name.split(",") if "=" in pair)


def rule(arm: str) -> str:
    """The rule the arm named `arm` runs, or its own name with the underscores opened out"""
    for key in sorted(RULES, key=len, reverse=True):
        if f"_{key}" in f"_{arm}":
            return RULES[key]
    return arm.replace("_", " ")


def labels(arms: Sequence[str]) -> dict[str, str]:
    """What to call each of `arms`: the rule it runs, or its own name where two arms run the one rule"""
    own = {arm: settings(arm).get("arm", arm) for arm in arms}
    named = {arm: rule(token) for arm, token in own.items()}
    shared = Counter(named.values())
    return {arm: words if shared[words] == 1 else own[arm].replace("_", " ") for arm, words in named.items()}


def claimed_speed(claim: Any) -> tuple[Any, Any]:
    """The speed a claim predicts and the spread it holds: `||mu||` and `(sum_i exp(v_i))^(1/2)`"""
    mean, log_variance = claim[..., 0:2], np.maximum(claim[..., 2:4], LOG_VARIANCE_FLOOR)
    return np.linalg.norm(mean, axis=-1), np.sqrt(np.sum(np.exp(log_variance), axis=-1))


def at_each_snapshot(claim: Any, snapshots: int, claim_every: int) -> Any:
    """A claim stored every `claim_every` moves, spread back over `snapshots` by carrying each reading forward"""
    return claim[np.minimum(np.arange(snapshots) // claim_every, claim.shape[0] - 1)]


def episode_curves(held: dict[str, Any], claim_every: int) -> dict[str, Any]:
    """What one episode achieved, a value per move"""
    oracle = held["objective_state/oracle"][1:]  # (moves,) constant: the best g on offer
    incumbent = held["objective_state/incumbent"][1:]  # (moves,) the fastest wind stood in
    truth = held["objective_state/truth"][1:]  # (moves, candidates) g at each candidate
    claim = at_each_snapshot(held["objective_state/claim"], oracle.size + 1, claim_every)
    speed, spread = claimed_speed(claim[1:])  # (moves, candidates) each
    regret = oracle - incumbent
    return {
        "simple_regret": regret,
        "cumulative_regret": np.cumsum(regret),
        "reconstruction_error": np.sqrt(np.mean((speed - truth) ** 2, axis=-1)),  # over candidates, per move
        "posterior_uncertainty": np.mean(spread, axis=-1),  # a deviation (m/s)
        # share of candidates the claim puts on the wrong side of the threshold
        "level_set_error": np.mean((truth > LEVEL_SET_THRESHOLD) != (speed > LEVEL_SET_THRESHOLD), axis=-1),
    }


def opening_moves(config: dict[str, Any], moves: int) -> int:
    """Moves walked at random before any waypoint exists (the whole episode where there is no rule)"""
    method = config.get("method")
    if method is None:
        return moves
    return int(method["opening_legs"]) * int(method["planner"]["max_steps"])


def episode_legs(held: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    """One row per finished waypoint leg: the move it ended at, the steps walked, the steps predicted, and whether it arrived"""
    method = config.get("method")
    if method is None:
        return {name: np.zeros(0) for name in ("ended", "walked", "predicted", "arrived")}
    budget = int(method["planner"]["max_steps"])
    since = held["agent_state/steps_since_waypoint"]  # (moves + 1,) zero on a replan and up one a move
    ends = np.flatnonzero(np.diff(since) < 0)  # the last snapshot of each waypoint leg, where the sawtooth drops
    ends = ends[ends > opening_moves(config, since.size - 1)]  # the drop closing the opening ends no waypoint leg
    walked = since[ends].astype(float)
    return {
        "ended": ends.astype(float),
        "walked": walked,
        "predicted": held["agent_state/scores.predicted_step_cost"][ends],
        "arrived": (walked < budget).astype(float),  # a waypoint leg ends on arrival or on the budget: short means arrived
    }


def episode_diagnostics(held: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """What one episode did to get there, a value per move"""
    moves = int(held["reward"].shape[0])
    stood = held["state/position"]  # (moves + 1, 2) degrees, already snapped to cells by the transition
    fresh = np.zeros(stood.shape[0], bool)
    fresh[np.unique(stood, axis=0, return_index=True)[1]] = True  # the first snapshot each distinct cell was stood on
    altitude = held["state/altitude"]  # (moves + 1,) which of the record's altitudes it flew at
    flown = held["objective_state/incumbent"][1:]  # (moves,) the fastest wind stood in by each move
    legs, after = episode_legs(held, config), np.arange(1, moves + 1) > opening_moves(config, moves)
    finished, arrived = np.zeros(moves + 1), np.zeros(moves + 1)
    np.add.at(finished, legs["ended"].astype(int), 1.0)
    np.add.at(arrived, legs["ended"].astype(int), legs["arrived"])
    counted = np.cumsum(finished)[1:]
    shares = {}  # a share of candidates is nan before a rule has scored anything
    for name, channel in (("zero_value_candidates", "frac_zero_value_candidates"), ("reachable_candidates", "frac_reachable_candidates")):
        column = held.get(f"agent_state/scores.{channel}")  # absent on an arm that runs no rule
        shares[name] = np.full(moves, np.nan) if column is None else np.where(after, column[1:], np.nan)
    return {
        "cells_visited": np.cumsum(fresh)[1:],  # (moves,) distinct cells stood on by each move
        "altitude_changes": np.cumsum(np.abs(np.diff(altitude))),  # (moves,) one unit of resource apiece
        "fastest_wind_flown": flown,
        "arrival_rate": np.where(counted > 0, np.cumsum(arrived)[1:] / np.maximum(counted, 1.0), np.nan),
        **shares,
    }


def episode_cost(held: dict[str, Any]) -> dict[str, float]:
    """What one episode cost: seconds whole and per move, and peak bytes of the process it ran in"""
    return {
        "seconds": float(held["time_per_episode"]),
        "seconds_per_move": float(np.mean(held["time_per_decision"])),
        "peak_bytes": float(held["peak_rss_per_process"]),
    }


def latest_launch(sweep: Path) -> Path:
    """The directory holding cells: `sweep` itself, or the newest run under it"""
    return sweep if any(sweep.glob("*/record.npz")) else max(launch for launch in sweep.iterdir() if launch.is_dir())


def finished_cells(sweep: Path) -> list[tuple[str, int, Path, dict[str, Any]]]:
    """Every finished cell under one sweep directory: the arm, the seed, where it was written, and the settings it ran"""
    found: list[tuple[str, int, Path, dict[str, Any]]] = []
    for marker in sorted(sweep.glob("*/record.npz")):  # a record is renamed into place whole, so its presence means finished
        varied = settings(marker.parent.name)
        arm = ",".join(f"{key}={value}" for key, value in varied.items() if key != "seed") or "one arm"
        found.append((arm, int(varied.get("seed", 0)), marker.parent, json.loads((marker.parent / "config.json").read_text())))
    if not found:
        raise ValueError(f"no finished cell under {sweep}")
    return found


def tables(cells: list[tuple[str, int, Path, dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three tables over one launch: what each episode achieved per move, what each waypoint leg walked, and what each episode cost"""
    curves: list[pd.DataFrame] = []
    legs: list[pd.DataFrame] = []
    spent: list[dict[str, Any]] = []
    for arm, seed, cell, config in cells:
        held = channels(cell, TABLE_CHANNELS)
        curves.append(
            pd.DataFrame(
                {
                    "arm": arm,
                    "seed": seed,
                    "step": np.arange(1, held["reward"].shape[0] + 1),
                    **episode_curves(held, int(config["claim_every"])),
                    **episode_diagnostics(held, config),
                }
            )
        )
        legs.append(pd.DataFrame({"arm": arm, "seed": seed, **episode_legs(held, config)}))
        spent.append({"arm": arm, "seed": seed, **episode_cost(held)})
        del held  # one cell's channels live at a time
    return pd.concat(curves, ignore_index=True), pd.concat(legs, ignore_index=True), pd.DataFrame(spent)
