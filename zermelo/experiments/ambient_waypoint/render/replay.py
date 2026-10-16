"""One recorded episode read back onto the grid it ran on, as the arrays a panel draws"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float, Int

from zermelo.experiments.ambient_waypoint.metrics.data import (
    CURVE_CHANNELS,
    at_each_snapshot,
    channels,
    episode_curves,
    opening_moves,
    rule,
    settings,
)
from zermelo.problems.ambient_dynamics import AmbientTransition, PaddedGridDomain, ambient_candidates, ambient_positions
from zermelo.run.record import Record


@dataclass(frozen=True)
class Plan:
    """What the method aimed at, at every snapshot"""

    opening_moves: int
    """Moves walked before any plan existed: the opening legs times the planner's truncation"""

    waypoint: Float[np.ndarray, "snapshots 2"]
    """The cell aimed at, as (y, q), NaN while no plan exists"""

    walk: Float[np.ndarray, "snapshots steps_plus_one 2"]
    """The route imagined to the waypoint, as (y, q) from the position the plan was made at, NaN while no plan exists"""

    replans: Int[np.ndarray, " n_plans"]
    """Snapshots at which a new waypoint took force"""

    steer: Float[np.ndarray, "snapshots n_states"]
    """What the plan steers from every cell, as a displacement along q in coordinates, NaN while no plan exists"""

    scored: Int[np.ndarray, "snapshots n_scored"]
    """Which cells the acquisition was computed at, as position indices"""

    acquisition: Float[np.ndarray, "snapshots n_scored"]
    """The score at each of those cells"""


@dataclass(frozen=True)
class Replay:
    """One episode with its grid resolved: what was true, what was believed, and where the actor went"""

    name: str
    """The directory's own name, arm and seed included"""

    label: str
    """The rule the arm runs, in the words a figure names it by"""

    n_moves: int

    shape: tuple[int, int]
    """Rows and columns of the whole grid, pad included: the controllable axis by the ambient one"""

    extent: tuple[float, float, float, float]
    """(y_lo, y_hi, q_lo, q_hi), the outer edges of the corner cells, in coordinates"""

    spacing: float
    """Width of one cell in coordinates"""

    amplitude: float
    """Standard deviation the field was drawn at, which arrow lengths saturate against"""

    prior_sd: float
    """Standard deviation the belief started at, the top of the uncertainty raster's range"""

    cell: Int[np.ndarray, "n_states 2"]
    """Row and column of every position index: its controllable cell, then its ambient one"""

    pad: Bool[np.ndarray, "rows columns"]
    """True through the ring where no sample counts"""

    leg: int
    """Moves one plan is followed for, the whole episode where there is no plan"""

    path: Float[np.ndarray, "snapshots 2"]
    """Where the actor truly was, as (y, q)"""

    drift: Float[np.ndarray, "snapshots 2"]
    """What the field does there, as (f, 0), or the step back inward through the pad"""

    control: Float[np.ndarray, "moves 2"]
    """What the actor steered on each move, as (0, dq)"""

    truth: Float[np.ndarray, " n_scored"]
    """f at every counted cell"""

    claim: Float[np.ndarray, "snapshots n_scored 2"]
    """What was claimed there: the mean, then the log-variance, NaN at the first snapshot"""

    scored: Int[np.ndarray, " n_scored"]
    """Position index of each of those cells"""

    seconds: Float[np.ndarray, " moves"]
    bytes_held: Int[np.ndarray, " moves"]

    plan: Plan | None
    """Absent from an arm with no rule"""

    curves: dict[str, Float[np.ndarray, " moves"]]
    """Every quantity this episode is scored by, one value per move"""

    def raster(self, name: str, move: int) -> Float[np.ndarray, "rows columns"]:
        """One named quantity laid on the whole grid at `move`, NaN wherever it is not defined"""
        if name == "acquisition":
            if self.plan is None:
                raise ValueError(f"{self.name} ran no rule, so it scored no cells and has no acquisition to draw")
            where, value = self.plan.scored[move], self.plan.acquisition[move]
        else:
            mean, log_variance = self.claim[move, :, 0], self.claim[move, :, 1]
            where = self.scored
            value = {
                "truth": self.truth,
                "belief": mean,
                "uncertainty": np.exp(0.5 * log_variance),  # sd = exp(log_var / 2)
                "error": np.abs(mean - self.truth),
            }[name]
        frame = np.full(self.shape, np.nan, np.float32)
        frame[self.cell[where, 0], self.cell[where, 1]] = value  # scattered by row and column: no index order is assumed
        return frame

    def limits(self, name: str, move: int) -> tuple[float, float]:
        """Low and high of the colour range `name` is drawn on at `move`. Fixed over the episode for every channel
        except for the score (which is rescaled per move)"""
        span = float(np.max(np.abs(self.truth)))
        if name in ("truth", "belief"):
            return -span, span
        if name == "uncertainty":
            return 0.0, self.prior_sd
        if name == "error":
            return 0.0, span
        assert self.plan is not None, "an arm with no rule scores no cells"
        scored = self.plan.acquisition[move]
        return float(np.min(scored)), float(max(np.max(scored), np.min(scored) + 1e-12))

    def milestones(self) -> Int[np.ndarray, " k"]:
        """The moves worth looking at: the first, the last, the end of the opening, and every move a new plan took force"""
        marks = [0, self.n_moves]
        if self.plan is not None:
            marks += [self.plan.opening_moves, *self.plan.replans.tolist()]
        return np.unique(np.clip(np.asarray(marks, int), 0, self.n_moves))

    def stills(self, count: int) -> Int[np.ndarray, " count"]:
        """`count` moves spread over the episode, with every milestone that fits kept"""
        marks = self.milestones()
        spread = np.linspace(0, self.n_moves, count).round().astype(int)
        chosen = marks if marks.size >= count else np.unique(np.concatenate([marks, spread]))
        return chosen[np.linspace(0, chosen.size - 1, min(count, chosen.size)).round().astype(int)]


def cells(launch: Path, *, seed: str | None = None, arm: str | None = None) -> list[Path]:
    """The directories under `launch`, one per episode, kept to one seed or one arm where asked and sorted by name"""
    found = sorted(path.parent for path in launch.glob("*/record.npz"))
    if seed is None and arm is None:  # many seeds of many arms is a sheet nobody reads: one axis is held by default
        seeds = sorted({settings(path.name).get("seed", "") for path in found})
        seed = seeds[0] if len(seeds) > 1 else None
    kept = [path for path in found if seed in (None, settings(path.name).get("seed")) and arm in (None, settings(path.name).get("arm"))]
    if not kept:
        raise ValueError(f"{launch} holds no cell at seed {seed} of arm {arm}; it holds {[path.name for path in found]}")
    return kept


def read(path: Path) -> Replay:
    """The episode recorded in the directory at `path`, on the grid its own configuration describes"""
    record = Record.load(path)
    problem, method, belief = record.config["problem"], record.config["method"], record.config["belief"]
    if (problem["ambient_axes"], problem["controllable_axes"]) != (1, 1):
        raise ValueError(
            f"a grid is drawn in two axes and {path.name} ran {problem['ambient_axes']} ambient "
            f"and {problem['controllable_axes']} controllable"
        )
    ambient = PaddedGridDomain(1, problem["ambient_cells"], problem["ambient_cell_spacing"], problem["ambient_pad"])
    controllable = PaddedGridDomain(1, problem["controllable_cells"], problem["controllable_cell_spacing"], problem["controllable_pad"])
    h = float(problem["ambient_cell_spacing"])
    element = ambient_positions(ambient, controllable).elements()
    y, q = np.asarray(element["ambient"])[:, 0], np.asarray(element["controllable"])[:, 0]  # (n_states,) coordinates of every index
    # the ambient axis is drawn across and the controllable one up, so a row is a controllable cell throughout
    rows = np.asarray(controllable.cell_of(element["controllable"]))[:, 0]  # (n_states,) which controllable cell each index is
    columns = np.asarray(ambient.cell_of(element["ambient"]))[:, 0]
    cell = np.stack([rows, columns], -1)
    grid = np.stack([y, q], -1)  # (n_states, 2) as (y, q), which is the order a panel plots in
    counts = np.asarray(ambient_candidates(ambient, controllable).live)
    scored = np.flatnonzero(counts)
    pad = np.ones((controllable.cells, ambient.cells), bool)
    pad[cell[scored, 0], cell[scored, 1]] = False  # every counted cell is inside the ring

    ambient_at, controllable_at = record.state["ambient"], record.state["controllable"]  # (snapshots, 1) each, as the domains take them
    position = np.stack([np.asarray(ambient_at)[:, 0], np.asarray(controllable_at)[:, 0]], -1)  # (snapshots, 2) where the actor was
    truth = np.asarray(record.objective_state["truth"])[0, :, 0]  # (n_scored,) f at each counted cell, the same at every move
    field = np.full((controllable.cells, ambient.cells), np.nan, np.float32)
    field[cell[scored, 0], cell[scored, 1]] = truth
    under = field[np.asarray(controllable.cell_of(controllable_at))[:, 0], np.asarray(ambient.cell_of(ambient_at))[:, 0]]  # (snapshots,)
    walked_back = np.asarray(ambient.inward_direction(ambient_at))[:, 0] * h  # one cell inward: what the pad does instead of a drift
    drift = np.stack([np.where(np.isnan(under), walked_back, under), np.zeros_like(under)], -1)  # (snapshots, 2) as (f, 0)

    transition = AmbientTransition(
        ambient,
        controllable,
        problem["sigma_flow"],
        problem["sigma_act"],
        problem["choices_per_axis"],
        problem["drift_bound_cells"],
        problem["control_step_cells"],
        problem["control_bound_cells"],
    )
    offsets = np.asarray(transition.offsets)  # (choices_per_axis,) signed cells, the middle one standing still
    steered = transition.control_step * offsets[np.asarray(transition.choices(record.decision["action"]))[:, 0]]  # (moves,) along q

    snapshots = int(record.reward.shape[0]) + 1
    claim = at_each_snapshot(np.asarray(record.objective_state["claim"]), snapshots, int(record.config["claim_every"])).copy()
    claim[0] = np.nan  # the objective opens with zeros in the shape of a claim, which is not one

    index = np.asarray(record.agent_state["scores.waypoint"]) if method is not None else np.zeros(0, int)
    replans = 1 + np.flatnonzero(np.diff(index))  # snapshots at which a new waypoint took force
    plan = None
    if replans.size:
        made = np.arange(index.size) >= replans[0]  # (snapshots,) a plan is in force from the first one onward
        made_at = replans[np.clip(np.searchsorted(replans, np.arange(index.size), "right") - 1, 0, None)]  # (snapshots,)
        walk = grid[np.asarray(record.agent_state["scores.imagined_walk_to_waypoint"])]  # (snapshots, steps, 2)
        act = np.asarray(record.agent_state["policy.act"])[:, :, 0]  # (snapshots, n_states) the plan's choice at every cell
        plan = Plan(
            opening_moves=opening_moves(record.config, int(record.reward.shape[0])),
            waypoint=np.where(made[:, None], grid[index], np.nan),
            walk=np.where(made[:, None, None], np.concatenate([position[made_at][:, None, :], walk], 1), np.nan),
            replans=replans,
            steer=np.where(
                made[:, None], transition.control_step * offsets[np.asarray(transition.choices(jnp.asarray(act)))[..., 0]], np.nan
            ),
            scored=np.asarray(record.agent_state["scores.candidate_indices"]),
            acquisition=np.asarray(record.agent_state["scores.acquisition_value"]),
        )

    return Replay(
        name=path.name,
        label=rule(settings(path.name).get("arm", path.name)),
        n_moves=int(record.reward.shape[0]),
        shape=(controllable.cells, ambient.cells),
        extent=(float(y.min()) - h / 2, float(y.max()) + h / 2, float(q.min()) - h / 2, float(q.max()) + h / 2),
        spacing=h,
        amplitude=float(problem["field_amplitude"]),
        prior_sd=float(belief["amplitude"]),
        cell=cell,
        pad=pad,
        leg=int(method["planner"]["max_steps"]) if method is not None else int(record.reward.shape[0]),
        path=position,
        drift=drift,
        control=np.stack([np.zeros_like(steered), steered], -1),
        truth=truth,
        claim=claim,
        scored=scored,
        seconds=np.asarray(record.time_per_decision),
        bytes_held=np.asarray(record.memory_per_decision),
        plan=plan,
        curves=episode_curves(channels(path, CURVE_CHANNELS), int(record.config["claim_every"])),
    )
