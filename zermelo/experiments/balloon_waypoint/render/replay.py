"""One recorded episode read back onto the grid it flew over, as the arrays a panel draws"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int

from zermelo.experiments.balloon_waypoint.metrics.data import (
    CURVE_CHANNELS,
    at_each_snapshot,
    channels,
    claimed_speed,
    episode_curves,
    opening_moves,
    rule,
    settings,
)
from zermelo.problems.balloon.grid import balloon_states
from zermelo.problems.balloon.objective import balloon_candidates
from zermelo.problems.balloon.world import load_wind
from zermelo.run.record import Record


@dataclass(frozen=True)
class Plan:
    """What the method aimed at, at every snapshot"""

    opening_moves: int
    """Moves walked before any plan existed: the opening legs times the planner's truncation"""

    waypoint: Float[np.ndarray, "snapshots 3"]
    """The state aimed at, as (lat, lon, altitude), NaN while no plan exists"""

    walk: Float[np.ndarray, "snapshots steps_plus_one 3"]
    """The route imagined to the waypoint, from the state the plan was made at, NaN while no plan exists"""

    replans: Int[np.ndarray, " n_plans"]
    """Snapshots at which a new waypoint took force"""

    scored: Int[np.ndarray, "snapshots n_scored"]
    """Which candidates the acquisition was computed at, as state indices"""

    acquisition: Float[np.ndarray, "snapshots n_scored"]
    """The score at each of those candidates"""


@dataclass(frozen=True)
class Replay:
    """One episode with its world resolved: what the wind was, what was believed, and where the balloon went"""

    name: str
    """The directory's own name, arm and seed included"""

    label: str
    """The rule the arm runs, in the words a figure names it by"""

    n_moves: int

    shape: tuple[int, int]
    """Rows and columns of the grid: latitudes by longitudes"""

    extent: tuple[float, float, float, float]
    """(lon_lo, lon_hi, lat_lo, lat_hi), the outer edges of the corner cells, in degrees"""

    aspect: float
    """Height one degree of latitude is drawn at, per degree of longitude, at the box's own middle"""

    altitude_km: tuple[float, ...]
    """What each altitude index stands for, in km"""

    wind: Float[np.ndarray, "alt lat lon uv"]
    """W at every altitude and cell, as (u, v) in m/s"""

    speed: Float[np.ndarray, "alt lat lon"]
    """||W|| there"""

    prior_sd: float
    """Standard deviation the belief started at, the top of the uncertainty raster's range"""

    cell: Int[np.ndarray, "n_candidates 2"]
    """Row and column of every candidate: its latitude cell, then its longitude one"""

    level: Int[np.ndarray, " n_candidates"]
    """Which altitude each candidate sits at"""

    index: Int[np.ndarray, " n_candidates"]
    """The state index of each candidate, as a plan names the ones it scored"""

    outside: Bool[np.ndarray, "lat lon"]
    """True through the margin where no candidate is scored"""

    leg: int
    """Moves one plan is followed for, the whole episode where there is no plan"""

    path: Float[np.ndarray, "snapshots 2"]
    """Where the balloon truly was, as (lat, lon) in degrees"""

    flown: Float[np.ndarray, " snapshots"]
    """The altitude it flew at, as a level: whole on the grid, fractional for a continuous balloon"""

    balloon_resource: Int[np.ndarray, " snapshots"]
    """What it had left to spend"""

    truth: Float[np.ndarray, " n_candidates"]
    """g at every candidate, the same at every move"""

    claim: Float[np.ndarray, "snapshots n_candidates 4"]
    """What was claimed there: the mean of (u, v), then their log-variances, NaN at the first snapshot"""

    seconds: Float[np.ndarray, " moves"]
    bytes_held: Int[np.ndarray, " moves"]

    plan: Plan | None
    """Absent from an arm with no rule"""

    curves: dict[str, Float[np.ndarray, " moves"]]
    """Every quantity this episode is scored by, one value per move"""

    def raster(self, name: str, move: int) -> Float[np.ndarray, "lat lon"]:
        """One named quantity laid on the grid at `move`, at the altitude being flown, NaN wherever it is not defined"""
        here = int(self.flown[min(move, self.flown.size - 1)])
        at_level = self.level == here  # (n_candidates,) the slice every panel is drawn from
        if name == "acquisition":
            if self.plan is None:
                raise ValueError(f"{self.name} ran no rule, so it scored no candidates and has no acquisition to draw")
            where, value = self.plan.scored[move], self.plan.acquisition[move]
            # each score belongs to its own candidate, so `where` is resolved to rows of `cell` one entry at a
            # time rather than by set membership: a move scores repeats, and names states that are no candidate
            order = np.argsort(self.index)
            slot = np.searchsorted(self.index[order], where)
            kept = slot < self.index.size
            seen = order[np.where(kept, slot, 0)]
            kept &= self.index[seen] == where
            kept &= at_level[seen]  # one score per cell: the candidate at the altitude being flown
            frame = np.full(self.shape, np.nan, np.float32)
            frame[self.cell[seen[kept], 0], self.cell[seen[kept], 1]] = value[kept]
            return frame
        speed, spread = claimed_speed(self.claim[move])  # (n_candidates,) each
        value = {"truth": self.truth, "belief": speed, "uncertainty": spread, "error": np.abs(speed - self.truth)}[name]
        frame = np.full(self.shape, np.nan, np.float32)
        frame[self.cell[at_level, 0], self.cell[at_level, 1]] = value[at_level]
        return frame

    def limits(self, name: str, move: int) -> tuple[float, float]:
        """Low and high of the colour range `name` is drawn on at `move`. Fixed over the episode for every channel
        except for the score (which is rescaled per move)"""
        span = float(np.max(self.truth))
        if name in ("truth", "belief"):
            return 0.0, span
        if name == "uncertainty":
            return 0.0, self.prior_sd
        if name == "error":
            return 0.0, span
        drawn = self.raster(name, move)
        if not np.isfinite(drawn).any():
            return 0.0, 1.0
        low, high = float(np.nanmin(drawn)), float(np.nanmax(drawn))
        return low, max(high, low + 1e-12)

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
    """The episode recorded in the directory at `path`, on the world its own configuration describes"""
    record = Record.load(path)
    problem, method, belief = record.config["problem"], record.config["method"], record.config["belief"]
    recording = load_wind(Path(problem["wind_path"]))
    grid, n_alt = recording.grid, recording.n_alt
    altitude_km = tuple(float(km) for km in np.asarray(recording.altitude_km))
    wind = np.asarray(recording.at(int(problem["frame"]))).reshape(n_alt, grid.n_lat, grid.n_lon, 2)

    states = balloon_states(grid, altitude_km)
    candidates = balloon_candidates(states, grid, int(problem["margin_lat"]), int(problem["margin_lon"]))
    live = np.flatnonzero(np.asarray(candidates.live))  # (n_candidates,) indices into the state domain
    element = states.elements()
    position = np.asarray(element["position"])[live]  # (n_candidates, 2) degrees
    cell = np.asarray(grid.cell_of(element["position"]))[live]  # (n_candidates, 2) row and column
    level = np.asarray(element["altitude"])[live].astype(int)
    outside = np.ones((grid.n_lat, grid.n_lon), bool)
    outside[cell[:, 0], cell[:, 1]] = False  # every scored cell is inside the margin

    snapshots = int(record.reward.shape[0]) + 1
    claim = at_each_snapshot(np.asarray(record.objective_state["claim"]), snapshots, int(record.config["claim_every"])).copy()
    claim[0] = np.nan  # the objective opens with zeros in the shape of a claim, which is not one
    flown = np.asarray(record.state["altitude"]).astype(float)

    index = np.asarray(record.agent_state["scores.waypoint"]) if method is not None else np.zeros(0, int)
    replans = 1 + np.flatnonzero(np.diff(index))  # snapshots at which a new waypoint took force
    plan = None
    if replans.size:
        whole = np.concatenate([np.asarray(element["position"]), np.asarray(element["altitude"])[:, None]], -1)  # (n_states, 3)
        made = np.arange(index.size) >= replans[0]  # (snapshots,) a plan is in force from the first one onward
        made_at = replans[np.clip(np.searchsorted(replans, np.arange(index.size), "right") - 1, 0, None)]  # (snapshots,)
        walk = whole[np.asarray(record.agent_state["scores.imagined_walk_to_waypoint"])]  # (snapshots, steps, 3)
        from_here = np.concatenate([np.asarray(record.state["position"]), flown[:, None]], -1)[made_at]  # (snapshots, 3)
        plan = Plan(
            opening_moves=opening_moves(record.config, int(record.reward.shape[0])),
            waypoint=np.where(made[:, None], whole[index], np.nan),
            walk=np.where(made[:, None, None], np.concatenate([from_here[:, None, :], walk], 1), np.nan),
            replans=replans,
            scored=np.asarray(record.agent_state["scores.candidate_indices"]),
            acquisition=np.asarray(record.agent_state["scores.acquisition_value"]),
        )

    lat_edges = (float(np.min(position[:, 0])), float(np.max(position[:, 0])))
    return Replay(
        name=path.name,
        label=rule(settings(path.name).get("arm", path.name)),
        n_moves=int(record.reward.shape[0]),
        shape=(grid.n_lat, grid.n_lon),
        extent=(
            grid.lon_first - grid.lon_step / 2,
            grid.lon_first + (grid.n_lon - 0.5) * grid.lon_step,
            min(grid.lat_first, grid.lat_first + grid.n_lat * grid.lat_step),
            max(grid.lat_first, grid.lat_first + grid.n_lat * grid.lat_step),
        ),
        aspect=1.0 / float(np.cos(np.radians(0.5 * (lat_edges[0] + lat_edges[1])))),
        altitude_km=altitude_km,
        wind=wind,
        speed=np.linalg.norm(wind, axis=-1),
        prior_sd=float(belief["amplitude"]) if float(belief["amplitude"]) > 0.0 else 1.0,
        cell=cell,
        level=level,
        index=live,
        outside=outside,
        leg=int(method["planner"]["max_steps"]) if method is not None else int(record.reward.shape[0]),
        path=np.asarray(record.state["position"]),
        flown=flown,
        balloon_resource=np.asarray(record.state["balloon_resource"]).astype(int),
        truth=np.asarray(record.objective_state["truth"])[0],
        claim=claim,
        seconds=np.asarray(record.time_per_decision),
        bytes_held=np.asarray(record.memory_per_decision),
        plan=plan,
        curves=episode_curves(channels(path, CURVE_CHANNELS), int(record.config["claim_every"])),
    )
