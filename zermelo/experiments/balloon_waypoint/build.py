"""One resolved configuration into the objects an episode runs on"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from hydra.utils import get_class, instantiate
from jaxtyping import Array, Float

from zermelo.experiments.balloon_waypoint.schema import RunConfig
from zermelo.interface import Agent
from zermelo.methods.random_agent import RandomAgent
from zermelo.methods.waypoint_bo.acquisition import Acquisition
from zermelo.methods.waypoint_bo.agent import GradientAgent, WaypointAgent, posterior_moments
from zermelo.methods.waypoint_bo.belief import BeliefKernel, GPBelief, OracleBelief
from zermelo.problems.balloon import (
    GaussianError,
    GEFSError,
    GridWind,
    WindError,
    balloon_objective,
    balloon_states,
    balloon_transition,
    balloon_world,
    load_wind,
    smooth_transition,
    smoothed,
)
from zermelo.run.episode import Episode

WHITELIST = ("zermelo.*", "gpjax.*")
"""Which modules a setting may name"""

WIND_COMPONENTS = 2
"""(u, v): what a belief over the wind is wide in"""


_FORECAST_DRAW = 0xF0EC
"""Folded into the world's key for a `noisy` forecast, so the error drawn over the forecast is this seed's
own and never the same stream the world resets on"""


def assemble(cfg: RunConfig) -> Episode:
    """The episode one configuration describes, that configuration carried onto its record"""
    k_world, k_agent, k_steps = jax.random.split(jax.random.key(cfg.seed), 3)
    recording = load_wind(Path(__file__).resolve().parents[3] / cfg.problem.wind_path)
    grid, n_alt = recording.grid, recording.n_alt
    is_drifting = cfg.problem.frames > 1
    over, held_at = recording.window(slice(cfg.problem.frame, cfg.problem.frame + cfg.problem.frames))
    truth, predicted = over if is_drifting else tuple(window[0] for window in over)  # (t, alt, pos, uv), or one frame held still
    hours: Float[Array, " t"] | None = held_at if is_drifting else None
    error: WindError
    drawn = GaussianError(
        length_scale_km=cfg.problem.error_lengthscale_km, amplitude_ms=cfg.problem.error_scale, jitter=cfg.problem.error_jitter
    )
    if cfg.problem.forecast == "GEFS":
        error = GEFSError(truth - predicted)  # W - F at every hour the window holds
    elif cfg.problem.forecast == "noisy":
        # the truth is the record's own and stays so; the forecast is it under a drawn error, and the error the
        # world carries is exactly what that took away, so W = F + e is the record again at every error scale
        predicted = truth - drawn.sample(jax.random.fold_in(k_world, _FORECAST_DRAW), grid, n_alt)
        error = GEFSError(truth - predicted)
    else:  # W is the forecast, so the drawn error is what makes them differ
        predicted = truth
        error = drawn
    states = balloon_states(grid, tuple(float(h) for h in recording.altitude_km))
    transition = balloon_transition(recording, states, cfg.problem.step_hours)  # the grid's law: the world's, or the trip pricing
    dynamics = smooth_transition(recording, states, cfg.problem.step_hours, cfg.problem.climb_cap_km) if cfg.problem.continuous else None
    world = balloon_world(
        recording,
        states,
        forecast=predicted,
        error=error,
        transition=transition if dynamics is None else dynamics,
        continuous=cfg.problem.continuous,
        readout=get_class(cfg.problem.readout),
        resource_units=cfg.problem.resource_units,
        margin_lat=cfg.problem.margin_lat,
        margin_lon=cfg.problem.margin_lon,
        hours=hours,
    )
    objective = balloon_objective(
        recording, states, instantiate(cfg.problem.target, _target_whitelist_=WHITELIST), cfg.problem.margin_lat, cfg.problem.margin_lon
    )
    candidates = objective.candidates
    forecast = GridWind(predicted[0] if is_drifting else predicted, grid)
    context = world.readout.context
    belief = (
        # the wind of the world the episode runs in, drawn on its key
        OracleBelief.empty(states, cfg.horizon, WIND_COMPONENTS, world.reset(k_world)[0]["field"], context)
        if cfg.belief.oracle
        else GPBelief.empty(
            states,
            cfg.horizon,  # one row per move, the readings an episode folds in
            WIND_COMPONENTS,
            n_context=sum(context.values()),
            kernel_factors=tuple(BeliefKernel.of(get_class(f.kernel), f.parts, f.lengthscale) for f in cfg.belief.kernel_factors),
            context=context,
            amplitude=jnp.asarray(cfg.belief.amplitude),
            noise=jnp.asarray(cfg.belief.noise),
            n_features=cfg.belief.n_features,
            refit=cfg.belief.refit,
            refit_steps=cfg.belief.refit_steps,
            prior_mean=forecast if cfg.belief.forecast_prior else None,
        )
    )
    agent: Agent[Any]
    if cfg.method is None:
        # no rule and no update: it claims the prior every step
        agent = RandomAgent(claim=posterior_moments(belief, jnp.zeros(sum(context.values()))))
    elif (dynamics is None) != (cfg.method.navigator is None):
        raise ValueError("a continuous balloon needs a navigator, and a balloon on the grid cannot use one")
    else:
        flown = (
            {}
            if dynamics is None
            else dict(
                dynamics=dynamics,
                navigator=instantiate(cfg.method.navigator, _target_whitelist_=WHITELIST),
                interpolate=lambda table: smoothed(table, grid, n_alt),
            )
        )
        agent = (WaypointAgent if dynamics is None else GradientAgent)(
            **flown,
            candidates=candidates,
            transition=transition,
            belief=belief,
            acquisition=Acquisition(
                utility=instantiate(cfg.method.utility, _target_whitelist_=WHITELIST),
                planner=instantiate(cfg.method.planner, _target_whitelist_=WHITELIST),
                n_fields=cfg.method.n_fields,
                n_walks=cfg.method.n_walks,
                n_candidates=cfg.method.n_candidates,
                improvement=cfg.method.improvement,
                step_rate=cfg.method.step_rate,
                steps_from=cfg.method.steps_from,
                combination=cfg.method.combination,
            ),
            position_key="position",
            reading_key="wind",
            context_key="context",
            horizon=cfg.horizon,
            opening_legs=cfg.method.opening_legs,
            claim_every=cfg.claim_every,
        )
    # the settings as plain data, carried onto the record
    return Episode(world, objective, agent, cfg.horizon, dict(cfg.recorded), asdict(cfg), k_world, k_agent, k_steps)
