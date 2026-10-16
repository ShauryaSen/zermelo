"""The three bounds as the forecast is made worse: hindsight, our controller, and the best fixed plan.

The truth is the record's own ERA5 at every level of noise (`forecast="noisy"`), so only what the balloon
believes changes. Three arms differ in exactly two things, the belief and the moment of solving:

    hind   the true field as the belief, solved once a leg   -- the ceiling; reads no forecast, so it is
                                                                one number per seed, the same at every sigma
    mpc    the GP belief, replanned every move               -- the controller we are measuring
    plan   the GP belief over N draws, solved once a leg     -- the floor: committed, cannot react

and the number reported is where the controller sits between the other two,

    (V_mpc - V_plan) / (V_hind - V_plan)

which is what says whether the controller holds up as the forecast decays, rather than merely that
everything gets worse together.

The cost is navigation alone: the distance to the waypoint summed over the steps of a leg. The resource
charge of the paper's formulation is priced at zero here (`resource_weight=0`), so an altitude change costs
the plan nothing. The world still caps altitude changes by `resource_units`, so the budget binds even when
the planner does not price it.

One sweep is registered per noise level, `bounds_noise_s0` through `bounds_noise_s96`, named for sigma in
tenths of a metre per second.
"""

import dataclasses

from zermelo.experiments.balloon_waypoint.schema import Resources
from zermelo.experiments.balloon_waypoint.setup import Implementation, arm, posterior_spread, sweep
from zermelo.experiments.balloon_waypoint.sweeps.era5_collect_continuous import (
    BELIEF,
    CLAIM_EVERY,
    ERA5_SH_JUL2021,
    HORIZON,
    NAVIGATOR,
    PLANNING_BUDGET,
    RECORDED,
    _method,
)

SIGMAS = (0.0, 1.2, 2.4, 4.8, 7.2, 9.6)
"""Forecast error swept in m/s, spanning a perfect forecast up to about three and a half times the record's
own persistence error, which measures 2.79 m/s per component (2.797 in u, 2.784 in v). Note that the
belief's amplitude is 2.4, a figure carried over from the Southern Hemisphere record, so the belief is
already a little overconfident at sigma = 0; see docs/noise-sweep.md"""

N_WINDS = 32
"""Draws from the belief the best fixed plan is averaged over. Chosen large rather than tuned: too few makes
the floor look worse than it is, which widens the gap and flatters the controller"""

SEEDS = range(8)

BELIEF_TRACKS_SIGMA = False
"""Whether the belief is told how wrong the forecast now is. False leaves its amplitude at the record's own
2.4 m/s, so the belief grows overconfident as the forecast decays -- which is what a real flight faces, and
what makes the proxy bound differ from the true one. True would calibrate it at every level instead"""

NAVIGATION_ONLY: Implementation = {**NAVIGATOR, "resource_weight": 0.0}
"""The paper's solver, its cost the summed distance to the waypoint and nothing else"""

RESOURCES = Resources(cpus=4, mem_gb=16, timeout_min=60, gres=None, partition="dean", constraint=None, array_parallelism=12)
"""One cell is about 70 s of flying after a two-minute compile, measured on dean-cpu-01 at sigma = 2.4, so an
hour is about ten times what a cell needs. The three-arm sweeps carry the oracle and the 32-draw plan, which
are slower, and still sit far inside it"""


def _arms(sigma: float) -> list:
    """The three bounds at one noise level, over a belief told or not told how wrong it is"""
    believed = dataclasses.replace(BELIEF, amplitude=max(sigma, 0.1)) if BELIEF_TRACKS_SIGMA else BELIEF
    base = _method(posterior_spread(), improvement=False, n_fields=0)
    rule = dataclasses.replace(base, navigator=NAVIGATION_ONLY)
    return [
        arm("hind", dataclasses.replace(rule, open_loop=True), dataclasses.replace(believed, oracle=True)),
        arm("mpc", rule, believed),
        arm("plan", dataclasses.replace(rule, navigator={**NAVIGATION_ONLY, "n_winds": N_WINDS}, open_loop=True), believed),
    ]


for _sigma in SIGMAS:
    # the controller alone at one noise level: what a day's run needs, and the middle line of the three
    sweep(
        f"mpc_noise_s{int(round(_sigma * 10))}",
        problem=dataclasses.replace(ERA5_SH_JUL2021, forecast="noisy", error_scale=_sigma),
        horizon=HORIZON,
        claim_every=CLAIM_EVERY,
        recorded={**RECORDED, "objective_state/claim": CLAIM_EVERY},
        seeds=SEEDS,
        resources=RESOURCES,
        arms=[_arms(_sigma)[1]],
    )
    sweep(
        f"bounds_noise_s{int(round(_sigma * 10))}",
        problem=dataclasses.replace(ERA5_SH_JUL2021, forecast="noisy", error_scale=_sigma),
        horizon=HORIZON,
        claim_every=CLAIM_EVERY,
        recorded={**RECORDED, "objective_state/claim": CLAIM_EVERY},
        seeds=SEEDS,
        resources=RESOURCES,
        arms=_arms(_sigma),
    )

HORIZONS = (PLANNING_BUDGET, HORIZON)
"""The two lookaheads hindsight is taken at: matched to the planner's budget, and the whole episode. The
first asks whether a planner with the same sight could have done better knowing the wind; the second is the
ceiling itself, and their difference is the price of a finite horizon"""
