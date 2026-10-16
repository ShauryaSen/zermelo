"""The controller as the forecast is made worse, and the arms a bound over it would need.

Only `mpc_noise_s*` is registered. The three-arm `bounds_noise_s*` sweeps are held back: see
`BOUNDS_NEED_A_COMMON_WAYPOINT` below for why they would not be bounds as the arms stand.

The three bounds as the forecast is made worse: hindsight, our controller, and the best fixed plan.

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

One sweep is registered per noise level, `mpc_noise_s0` through `mpc_noise_s96`, named for sigma in tenths
of a metre per second. The three-arm `bounds_noise_s*` sweeps this file was written for are not registered;
see `BOUNDS_NEED_A_COMMON_WAYPOINT`.
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

SEEDS = range(32)
"""An episode finishes only about four waypoint legs, so arrival is a mean over four binary draws and its
spread across eight seeds measured 0.34 -- a standard error of 0.12, too coarse to separate neighbouring
rungs of the ladder. Thirty-two seeds halve it, and a cell is 70 s"""

BELIEF_TRACKS_SIGMA = False
"""Whether the belief is told how wrong the forecast now is. False leaves its amplitude at the record's own
2.4 m/s, so the belief grows overconfident as the forecast decays -- which is what a real flight faces, and
what makes the proxy bound differ from the true one. True would calibrate it at every level instead"""

NAVIGATION_ONLY: Implementation = {**NAVIGATOR, "resource_weight": 0.0}
"""The paper's solver, its cost the summed distance to the waypoint and nothing else"""

RESOURCES = Resources(
    cpus=1,
    mem_gb=6,
    timeout_min=60,
    gres=None,
    partition="dean",
    constraint=None,
    array_parallelism=12,
    account=None,  # the submitter's default association, as an interactive job on this partition uses
)
"""One cell is about 70 s of flying after a two-minute compile, measured on dean-cpu-01 at sigma = 2.4, so an
hour is about ten times what a cell needs. A cell is effectively single-threaded -- eight run serially took
15 m 29 s of user time against 9 m 55 s of real, most of that the compile -- and peaked at 3.3 GB, so it asks
for one cpu and 6 GB rather than holding four cores to use one. The three-arm sweeps carry the oracle and the
32-draw plan, which are slower, and still sit far inside the hour"""


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


BOUNDS_NEED_A_COMMON_WAYPOINT = """The three arms cannot be read as bounds while each chooses its own waypoints.

`V_plan <= V_mpc <= V_hind` holds between a plan, a policy and hindsight *on one task*. The arms here differ
in their belief -- `hind` is told the true field, `plan` averages 32 draws -- and the waypoint is chosen off
that belief, so each arm flies to a different sequence of targets. Their summed distances are then three
numbers about three different problems, and their ratio is not a bound. The same objection applies to the
arrival rate, so it is not a consequence of how navigation is scored.

What the bounds need is one waypoint sequence shared by all three arms, so that only the navigator differs.
That is a change to how an arm is composed rather than a sweep, and it is deliberately left undone here."""

for _sigma in SIGMAS:
    # the controller alone at one noise level: the whole of this sweep, the bounds held back until the arms
    # can be made to share a waypoint sequence (see BOUNDS_NEED_A_COMMON_WAYPOINT)
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

HORIZONS = (PLANNING_BUDGET, HORIZON)
"""The two lookaheads hindsight is taken at: matched to the planner's budget, and the whole episode. The
first asks whether a planner with the same sight could have done better knowing the wind; the second is the
ceiling itself, and their difference is the price of a finite horizon"""
