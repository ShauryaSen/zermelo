"""The noise ladder again, with waypoints the balloon can actually reach

The first ladder found that a perfect forecast buys almost nothing: at sigma = 0 the controller closed 5.5%
of the average leg's gap, against 14.5% at sigma = 1.2, the two inside each other's error bars. The reason
is in the same table. A leg can cover about 3 700 km at the very best -- twenty moves of three hours at the
fastest wind the balloon flew, granted in exactly the right direction -- and the average waypoint was 2 214
km away, with 17% of them further off than any leg could reach. The balloon is mostly being sent somewhere
it cannot get to, and the accuracy of its map makes little difference to that.

So this sweep is the same ladder with the candidate region narrowed, by raising `margin_lon` from 3 to 22.
That leaves 17 of the 61 longitude columns, about 2 800 km across, which is inside a leg's reach. Every
waypoint then becomes reachable by construction and `out of reach` should go to zero.

The point is to find out whether forecast quality matters once the task is possible. If `gap closed` at
sigma = 0 rises well clear of the higher levels, the ladder means something and this is the configuration to
run it in. If it stays flat, then something other than the forecast is the binding constraint and we have to
look further upstream before any noise result is worth reporting.

The cost is comparability: this is a different task from `mpc_noise_s*` and from `era5_collect_continuous`,
which both use `margin_lon = 3`. It is a second configuration, not a correction of the first.
"""

import dataclasses

from zermelo.experiments.balloon_waypoint.setup import arm, posterior_spread, sweep
from zermelo.experiments.balloon_waypoint.sweeps.bounds_noise import NAVIGATION_ONLY, RESOURCES, SEEDS, SIGMAS
from zermelo.experiments.balloon_waypoint.sweeps.era5_collect_continuous import (
    BELIEF,
    CLAIM_EVERY,
    ERA5_SH_JUL2021,
    HORIZON,
    RECORDED,
    _method,
)

MARGIN_LON = 22
"""Longitude columns at each edge no candidate sits in, up from 3. Leaves 61 - 44 = 17 columns, about
2 800 km, against the roughly 3 700 km a leg could cover at the very best. The margin also constrains where
an episode starts, so the whole flight is confined to the middle of the domain"""

NEAR = dataclasses.replace(ERA5_SH_JUL2021, margin_lon=MARGIN_LON)

for _sigma in SIGMAS:
    sweep(
        f"near_noise_s{int(round(_sigma * 10))}",
        problem=dataclasses.replace(NEAR, forecast="noisy", error_scale=_sigma),
        horizon=HORIZON,
        claim_every=CLAIM_EVERY,
        recorded={**RECORDED, "objective_state/claim": CLAIM_EVERY},
        seeds=SEEDS,
        resources=RESOURCES,
        arms=[arm("mpc", dataclasses.replace(_method(posterior_spread(), improvement=False, n_fields=0), navigator=NAVIGATION_ONLY), BELIEF)],
    )
