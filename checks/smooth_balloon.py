"""Two checks on the continuous balloon: its gradient is the finite difference, and the navigator climbs to a waypoint in still air

pixi run crosscheck smooth_balloon
"""

import jax
import jax.numpy as jnp

from zermelo.methods.waypoint_bo.navigator import GradientMPC
from zermelo.problems.balloon import SmoothWind, SphereGrid, Steps, balloon_states
from zermelo.problems.balloon.transition import SmoothBalloonTransition

jax.config.update("jax_enable_x64", True)


def main() -> None:
    grid = SphereGrid(n_lat=16, n_lon=31, lat_first=20.0, lon_first=180.0, lat_step=2.0, lon_step=2.0)
    altitude_km = (20.2, 19.1, 18.0, 16.9, 15.8, 14.8)
    states, altitudes = balloon_states(grid, altitude_km), Steps(altitude_km)
    dynamics = SmoothBalloonTransition(grid, altitudes, step_hours=3.0, climb_cap_km=1.1)
    key = jax.random.key(0)
    wind = SmoothWind(4.0 * jax.random.normal(key, (3, len(altitude_km), grid.size(), 2)), jnp.asarray([0.0, 12.0, 24.0]), grid)
    start = {
        "position": jnp.asarray([33.3, 207.7]),
        "altitude": jnp.asarray(2.4),
        "balloon_resource": jnp.asarray(40.0),
        "hours": jnp.asarray(0.0),
    }
    target = states.embed({"position": jnp.asarray([38.0, 220.0]), "altitude": jnp.asarray(4)})

    def summed_distance_in(field: SmoothWind, actions: jnp.ndarray) -> jnp.ndarray:
        def one(s: dict, a: jnp.ndarray) -> tuple[dict, jnp.ndarray]:
            s2 = dynamics.step({**s, "field": field}, a)
            return s2, jnp.linalg.norm(states.embed({"position": s2["position"], "altitude": s2["altitude"]}) - target)

        return jnp.sum(jax.lax.scan(one, start, actions)[1])

    def summed_distance(actions: jnp.ndarray) -> jnp.ndarray:
        return summed_distance_in(wind, actions)

    # (a) the gradient through a 12-step rollout against central differences
    actions = 0.3 * jax.random.normal(jax.random.key(1), (12,))
    analytic = jax.grad(summed_distance)(actions)
    eps = 1e-5
    numeric = jnp.stack(
        [(summed_distance(actions.at[i].add(eps)) - summed_distance(actions.at[i].add(-eps))) / (2 * eps) for i in range(12)]
    )
    gap = float(jnp.max(jnp.abs(analytic - numeric)) / (jnp.max(jnp.abs(numeric)) + 1e-12))
    print(f"gradient vs finite differences: max relative gap {gap:.2e}  (analytic {analytic[:3]} numeric {numeric[:3]})")
    assert gap < 1e-4, "the gradient does not match the finite difference"

    # (b) still air, waypoint four levels up: the navigator must climb there and stay
    still = SmoothWind(jnp.zeros((1, len(altitude_km), grid.size(), 2)), jnp.zeros(1), grid)
    navigator = GradientMPC(
        horizon=20,
        iterations=100,
        step_size=1.0,
        tolerance=1e-7,
        resource_weight=0.0,
        candidates=100,
        descend_top=1,
        perturbation=0.2,
        n_winds=0,
    )
    here = {**start, "altitude": jnp.asarray(1.0)}
    above = states.embed({"position": here["position"], "altitude": jnp.asarray(5)})
    batch = lambda *fields: jax.tree.map(lambda *a: jnp.stack(a), *fields)  # winds go in as one batch along a leading axis
    plan = navigator.plan(jax.random.key(2), dynamics, states, batch(still), here, above, jnp.zeros(20))
    s, trace = here, []
    for a in plan:
        s = dynamics.step({**s, "field": still}, a)
        trace.append(float(s["altitude"]))
    print(
        "still air, from level 1.0 toward level 5: climbs",
        [round(a, 2) for a in plan.tolist()[:6]],
        "... levels flown",
        [round(x, 2) for x in trace[::4]],
    )
    assert trace[-1] > 4.9 and all(b >= a - 1e-6 for a, b in zip(trace, trace[1:], strict=False)), (
        "the navigator did not climb straight to the waypoint"
    )

    # (c) the same wind four times over is the same problem: the sample-average plan must equal the single-wind plan
    four = navigator.plan(jax.random.key(2), dynamics, states, batch(wind, wind, wind, wind), start, target, jnp.zeros(20))
    single = navigator.plan(jax.random.key(2), dynamics, states, batch(wind), start, target, jnp.zeros(20))
    print("four copies of one wind vs that wind alone: largest climb gap", f"{float(jnp.max(jnp.abs(four - single))):.1e} km")
    assert float(jnp.max(jnp.abs(four - single))) < 1e-4, "averaging over identical winds changed the plan"

    # (d) two winds that disagree: the plan must differ from either wind's own plan, and score no worse than them on the pair
    other = SmoothWind(-wind.wind, wind.hours, grid)
    pair = batch(wind, other)
    hedged = navigator.plan(jax.random.key(2), dynamics, states, pair, start, target, jnp.zeros(20))
    own = navigator.plan(jax.random.key(2), dynamics, states, batch(other), start, target, jnp.zeros(20))
    score = lambda plan: 0.5 * (summed_distance(plan) + summed_distance_in(other, plan))
    print(
        "two opposed winds: hedged plan scores",
        f"{float(score(hedged)):.0f}",
        "vs the plans for each wind alone",
        f"{float(score(single)):.0f} / {float(score(own)):.0f}",
    )
    assert float(score(hedged)) <= min(float(score(single)), float(score(own))) + 1e-3, (
        "the hedged plan is worse on the pair than a one-wind plan"
    )
    print("ok")


if __name__ == "__main__":
    main()
