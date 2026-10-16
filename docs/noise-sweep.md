# The forecast-noise sweep: a specification

What the balloon is, what it is asked to do, what it is told, and how the forecast is degraded. Every number
here is the one the code runs at; where a number is a choice rather than a measurement, the choice is named.

## 1. The world

**Record.** `zermelo/problems/balloon/data/era5_npac_wide.npz`, ERA5 over the North Pacific, built by
`unicorn/make_zermelo_record.py` in `sdean-group/balloon-env-dev`. Copy it from
`/share/dean/sps252/era5_npac_wide.npz`; `.npz` files are gitignored and no clone carries it.

| | |
|---|---|
| domain | 20–50 N × 120–240 E |
| grid | 2°, so 16 × 61 = **976 cells**; a cell is about 220 km |
| levels | 6, from 14.8 km to 20.2 km, 1.08 km apart on average |
| frames | 81, 3 h apart, covering **240 h** |
| regime | easterlies aloft, westerlies below — the Loon regime |
| forecast column | ERA5 **24 h earlier** (persistence), RMS of `W − F` = **2.4 m/s** |

The wind drifts under the balloon: the episode runs over the whole record rather than holding one frame
still, so `W = W(lat, lon, p, t)`.

## 2. State, action, dynamics

Continuous in position and in altitude (`SmoothBalloonTransition`), so nothing is snapped to a cell.

```
s = (lat, lon, p, r, t)        p real in [0, 5] levels, r the resource left, t hours elapsed
a = climb in km, |a| <= 1.1    one level per 3 h step, the cap a real balloon of this class holds
s' : carried by W(s) for 3 h, p' = p + a, r' = r − |a| in levels, t' = t + 3
```

| | | |
|---|---|---|
| `step_hours` | 3.0 | one move |
| `climb_cap_km` | 1.1 | about one level per step |
| `resource_units` | 40 | altitude changes an episode can afford, over 80 moves |
| `margin_lat`, `margin_lon` | 2, 3 | cells at the edge no start or target sits in |

**The budget binds even where the plan ignores it.** The cost no longer prices an altitude change (§5), but
the world still refuses one once `r` is spent.

## 3. The episode

| | | |
|---|---|---|
| `HORIZON` | 80 moves | 240 h, the whole record |
| `PLANNING_BUDGET` (`L`) | 20 moves | 60 h: a leg, after which a waypoint is rechosen |
| opening | 20 moves | uniform random climbs, no waypoint and no planning |
| `ARRIVAL_RADIUS_KM` | 160 km | under one cell, since cells are ~220 km apart |
| `CLAIM_EVERY` | 5 moves | how often the claimed field is written out |

A leg ends when the balloon is within the radius of its waypoint or when the 20 moves are spent; a new
waypoint is then chosen and a new leg begins.

## 4. What the agent sees

**Readout.** The wind column where the balloon stands, plus its own state. Never the field.

**Belief.** A GP over `W`, its prior mean the forecast it was given, fitted to the readings it collects.

| factor | kernel | over | lengthscale |
|---|---|---|---|
| spatial | Matérn 5/2 | position, altitude | 800 km, 800 km, 800 km, **3.0 levels** |
| temporal | Matérn 3/2 | hours | 24 h |

with `amplitude = 2.4 m/s` (the record's own `W − F` RMS), `n_features = 256`, pathwise draws.

**A mismatch worth naming.** The belief's altitude lengthscale of 3.0 levels assumes the forecast error is
coherent up the column. The error injected in §6 is drawn *independently per altitude*. So the belief is
mis-specified in exactly the direction that matters for a balloon, whose only lever is which level to fly.
This is a consequence of the error model, not a decision anyone made; it is left in deliberately because a
well-specified belief would make the proxy and true bounds coincide.

## 5. The task and the cost

**Waypoints** are chosen by maximum posterior variance (`posterior_spread`) and priced by value iteration
(`cost_weight = 0.25`, `step_rate = 0.5`). The flight to one is a navigation problem.

**The leg is flown by first-order MPC** (`GradientMPC`), after Pasetsky, Lin, Guo and Dean, arXiv 2511.07761.

```
J(u) = sum over k of || x_k − target ||          the distance to the waypoint, summed over the horizon
```

Minimising a *sum* rather than a terminal distance means arriving early and staying both pay.

| | | |
|---|---|---|
| `horizon` | 20 | the leg budget |
| `candidates` (`M`) | 100 | random plans, plus the previous plan shifted and a coast plan |
| `descend_top` | 1 | candidates refined by gradient descent |
| `iterations` (`S_max`) | 100 | gradient steps, stopping at `‖g‖ < 1e-7` |
| `step_size` (`η`) | 1.0 | normalised gradient step under `u = tanh(v/2)` |
| `perturbation` | 0.2 | half-width of the noise on a candidate's tail |
| `n_winds` | 0 | plans on the posterior mean; `N` averages `J` over `N` draws |
| **`resource_weight`** | **0.0** | **the paper's resource charge, priced at zero: the score is navigation alone** |

`resource_weight = 0` is a decision, not the paper's default (which is 50 km per level). It makes the cost
purely navigational. The altitude budget of §2 still binds, so the expected behaviour is a balloon that
spends freely early and is then stuck — worth watching rather than assuming away.

## 6. How the forecast is degraded

The truth is the record and stays the record. Only the forecast moves.

```
truth    = ERA5                     unchanged at every σ
forecast = truth − n(σ)             what the agent is told, and the belief's prior mean
error    = truth − forecast = n(σ)  what the world carries
W        = forecast + error = ERA5  exactly, at every σ
```

Selected by `problem.forecast = "noisy"`, beside the existing `"GEFS"` (the file's own column) and
`"gaussian"` (which corrupts the truth instead). The draw uses `fold_in(k_world, 0xF0EC)`, so no existing
sweep's random stream changes.

**The draw.** `GaussianError`: `e ~ N(0, K)` with

```
K(a, b) = σ² exp(−‖embed(a) − embed(b)‖² / 2 L²),    L = error_lengthscale_km = 400 km
```

independent per altitude and per component of `(u, v)`, and static over the window.

**Why this direction.** Hindsight reads only the true wind, so with the truth fixed it is one number per
seed, reused at every σ. Sarah's requirement — the same ground truth on every line, the noise only in the
forecast — then holds by construction. Corrupting the truth instead would move the ceiling with σ, confound
"worse forecast" with "different weather", and make the GP's error model correct by construction.

**The ladder.** σ ∈ {0, 1.2, 2.4, 4.8, 7.2, 9.6} m/s — a perfect forecast, then ½, 1, 2, 3 and 4 times the
record's own persistence error of 2.4 m/s.

**σ = 0 is a test, not a data point.** The forecast is then exactly the truth, so the controller must match
the hindsight bound. If it does not, the harness is wrong.

**Two properties of this noise to state when reporting.** It is independent per altitude, so the error on
the quantity the balloon actually decides on — the difference between levels — has variance `2σ²(1 − ρ)`
with `ρ = 0`, i.e. `√2 σ`. And it is static, so it does not decorrelate with lead time as a real forecast's
error does. Both make this harsher than real forecast degradation at the same σ.

**The belief is not told.** `BELIEF_TRACKS_SIGMA = False`: the GP keeps its 2.4 m/s amplitude while the real
error grows, so it becomes overconfident as σ rises. Setting it `True` recalibrates at every level. This
changes what the experiment measures and is a live question.

## 7. The sweeps

| name | arms | what it is for |
|---|---|---|
| `mpc_noise_s{0,12,24,48,72,96}` | `mpc` | the controller across the ladder |
| `bounds_noise_s{…}` | `hind`, `mpc`, `plan` | the three bounds, when hindsight is wanted |

`SEEDS = range(8)`; resources are 4 cpus, 16 GB, 480 min, partition `dean`.

The three arms differ in exactly two things:

| arm | belief | solves |
|---|---|---|
| `hind` | `OracleBelief` — mean is the true field, variance zero | once a leg |
| `mpc` | `GPBelief` | every move |
| `plan` | `GPBelief`, `n_winds = 32` | once a leg |

and the number they are read through is

```
(V_mpc − V_plan) / (V_hind − V_plan)
```

1 meaning the controller captures everything closed-loop control can buy, 0 meaning it does no better than
committing to a plan and flying it blind.

## 8. Decisions that are still open

1. Open loop is **once a leg**, not once an episode. A leg is the natural unit because waypoints are
   rechosen; per episode would be a lower and different bound.
2. `BELIEF_TRACKS_SIGMA` — overconfident (current) or recalibrated.
3. The hindsight horizon — matched to the 20-move budget, the full 80-move episode, or both.
4. `n_winds = 32` for the floor, chosen large rather than tuned; the convergence check is plan agreement
   against `N = 64`, not cost, since Monte Carlo error always looks like it is settling.

## 9. Running it

```
cp /share/dean/sps252/era5_npac_wide.npz ~/zermelo/zermelo/problems/balloon/data/
pixi run check
pixi run sweep  balloon_waypoint mpc_noise_s24     # one level, serially, in this process
pixi run submit balloon_waypoint mpc_noise_s24     # the same cells as a Slurm array
pixi run draw   balloon_waypoint mpc_noise_s24
```
