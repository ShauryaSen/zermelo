# The forecast-noise experiments

What the setup is, what we run, and what it has told us so far. The FAQ at the end covers why each design
decision is what it is.

---

## 1. The question

A balloon cannot steer. All it can do is change altitude and let whatever wind is at that altitude carry it.
Choosing well therefore depends on knowing what the wind is doing, and in practice that knowledge comes from
a forecast, which is always somewhat wrong.

We want to know how much a controller's performance depends on the quality of the forecast it is handed. A
perfect forecast should give good performance and pure noise should give none, and the interesting part is
the shape in between. If performance falls off a cliff at some level of forecast error, knowing where that
cliff sits tells you how good a forecast you actually need, which is a question anyone flying one of these
has to answer. If it decays smoothly, the controller is robust and forecast quality is a tuning concern.

The second question follows from the first. If the forecast is poor, perhaps the right response is to plan
for several possible winds rather than for the single most likely one. So we also sweep how many wind
samples the controller averages its plan over.

---

## 2. The world

Everything happens inside one recorded slice of real weather, `era5_npac_wide.npz`, copied from
`/share/dean/sps252/` and built by `unicorn/make_zermelo_record.py` in `sdean-group/balloon-env-dev`. The
file is gitignored, so a fresh clone has to copy it in.

The slice covers the North Pacific, 20 to 50 degrees north and 120 to 240 degrees east, on a 2-degree grid.
That is 16 latitudes by 61 longitudes, so 976 horizontal cells about 220 km across. There are 6 altitude
levels from 14.78 km to 20.22 km, a little over a kilometre apart, which is the band a stratospheric balloon
of this class operates in. The record holds 81 frames spaced 3 hours apart, covering 240 hours.

The wind is not frozen during an episode. All 81 frames are used, so the field drifts and evolves underneath
the balloon as it flies. The wind is a function of latitude, longitude, altitude and time.

The file also holds a second wind field under `forecast_u` and `forecast_v`. This is the same ERA5 data
shifted 24 hours earlier, which is to say a persistence forecast. We verified it directly: comparing the
forecast at each frame against the truth 8 frames earlier gives a maximum absolute difference of exactly
zero. The config calls this mode `"GEFS"`, but no GEFS data is involved.

The error of that persistence forecast, as the RMS of truth minus forecast across the whole record, is
**2.797 m/s** eastward and **2.784 m/s** northward. This is the natural scale for any noise we add.

---

## 3. The balloon

The state is latitude, longitude, altitude, remaining altitude-change capacity, and elapsed hours.

Altitude is a real number between 0 and 5 rather than an integer level, and position is a real latitude and
longitude rather than a cell index. Nothing is snapped to the grid, which is what makes the gradient-based
controller possible.

One step advances 3 hours. The balloon is carried by whatever wind is at its position and altitude, and
separately may climb or descend by up to 1.1 km, about one level. The resource counter drops by the size of
whatever altitude change was made.

The resource budget is 40 units in levels, across an episode of 80 moves. Since the cap is about one level
per move, a balloon changing altitude at every opportunity would want 80, so the budget is about half of
unlimited. A margin of 2 latitude rows and 3 longitude columns at each edge keeps starts and targets away
from the boundary of the recorded domain.

---

## 4. The episode

An episode is 80 moves, which at 3 hours each is the record's full 240 hours.

It is broken into legs. At the start of each leg a waypoint is chosen, and the balloon flies toward it until
either it gets within 160 km, which counts as arrival, or it spends a budget of 20 moves, which is 60 hours.
Then a new waypoint is chosen and a new leg begins. The 160 km radius is a little under one grid cell, so
arriving means being at the waypoint rather than near it.

The first 20 moves are an opening in which the balloon climbs randomly and no waypoint exists. This gives
the belief readings to work with before it has to make a decision, and means every arm starts from a
comparably informed state. So 60 of the 80 moves are scored, and since a leg can take up to 20 moves, a
typical episode finishes about 4 legs.

Four legs per episode is one of the most consequential facts about the setup, because legs are the unit we
measure, and it drives the seed counts in section 8.

---

## 5. What the agent knows

The agent never sees the wind field. At each step it gets a reading of the wind column where it stands,
meaning the wind at all six altitudes at its own position, plus its own state. That is what a real balloon
has.

On top of those readings it maintains a Gaussian process belief about the whole field. The prior mean is the
forecast it was given, and it conditions on readings as they arrive, so the belief starts as the forecast
and is pulled toward the truth wherever the balloon has been.

The kernel is a product of two factors. Spatially it is a Matérn 5/2 over position and altitude with
lengthscales of 800 km in each spatial direction and 3.0 in levels of altitude. Temporally it is a Matérn
3/2 over elapsed hours with a lengthscale of 24 hours. The amplitude is 2.4 m/s and 256 random features are
used, with pathwise draws when samples are needed.

---

## 6. The controller

Waypoints are chosen by looking for where the belief is least certain, the `posterior_spread` acquisition,
discounted by how far away it is. The discount uses a value-iteration table over the grid with a step budget
of 20, a cost weight of 0.25 and a step rate of 0.5.

Flying to the waypoint is done by first-order model predictive control, following Pasetsky, Lin, Guo and
Dean, arXiv 2511.07761, implemented as `GradientMPC`. Each solve generates 100 random candidate altitude
plans over a 20-step horizon, plus the previous plan shifted forward one step, plus a plan that holds
altitude. It scores all of them, takes the best, and refines that one by normalised gradient descent for up
to 100 iterations at step size 1.0 under the reparameterisation `u = tanh(v/2)`, stopping early if the
gradient norm falls below 1e-7. Candidate tails are perturbed by up to 0.2 for diversity.

The cost minimised is the distance to the waypoint summed over every step of the horizon:

```
J(u) = sum over k of || x_k - target ||
```

The paper's formulation adds a charge for altitude changes at 50 km of summed distance per level. We set
that charge to **zero**, so the score is navigation alone. The world still enforces the 40-unit budget, and
it binds: a typical episode spends 36 of its 40 units.

This whole solve runs at **every move**, which is what makes the controller closed-loop. Every new reading
updates the belief and the plan is recomputed against it.

---

## 7. How the forecast is degraded

The truth stays exactly as recorded and the forecast is the truth with noise subtracted:

```
truth    = ERA5                     identical at every sigma
forecast = truth - n(sigma)         what the agent is told, and the belief's prior mean
error    = truth - forecast         what the world carries, equal to n(sigma)
W        = forecast + error = ERA5  exactly, at every sigma
```

Because the world carries an error that is exactly what the noise took away, the wind the balloon
experiences is the recorded ERA5 field at every noise level, bit for bit. The ground truth is genuinely
fixed rather than fixed on average.

This is `problem.forecast = "noisy"`, alongside `"GEFS"` (the file's persistence column) and `"gaussian"`
(which corrupts the truth instead). The draw folds in its own key, `0xF0EC`, so no existing sweep's random
stream shifts.

The noise is a Gaussian field with a squared-exponential kernel,

```
K(a, b) = sigma^2 * exp( -||embed(a) - embed(b)||^2 / (2 L^2) ),   L = 400 km
```

drawn independently for each altitude and each of the two wind components, and static over the window.

Two properties of it need stating whenever a result is reported, because both make it harsher than real
forecast error at the same nominal sigma:

**It is independent across altitude.** The balloon's only decision is which altitude to fly at, so what
matters is the *difference* in error between levels, since that is what could rank two altitudes the wrong
way round. For errors with correlation rho the variance of their difference is `2*sigma^2*(1-rho)`, and ours
has rho zero, so the effective noise on the decision is `sqrt(2)*sigma`, about 41 percent larger than the
nominal figure.

**It is static.** A real forecast's error grows with lead time. Ours is a persistent bias instead, which the
belief cannot average away by waiting.

### The ladder

Sigma runs over **0, 1.2, 2.4, 4.8, 7.2 and 9.6 m/s**. Against the record's real forecast error of 2.79
m/s, those are 0, 0.43, 0.86, 1.7, 2.6 and 3.4 times the real thing.

Sigma zero is a test of the harness rather than a data point. The forecast there is exactly the truth, so
the belief's prior mean is right everywhere and the controller has nothing to be wrong about. It should do
about as well as this controller can.

---

## 8. How we score it

The headline number is the share of the opening gap a leg closes:

```
gap closed = 1 - d_j(k_j) / d_j(1)
```

where `d_j(t)` is the distance from the balloon to leg `j`'s waypoint at that leg's move `t`. One number per
leg. It caps at 1 for a leg that arrives exactly and goes negative for one that ends further off than it
started. Being scale-free, a far waypoint and a near one contribute comparably.

We score only legs that had something to fly, `F = { legs j : d_j(1) > arrival radius }`.

Reported alongside it:

| | |
|---|---|
| `arrived` | share of those legs reaching the radius inside the budget |
| `legs flown` | how many legs per episode had something to measure |
| `out of reach` | share of legs whose waypoint was further than any leg could travel |
| `opening gap` | how far the waypoints were, in km |
| `V`, `distance a leg` | the navigator's own cost, the summed distance, in km |
| `arrived, every leg` | arrival without the restriction, so its effect is visible |

`V` is defined as

```
V = sum over scored moves of || chi(z_t, p_t) - chi(x*_t) ||
```

with `chi` placing a state in kilometres as the chord across the sphere followed by the altitude, matching
`SphereGrid.embed` and `Steps.embed` exactly, so this is the same quantity `J(u)` is built from.

The reachability bound behind `out of reach` is a leg's 20 moves times 3 hours times the fastest wind the
balloon actually flew in that episode. At around 17 m/s that is about 3,700 km. It is generous, granting the
whole leg at the best speed seen in exactly the right direction, so anything failing it is definitively
unreachable.

All of these are computed from the saved record after the fact, since positions, altitudes and the standing
waypoint are already stored. No simulation code is involved in producing them.

---

## 9. The experiments

### The controller across the noise ladder

Six sweeps, `mpc_noise_s0` through `mpc_noise_s96`, named for sigma in tenths of a metre per second. One arm
each: GP belief, replanning every move, planning on the posterior mean. 32 seeds per level, 192 episodes.

We are looking for the shape of `gap closed` against sigma, with sigma zero as the best row. Two things need
to stay flat across the ladder for the comparison to be clean, `legs flown` and `out of reach`. If they
drift with sigma then the forecast is changing which waypoints get chosen enough to change the task itself.

A cell takes about 4 minutes on one cpu, and a level of 32 seeds at 12 concurrent takes 8 to 10 minutes.

### Does sampling the belief help more when the forecast is worse?

Six sweeps, `winds_noise_s0` through `winds_noise_s96`, three arms each differing only in `n_winds`, the
number of posterior draws the MPC averages its cost over per solve: 0, 4 and 16. Zero is the posterior mean
alone. 16 seeds per level, 288 episodes.

With a perfect forecast the posterior mean is the truth, so averaging over draws can only add sampling noise
and all three arms should land together. As the forecast decays the mean becomes a worse summary of what the
balloon knows, and a plan that does reasonably well across many plausible winds should beat one optimised
for a single wrong guess. So we expect the lines to fan out with sigma. If they stay parallel, sampling buys
nothing the mean does not already give, which is also a useful result since sampling is the expensive
option.

A measured cell at `n_winds = 16` took 6 minutes 38 seconds against about 4 minutes at `n_winds = 0`, so
sixteen sampled winds cost roughly double rather than sixteen times: the solve is vectorised across the wind
batch, and much of the per-move cost is the GP work and the compile. The whole grid should take 2 to 3
hours.

### The reachable-waypoint ladder

`near_noise_s0` through `near_noise_s96`, the same ladder with `margin_lon` raised from 3 to 22. That leaves
17 of the 61 longitude columns, about 2,800 km, inside a leg's reach, so every waypoint becomes reachable by
construction. Section 10 explains why this exists.

---

## 10. What we have found so far

The first two levels of the ladder came back before the rest, at 32 seeds each:

| sigma | gap closed | arrived | legs flown | out of reach | opening gap |
|---|---|---|---|---|---|
| 0.0 | +0.055 ± 0.099 | 0.366 ± 0.057 | 3.2 ± 0.2 | 0.168 ± 0.060 | 2214 ± 397 km |
| 1.2 | +0.145 ± 0.068 | 0.380 ± 0.056 | 2.9 ± 0.2 | 0.174 ± 0.057 | 2391 ± 409 km |

**The harness check does not pass.** Sigma zero should have been clearly the best row, and it is nominally
the worse of the two. The difference sits inside the error bars, so a degraded forecast is not better, but a
perfect one is not better either, which is what the test required.

Sigma zero really is a perfect forecast. `GaussianError` builds its draw as `amplitude_ms * cholesky(K) @
white`, so at zero amplitude the draw is exactly zero, the forecast equals the truth bit for bit, and the
belief's prior mean is right everywhere.

So the finding is larger than the question we set out to ask: **with a perfect forecast the controller closes
about five percent of the average leg's gap.** Forecast quality cannot be the binding constraint, because
removing forecast error entirely does not help.

The same two rows say why. A leg covers roughly 3,700 km at the absolute best, and the average waypoint is
2,214 km away — about sixty percent of a maximum that assumes the record's fastest wind blowing in exactly
the right direction for sixty consecutive hours. Seventeen percent of waypoints are beyond any reach at all.
The balloon is mostly being asked to do something it cannot do, and under those conditions the accuracy of
its map is a second-order concern.

`near_noise_s*` is the follow-up. If `gap closed` at sigma zero rises clear of the higher levels once every
waypoint is reachable, the ladder means something and that is the configuration to run it in. If it stays
flat, the binding constraint is somewhere else and the next suspects are the 20-step MPC horizon against
2,800 km waypoints, and the altitude budget running out at 36 of 40.

One caution about small samples: the 8-seed figures that preceded this run gave `gap closed` as −0.27, `out
of reach` as 0.062 and the opening gap as 1,151 km. Every one of those moved substantially at 32 seeds.

---

## 11. Known defects

These are recorded rather than fixed, since they live in code that predates this work.

**The belief's amplitude belongs to a different record.** It is 2.4 m/s, annotated as this record's forecast
error, which is really 2.79. The config is still named `ERA5_SH_JUL2021`, so the figure looks measured on the
Southern Hemisphere record and carried over when the North Pacific record was swapped in. The belief is
therefore about 16 percent overconfident in amplitude and 35 percent in variance before any noise is added.
It is identical at every sigma, so it shifts all the lines together rather than bending the trend.

**The belief and the noise disagree about altitude.** The kernel's altitude lengthscale of 3.0 levels
expects forecast error to be coherent up the column, while `GaussianError` draws every altitude
independently. The belief is mis-specified in the one direction that matters most, since altitude is all the
balloon controls. Whether this was intentional is a question for Shaurya.

**The planner believes everything is reachable.** The diagnostics report `reachable = 1.000` exactly, so the
value-iteration table puts every candidate within 20 steps, including one 6,500 km away against a leg's
actual range of about 3,700 km. The acquisition prices waypoints with that table, so the chooser dispatches
the balloon after targets it has no range for. One seed spent two full legs, 40 moves, sitting at a constant
6,500 km from its waypoint. This is the biggest problem with the setup as it stands, and section 10 is it
showing up as a measured result.

---

## 12. Running it

```
# the record has to be in place; .npz files are gitignored
cp /share/dean/sps252/era5_npac_wide.npz ~/zermelo/zermelo/problems/balloon/data/

pixi run check                                      # ruff, mypy, import contracts, the sweep registry

pixi run sweep  balloon_waypoint mpc_noise_s24      # one level serially in this process
pixi run submit balloon_waypoint mpc_noise_s24      # the same cells as a Slurm array

pixi run draw balloon_waypoint results/balloon_waypoint/mpc_noise_s24   # figures and tables for one sweep

# the ladder: every level on one line, a line per arm
pixi run python -m zermelo.experiments.balloon_waypoint.metrics.ladder \
    mpc_noise_s0 mpc_noise_s12 mpc_noise_s24 mpc_noise_s48 mpc_noise_s72 mpc_noise_s96

# the same as a figure
pixi run python -m zermelo.experiments.balloon_waypoint.metrics.ladder --plot noise.png \
    winds_noise_s0 winds_noise_s12 winds_noise_s24 winds_noise_s48 winds_noise_s72 winds_noise_s96
```

Three practical things. Hydra's submitit launcher blocks until the whole array finishes, so a shell loop
over six sweeps runs them one after another; background each submission to queue them together. The account
is a field on a sweep's `resources` and these sweeps set it to None, which means no `--account` flag so the
scheduler uses the submitter's default association — the partition is the separate thing that decides which
nodes you land on. And do not run `pixi run sweep` inside an interactive Slurm job you intend to disconnect
from, because the allocation ends with the session and the scheduler kills everything in it regardless of
`nohup`.

---

## FAQ

**Why subtract noise from the truth instead of corrupting the truth?**
Three reasons. The truth would otherwise differ at every noise level, so the achievable ceiling would move
as we swept and we could not tell a worse forecast from different weather. Sarah asked specifically for the
ground truth to be identical on every line with the noise only in the forecast. And corrupting the truth
would make the GP's error model correct by construction, since the error really would be the Gaussian thing
the GP assumes, which flatters the method in a way real forecast error never does.

**Why is `resource_weight` zero when the paper charges 50 km per level?**
So that when a number moves we know it moved because the balloon navigated better or worse, not because it
traded off differently against fuel. The 40-unit budget still binds in the world, so the balloon cannot
manoeuvre without limit — the planner simply does not price it. The predictable consequence is spending the
budget early and being stuck later, and we do see 36 of 40 used. If a run ever hits 40 we would be measuring
an exhausted budget instead of a degraded forecast, so it is worth watching.

**Why `gap closed` rather than arrival rate?**
Arrival is binary over about four legs an episode, so it is a mean of four coin flips; at 8 seeds its
standard error was 0.12, wider than any difference we expect between adjacent rungs. It was also inflated,
because a leg counts as arrived if it ends before its 20-move budget, and legs were ending after one move
having gone nowhere because the acquisition had picked a waypoint already inside the 160 km radius. One
example: one move, 143 km at the start and 143 km at the end, counted as an arrival. Of nine legs inspected
by hand, four counted as arrivals and one was a genuine flight. `gap closed` is continuous, so it is also a
lower-variance estimator.

**Why not score on the summed distance, which is what the MPC minimises?**
It is absolute, and the waypoint chooser is unconstrained, so a seed handed a waypoint thousands of km away
accumulates enormous distance however well it flies. At sigma 2.4 it measured 88,445 km with a standard
deviation across seeds of 100,308, wider than its own mean. Worse for a sweep: the forecast builds the
belief and the belief chooses the waypoints, so sigma changes which waypoints get chosen, and an absolute
distance would confound flying worse with being sent further. We still report it, because the bounds
experiment needs it.

**Why exclude legs that started inside the arrival radius?**
There was no navigation to do. Including them inflates arrival and dilutes the gap-closed average with
zeros.

**Why these six noise levels?**
They were meant to be anchored on the record's own forecast error so each level means something physical,
spaced finely near that error and coarsely out in the tail where we only need to see whether performance
collapses. The anchor used was 2.4, taken from the belief's amplitude comment, and the real error is 2.79,
so the intended multiples of a half, one, two, three and four are really 0.43, 0.86, 1.7, 2.6 and 3.4. The
range is still sensible. None of the spacing was informed by a pilot, and section 10 suggests a second pass
should be denser at the low end.

**Why replan at every move when the paper replans every 24?**
Reacting to every new reading is what makes it closed-loop, and closed-loop behaviour is what we are
measuring. It does mean we run 24 times more solves than the method as published, which is the obvious knob
if cost becomes binding. It also deserves its own figure: what does replanning 24 times more often actually
buy?

**Why 32 seeds on the ladder and 16 on the n_winds grid?**
Legs are the unit we measure and there are only about four per episode, so arrival is a mean of four binary
draws. At 8 seeds the standard error on it was 0.12, wider than the effect. The n_winds grid is three arms
wide at every level, so 16 seeds keeps it to 288 episodes; `gap closed` is continuous and tighter than
arrival, which makes 16 tolerable there.

**Why is sigma zero in the ladder at all if it is not a data point?**
Because it has to be right. With a perfect forecast the controller must do about as well as it can, and if
it does not, nothing else in the table is worth reading. That is exactly what happened, which is why section
10 exists.

**Why aren't the three bounds running?**
`V_plan <= V_mpc <= V_hind` is a statement about three strategies on one task. Each arm has a different
belief — `hind` is told the true field, `plan` averages 32 draws — and each chooses its waypoints from its
own belief, so each flies to a different sequence of targets. Three summed distances over three different
problems have a ratio, but it is not a bound. A real bound needs one waypoint sequence shared by all three
arms so the navigator is the only difference, which is a change to how an arm is composed rather than a
sweep. The arms are written in `sweeps/bounds_noise.py` and `OpenLoopAgent`, which solves once a leg, is in
`methods/waypoint_bo/agent.py` and works.

**Why is the open-loop arm once a leg rather than once an episode?**
A leg is the natural unit, since waypoints are rechosen anyway. Once an episode would be a lower and
differently-meaningful bound. Still open.

**Should the belief be told how bad the forecast has become?**
Currently not: `BELIEF_TRACKS_SIGMA` is False, so the belief keeps its 2.4 m/s amplitude while the real
error grows to 9.6, becoming progressively more overconfident. Setting it True recalibrates at each level.
These measure different things — an overconfident belief is what a real flight faces, a calibrated one
isolates the effect of the forecast itself. Still open.
