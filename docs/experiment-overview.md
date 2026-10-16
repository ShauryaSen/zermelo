# The forecast-noise experiments: a full account of the setup

This document is meant to be enough on its own. If someone asks why a number is what it is, the answer
should be here, along with what we considered instead and what we know is still wrong.

---

## 1. The question

A balloon cannot steer. All it can do is change altitude, and let whatever wind is at that altitude carry
it. Choosing well therefore depends entirely on knowing what the wind is doing, and in practice that
knowledge comes from a forecast, which is always somewhat wrong.

So the question we want to answer is how much a controller's performance depends on the quality of the
forecast it is handed. If we give it a perfect forecast it should do well, and if we give it pure noise it
should do no better than drifting. What happens in between is the interesting part. Does performance decay
gracefully as the forecast degrades, so that a moderately bad forecast is still useful? Or is there a point
where the forecast stops helping at all and the controller collapses?

That shape matters for a practical reason. If performance falls off a cliff at some level of forecast error,
then knowing where that cliff sits tells you how good a forecast you actually need, which is a question
anyone flying one of these has to answer. If instead it decays smoothly, then the controller is robust and
the forecast quality is a tuning concern rather than a hard requirement.

We also want to know whether a controller that accounts for its own uncertainty does better under a bad
forecast than one that simply trusts the forecast's best guess. That is the second experiment, and it is
the natural follow-up: if the forecast is poor, perhaps the right response is to plan for several possible
winds rather than for the single most likely one.

---

## 2. The world the balloon flies in

Everything happens inside one recorded slice of real weather, stored as
`zermelo/problems/balloon/data/era5_npac_wide.npz` and copied from `/share/dean/sps252/`. It was built by
`unicorn/make_zermelo_record.py` in `sdean-group/balloon-env-dev`. The file is gitignored because of its
size, so a fresh clone will not have it and has to copy it in.

The slice covers the North Pacific, from 20 to 50 degrees north and from 120 to 240 degrees east, on a
2-degree grid. That works out to 16 latitudes by 61 longitudes, so 976 horizontal cells, each roughly 220 km
across. There are 6 altitude levels running from 14.78 km up to 20.22 km, which is the band a stratospheric
balloon of this class actually operates in, and they sit a little over a kilometre apart on average. The
record holds 81 time frames spaced 3 hours apart, so it covers 240 hours, or ten days.

This region was chosen because it has the wind structure that makes balloon navigation possible at all.
There are easterlies at the top of the band and westerlies lower down, which means a balloon can choose
between going broadly east or broadly west by choosing its altitude. That is the regime Loon operated in,
and without something like it the problem is uninteresting, because every altitude would take you the same
way.

An important detail about how the episode uses the record: the wind is not frozen. The episode runs across
all 81 frames rather than holding one still, so the wind field drifts and evolves underneath the balloon as
it flies. The wind is a function of latitude, longitude, altitude and time.

The file also contains a second wind field under the keys `forecast_u` and `forecast_v`, which is what the
agent would ordinarily be told. We established exactly what this is: it is the same ERA5 data shifted 24
hours earlier, which is to say a persistence forecast. We checked this directly by comparing the forecast at
each frame against the truth 8 frames (24 hours) earlier, and the maximum absolute difference is exactly
zero. The mode is called `"GEFS"` in the config, which is misleading, since no GEFS data is involved. This
had been an open question in the meeting and is now settled.

The error of that persistence forecast, measured as the RMS of the difference between truth and forecast
across the whole record, is 2.797 m/s in the eastward component and 2.784 m/s in the northward one. This
number matters a great deal later, because it is the natural scale against which to measure any artificial
noise we add.

---

## 3. What the balloon is

The balloon's state has five parts: its latitude, its longitude, its altitude, how much altitude-changing
capacity it has left, and how many hours have elapsed.

Altitude is represented as a real number between 0 and 5 rather than an integer level, and position is a
real latitude and longitude rather than a cell index. Nothing is snapped to the grid. This is handled by
`SmoothBalloonTransition`, and it matters because the whole point of the gradient-based controller is that
it differentiates through the dynamics. A discrete state space would have no gradients to follow.

One step of the simulation advances 3 hours. During that step, the balloon is carried by whatever wind is at
its position and altitude, and separately it may climb or descend by up to 1.1 km, which is about one level.
The resource counter decreases by the size of whatever altitude change was made.

The resource budget is 40 units, measured in levels, spread across an episode of 80 moves. Since the cap per
move is about one level, a balloon that changed altitude at every opportunity would want 80 units, so the
budget is deliberately about half of unlimited. It is meant to represent the fact that changing altitude
costs a real balloon something, whether that is ballast, gas or battery.

There is also a margin of 2 latitude rows and 3 longitude columns at each edge of the grid inside which no
start point or target may sit. This keeps episodes away from the edges of the recorded domain, where the
wind field has no data beyond the boundary and behaviour would be an artifact of the record rather than of
the physics.

---

## 4. How an episode is structured

An episode is 80 moves, which at 3 hours each is the full 240 hours of the record.

The episode does not consist of one long flight to one destination. Instead it is broken into legs. At the
start of each leg a waypoint is chosen, and the balloon then flies toward that waypoint until one of two
things happens: either it gets within 160 km of it, which counts as arrival, or it uses up a budget of 20
moves, which is 60 hours. Either way, a new waypoint is then chosen and a new leg begins.

The arrival radius of 160 km is a little smaller than one grid cell, which is about 220 km across. The
intent is that arriving means genuinely being at the waypoint rather than merely being in its
neighbourhood.

The first 20 moves of every episode are an opening in which the balloon climbs randomly and no waypoint
exists at all. This gives the belief some readings to work with before it is asked to make a decision, and
it means every arm of every experiment starts from a comparably informed state rather than from total
ignorance. The consequence is that only 60 of the 80 moves are scored, and since a leg can take up to 20
moves, a typical episode finishes around 4 legs.

That number, 4 legs per episode, turns out to be one of the most important facts about the whole setup,
because legs are the unit we measure. Four per episode is not many, and it drives the seed counts discussed
in section 8.

---

## 5. What the agent is allowed to know

The agent never sees the wind field. What it gets at each step is a reading of the wind column where it
currently stands, which is to say the wind at all six altitudes at its own position, plus its own state.
This is what a real balloon has: an onboard measurement of local conditions and nothing else.

On top of those readings it maintains a belief about the whole wind field, implemented as a Gaussian
process. The prior mean of that GP is the forecast it was given, and as it collects readings it conditions
on them, so the belief starts out as the forecast and gets pulled toward the truth in the places the balloon
has actually visited.

The kernel is a product of two factors. The spatial factor is a Matérn 5/2 over position and altitude, with
lengthscales of 800 km in each of the three spatial directions and 3.0 in levels of altitude. The temporal
factor is a Matérn 3/2 over elapsed hours with a lengthscale of 24 hours. The amplitude is 2.4 m/s, and 256
random features are used, with pathwise draws when samples are needed.

Two things about this belief are worth knowing, and both are discussed properly in section 10 because they
are defects rather than decisions. The amplitude of 2.4 m/s is annotated in the code as the record's own
forecast error, but as noted above that error is actually 2.79, so the belief is somewhat overconfident
before we do anything to it. And the altitude lengthscale of 3.0 levels says the belief expects forecast
error to be similar up and down the column, which conflicts with how we generate the error.

---

## 6. The task and what the controller optimises

Waypoints are chosen by looking for the place the belief is least certain about, which is the
`posterior_spread` acquisition, and then discounting that by how far away it is. The discounting uses a
value-iteration table over the grid with a step budget of 20, a cost weight of 0.25 and a step rate of 0.5.
Without that travel charge the chooser picked waypoints so far away that no plan could reach them, which is
a problem we have not entirely escaped even with it.

Flying to the chosen waypoint is where the actual control happens, and it is done by first-order model
predictive control, following Pasetsky, Lin, Guo and Dean, arXiv 2511.07761. The implementation is
`GradientMPC`. At each solve it generates 100 random candidate altitude plans over a 20-step horizon, plus
the previous plan shifted forward by one step, plus a plan that simply holds altitude. It scores all of them,
takes the single best, and refines that one by normalised gradient descent for up to 100 iterations, using a
step size of 1.0 under the reparameterisation `u = tanh(v/2)` which keeps climbs inside their physical
limits, stopping early if the gradient norm drops below 1e-7. Candidate tails are perturbed by up to 0.2 to
give the random candidates some diversity.

The cost being minimised is the distance to the waypoint summed over every step of the horizon:

```
J(u) = sum over k of || x_k - target ||
```

Summing the distance rather than only penalising the final distance means that arriving early is rewarded
and so is staying once you have arrived, which is the behaviour we want from something that is meant to hold
a position.

The published formulation includes a second term charging for altitude changes, priced at 50 km of summed
distance per level. **We have set that charge to zero.** The reasoning is that we want this experiment to
measure navigation and nothing else, so that when a number moves we know it moved because the balloon
navigated better or worse rather than because it traded off differently against fuel. The world still
enforces the 40-unit resource budget, so the balloon cannot actually change altitude without limit; the
planner simply does not price it. The predictable consequence is a balloon that spends its altitude budget
freely early on and may find itself unable to manoeuvre later, and we see exactly that: a typical episode
uses 36 of its 40 units. This is worth watching rather than assuming away, because if a run ever hits 40 it
stops being steerable and we would be measuring an exhausted budget instead of a degraded forecast.

Crucially, this whole solve is repeated at **every move**, not once per leg. That is what makes it a
closed-loop controller: every new reading updates the belief, and the plan is recomputed against the updated
belief. It is also why the thing is expensive, since 80 moves each doing roughly 200 twenty-step rollouts
adds up. The paper itself replans every 24 steps, so we are doing 24 times more solves than the method as
published, which is a deliberate choice about what we are measuring but also the obvious place to look if
cost ever becomes the binding constraint.

---

## 7. How the forecast is degraded

This is the heart of the experiment, and the direction of the construction took some working out.

The naive approach would be to take the forecast as given and corrupt the truth to be some noisy version of
it. We rejected that, for three reasons. The truth would then be different at every noise level, so the
achievable ceiling would move as we swept, and we would be unable to tell whether a change in performance
came from a worse forecast or from different weather. Sarah's requirement in the meeting was specifically
that the ground truth be identical on every line, with the noise living only in the forecast. And in that
construction the Gaussian process belief's error model would be correct by construction, since the error
really would be the Gaussian thing the GP assumes, which would flatter the method in a way real forecast
error never does.

So we go the other way. The truth stays exactly as recorded, and the forecast is the truth with noise
subtracted:

```
truth    = ERA5                     identical at every sigma
forecast = truth - n(sigma)         what the agent is told, and the belief's prior mean
error    = truth - forecast         what the world carries, which equals n(sigma)
W        = forecast + error = ERA5  exactly, at every sigma
```

Because the world carries an error term that is exactly what the noise took away, the wind the balloon
actually experiences is the recorded ERA5 field at every noise level, bit for bit. The ground truth is
genuinely fixed, not fixed on average.

This is selected by setting `problem.forecast = "noisy"`, which sits alongside the existing `"GEFS"` mode
(the file's own persistence column) and `"gaussian"` mode (which corrupts the truth instead). The random
draw folds in its own key, `0xF0EC`, so adding this mode does not shift the random stream of any sweep that
already exists.

The noise itself is drawn by `GaussianError` as a Gaussian field with a squared-exponential kernel:

```
K(a, b) = sigma^2 * exp( -||embed(a) - embed(b)||^2 / (2 L^2) ),   L = 400 km
```

It is drawn independently for each altitude and independently for each of the two wind components, and it is
static over the episode window rather than evolving.

Two properties of that noise need stating whenever a result from this experiment is reported, because both
make it harsher than real forecast error at the same nominal sigma.

The first is about altitude independence. The balloon's only decision is which altitude to fly at, so what
actually matters to it is not the absolute error at any one level but the *difference* in error between
levels, since that is what could make it rank two altitudes the wrong way round. If the error at two levels
has correlation rho, the variance of their difference is `2*sigma^2*(1 - rho)`. Our noise is independent
across altitude, so rho is zero and the effective noise on the decision is `sqrt(2)*sigma`, about 41 percent
larger than the nominal figure. A real forecast error that is coherent up the column would preserve the
ranking of altitudes much better even at the same sigma.

The second is that the noise is static. A real forecast gets worse the further ahead it looks, so its error
decorrelates with lead time. Ours does not, which means it is a persistent bias rather than growing
uncertainty, and the belief cannot average it away by waiting.

### The ladder of noise levels

We sweep sigma over 0, 1.2, 2.4, 4.8, 7.2 and 9.6 m/s.

The design intent was to anchor these on the record's own forecast error so that each level means something
physical rather than being an arbitrary number of metres per second, and to space them more finely near that
real error than out in the tail. The anchor used was 2.4, so the ladder was meant to read as zero, a half,
one, two, three and four times the real thing.

As discussed, that anchor was taken from the belief's amplitude comment and the real error is 2.79, so what
the ladder actually represents is 0, 0.43, 0.86, 1.7, 2.6 and 3.4 times the record's forecast error. The
range is still sensible, covering a perfect forecast up to something around three and a half times worse
than persistence, but the tidy multiples are not real and we should not describe them that way.

Sigma equals zero deserves separate mention, because it is not really a data point. With sigma at zero the
forecast is exactly the truth, so the belief's prior mean is exactly right everywhere and the controller has
nothing to be wrong about. It should therefore do about as well as this controller can possibly do. If it
does not, something in the harness is broken and no other row in the table is worth reading. We run it as a
test of ourselves.

One honest limitation: none of this spacing was informed by a pilot run. It came from reasoning about the
physics rather than from seeing where performance actually bends. The single data point we had before
launching, at sigma 2.4, suggests the controller is already struggling at 0.86 times the real error, which
raises the real possibility that whatever transition exists happens below 1.2 and that four of our six
levels sit in a flat uninformative tail. The figure will tell us, and if so the second pass should be denser
at the low end.

---

## 8. How we score it

This changed during development, twice, and the reasons are worth keeping.

### Why not arrival rate alone

The obvious measure is the fraction of legs that reach their waypoint. It is interpretable and it was
already implemented. It has two problems.

It is binary and there are only about four legs per episode, so it is a mean over four coin flips. With 8
seeds the standard error came out at 0.12, which is wider than any difference we would expect between
adjacent rungs of the ladder. That is the main reason seeds went up to 32.

Worse, it was inflated. A leg counts as arrived if it ends before its 20-move budget, and we found legs
ending after a single move having gone nowhere at all, because the acquisition had picked a waypoint that
was already inside the 160 km arrival radius. One example from seed 0: one move, 143 km away at the start
and 143 km away at the end, counted as an arrival. Of nine legs we inspected by hand across four seeds, four
counted as arrivals and only one was a genuine flight to a distant waypoint. Every arrival rate reported
before we caught this is inflated the same way, including in earlier sweeps that have nothing to do with
noise.

### Why not summed distance either

The next candidate is the thing the controller actually minimises, which is the summed distance to the
waypoint. We added it, computed after the fact from the record rather than by changing any simulation code,
since the positions, altitudes and standing waypoint are all already saved. Define it as

```
V = sum over scored moves of || chi(z_t, p_t) - chi(x*_t) ||
```

where `chi` places a state in kilometres as the chord across the sphere followed by the altitude, matching
exactly what `SphereGrid.embed` and `Steps.embed` do inside the controller. So this is the same quantity
`J(u)` is built from and not an approximation of it.

At sigma 2.4 it measured 88,445 km with a standard deviation across seeds of 100,308, which is wider than its
own mean. The reason is that it is absolute. The waypoint chooser is unconstrained, so a seed that happens to
be handed a waypoint thousands of kilometres away accumulates enormous distance no matter how well it flies,
and a seed handed a nearby one accumulates little. `V` therefore reports the acquisition function's taste
more loudly than it reports navigation.

For a noise sweep that is worse than merely noisy, and this is the part that took a moment to see. The
forecast is what the belief is built from, and the belief is what waypoints are chosen from. So changing
sigma changes which waypoints get chosen. An absolute distance would confound flying worse with being sent
further, and those are precisely the two things this experiment needs to keep apart.

### What we actually report

The headline number is the share of the opening gap that a leg closes:

```
gap closed = 1 - d_j(k_j) / d_j(1)
```

where `d_j(t)` is the distance from the balloon to leg `j`'s waypoint at that leg's move `t`. It is one
number per leg, it is scale-free so a far waypoint and a near one contribute comparably, it caps at 1 for a
leg that arrives exactly, and it goes negative for a leg that ends further away than it started. Being
continuous, it is also a lower-variance estimator than the binary arrival indicator.

And we only score legs that had something to fly, which we define as

```
F = { legs j : d_j(1) > arrival radius }
```

Two kinds of leg are excluded, for different reasons. Legs whose waypoint was already inside the arrival
radius are excluded because there was no navigation to do and including them both inflates arrival and
dilutes the gap-closed average with zeros. Legs whose waypoint was further away than the balloon could
possibly have travelled are reported separately as `out of reach` rather than scored, since no controller
could have arrived and scoring them measures the chooser instead of the navigator.

The reachability bound is the distance a leg could cover at best, which is its 20 moves times 3 hours times
the fastest wind the balloon actually flew in during that episode. At around 17 m/s that comes to about 3,700
km. This is a generous bound, since it grants the whole leg at the best speed seen and in exactly the right
direction, so anything failing it is definitively unreachable. We originally built the bound from the
fastest wind anywhere in the record, which was wrong: the record reaches down to 14.8 km where a jet runs at
60 m/s, putting the bound at 13,000 km so that nothing ever failed it. A balloon holding one level cannot
draw on a wind at another level, so the speed has to be one it could actually use.

Alongside those we report `V` and the mean per-leg distance, because the bounds experiment will need them,
the mean opening gap so the reader can see how far the waypoints were, the unrestricted arrival rate so the
effect of the restriction is visible rather than silent, and the existing diagnostics for altitude changes,
cells visited and so on.

---

## 9. The experiments themselves

### Experiment one: the controller across the noise ladder

Six sweeps, `mpc_noise_s0` through `mpc_noise_s96`, named for sigma in tenths of a metre per second. One arm
each, which is the controller as described: GP belief, replanning every move, planning on the posterior mean
alone. 32 seeds per level, so 192 episodes in total.

What we are looking for is the shape of `gap closed` against sigma. Sigma zero should be the best of the six
and that is the self-check. Below that we expect a decline, and the question is whether it is gradual or
whether it falls off somewhere. We also expect a floor, since once the forecast carries less information
than the balloon's own readings, further degradation should stop mattering.

Two things need to stay flat across the ladder for the comparison to be clean: `legs flown` and `out of
reach`. If those drift with sigma, then the forecast is changing which waypoints get chosen enough to change
the task itself, and the confound that `gap closed` was designed to remove has come back in a different
form.

On the cluster each cell takes about 4 minutes with one cpu, and a level of 32 seeds at 12 concurrent takes
8 to 10 minutes.

### Experiment two: does sampling the belief help more when the forecast is worse?

Six sweeps, `winds_noise_s0` through `winds_noise_s96`, each with three arms differing only in `n_winds`,
the number of draws from the posterior that the MPC averages its cost over per solve. Zero means planning on
the posterior mean alone, which is the paper's own setting, and 4 and 16 are the sampled extension. 16 seeds
per level, so 288 episodes.

The reasoning behind the hypothesis is straightforward. With a perfect forecast the posterior mean is the
truth, so averaging over draws can only add sampling noise to a problem that has no uncertainty worth
hedging against, and all three arms should land together. As the forecast decays the posterior mean becomes
a worse summary of what the balloon actually knows, and a plan chosen to do reasonably well across many
plausible winds should beat one optimised for a single wrong guess. So we expect the lines to fan out as
sigma increases. If they stay parallel, sampling buys nothing the mean does not already give, which is also
a result and a useful one, since sampling is the expensive option.

The arms here are a fair comparison but they are not bounds on one another. Their trajectories diverge from
the first solve onward, so each one ends up choosing its own waypoints, and nothing guarantees an ordering.
We are comparing three controllers, which is all we claim.

A measured cell at `n_winds = 16` took 6 minutes 38 seconds against about 4 minutes for `n_winds = 0`. That
is a useful surprise: sixteen sampled winds cost roughly double the mean-only solve rather than sixteen
times, because the solve is vectorised across the wind batch and much of the per-move cost is the GP work
and the compile, neither of which `n_winds` touches much. The whole grid should take 2 to 3 hours.

### What is deliberately not running: the bounds

The original plan included three arms per noise level, forming the sandwich

```
V_plan  <=  V_mpc  <=  V_hind
```

where `hind` is given the true wind field as its belief and solves once per leg, making it the best any plan
could have done knowing the weather; `mpc` is the controller; and `plan` averages over draws from the belief
and commits once per leg, making it the best you could do without reacting. The ratio
`(V_mpc - V_plan) / (V_hind - V_plan)` would then say where the controller sits between committing blind and
knowing the future, with 1 meaning it captures everything closed-loop control can buy.

We have unregistered these sweeps, because that chain of inequalities is a statement about three strategies
on **one task**, and our three arms do not share a task. Each has a different belief, and each chooses its
waypoints from its own belief, so each flies to a different sequence of targets. Three summed distances over
three different problems do have a ratio, but that ratio is not a bound and it is not interpretable. The same
objection applies to arrival rate, so this is not an artifact of how we score navigation.

What a real bound needs is a single waypoint sequence shared by all three arms, so that the navigator is the
only thing that differs between them. That is a change to how an arm is composed rather than a sweep
configuration, which is why it is a separate piece of work. The arms themselves are written and left in
`sweeps/bounds_noise.py`, and `OpenLoopAgent`, which solves once per leg instead of every move, is
implemented in `methods/waypoint_bo/agent.py` and works.

---

## 10. Things we know are wrong

These are recorded rather than fixed, because they are in code that predates this work and changing someone
else's vetted design silently is worse than documenting it.

**The belief's amplitude belongs to a different record.** It is 2.4 m/s, annotated as this record's forecast
error, which is really 2.79. The config is still named `ERA5_SH_JUL2021`, so the figure appears to have been
measured on the Southern Hemisphere record and carried over when the North Pacific record was swapped in.
The effect is a belief about 16 percent overconfident in amplitude and 35 percent in variance before we add
any noise. It is identical at every sigma, so it shifts all the lines together rather than bending the
trend, which is why it is tolerable for now.

**The belief and the noise disagree about altitude.** The kernel's altitude lengthscale of 3.0 levels says
the belief expects forecast error to be coherent up the column, while `GaussianError` draws every altitude
independently. The belief is therefore mis-specified in precisely the direction that matters most, since
altitude is the only thing the balloon controls. We do not know whether this was intentional; it may be a
reasonable deliberate choice, since a perfectly specified belief would make the proxy and true bounds
coincide and hide an error we want to be able to see.

**The planner believes everything is reachable.** The diagnostics report `reachable = 1.000` exactly, which
means the value-iteration table puts every single candidate within 20 steps, including one 6,500 km away
when a leg's actual range is about 3,700 km. Since the acquisition prices waypoints using that table, the
chooser keeps dispatching the balloon after targets it has no range for. One seed spent two full legs, 40
moves, sitting at a constant 6,500 km from its waypoint. This sits upstream of everything the noise sweep
measures, and it is the single biggest problem with the setup as it stands.

**The waypoint region is far larger than a leg's reach.** Related to the above but separately fixable. The
domain is around 10,000 km wide and candidates are restricted only by a 3-column margin, so the most
uncertain cell is routinely further away than any leg could fly. Raising `margin_lon` to around 22 would
confine candidates to roughly 2,800 km, inside a leg's reach, which would drive `out of reach` to zero by
construction. We have not done it, because it changes the task and would make these runs incomparable both
to each other and to `era5_collect_continuous`. It is the obvious thing to do in a second pass.

---

## 11. Decisions still genuinely open

1. Whether the open-loop arms should solve once per leg or once per episode. Per leg is what is implemented,
   on the grounds that a leg is the natural unit since waypoints are rechosen anyway. Per episode would be a
   lower and differently-meaningful bound.
2. Whether the belief should be told how bad the forecast has become. Currently `BELIEF_TRACKS_SIGMA` is
   False, so the belief keeps its 2.4 m/s amplitude while the real error grows to 9.6, becoming progressively
   more overconfident. Setting it True recalibrates at each level. These measure different things: an
   overconfident belief is what a real flight faces, while a calibrated one isolates the effect of the
   forecast itself.
3. What horizon hindsight should be computed at: matched to the planner's 20-move budget, over the whole
   80-move episode, or both, with the difference between them being the price of a finite horizon.
4. Whether `n_winds = 32` is enough for the floor arm when the bounds experiment runs. It was chosen large
   rather than tuned. The right convergence check is whether the plans agree against `N = 64`, not whether
   the cost looks settled, since Monte Carlo error always looks like it is settling.
5. How the three bound arms come to share a waypoint sequence, which is what section 9 waits on.
6. Whether the ladder should be re-anchored and re-spaced once we can see where performance actually bends.

---

## 12. Running any of it

```
# the record has to be in place; .npz files are gitignored
cp /share/dean/sps252/era5_npac_wide.npz ~/zermelo/zermelo/problems/balloon/data/

pixi run check                                      # ruff, mypy, import contracts, and the sweep registry

pixi run sweep  balloon_waypoint mpc_noise_s24      # one level serially in this process
pixi run submit balloon_waypoint mpc_noise_s24      # the same cells as a Slurm array

pixi run draw balloon_waypoint results/balloon_waypoint/mpc_noise_s24   # figures and tables for one sweep

# the ladder: every level on one line, a line per arm, which is the result of the experiment
pixi run python -m zermelo.experiments.balloon_waypoint.metrics.ladder \
    mpc_noise_s0 mpc_noise_s12 mpc_noise_s24 mpc_noise_s48 mpc_noise_s72 mpc_noise_s96

# the same as a figure
pixi run python -m zermelo.experiments.balloon_waypoint.metrics.ladder --plot noise.png \
    winds_noise_s0 winds_noise_s12 winds_noise_s24 winds_noise_s48 winds_noise_s72 winds_noise_s96
```

A note on the scheduler, since it cost us an afternoon. The launcher used to demand `--account=dean`, and
not everyone in the group is associated with that account, which made every array submission fail with
"Invalid account or account/partition combination". The account is now a field on a sweep's `resources`
block, and these sweeps set it to None, which means no `--account` flag at all so the scheduler uses the
submitter's own default association. The partition is a separate thing and is what actually determines which
nodes you land on; `partition=dean` keeps cells on `dean-compute-01`, `dean-compute-02` and `dean-cpu-01`.

Also worth knowing: Hydra's submitit launcher blocks until the whole array finishes, so a shell loop over
six sweeps runs them one after another rather than all at once. Background each submission if you want them
queued together. And do not run `pixi run sweep` inside an interactive Slurm job you intend to disconnect
from, because the allocation ends when you disconnect and the scheduler kills everything in it, `nohup`
notwithstanding. We learned that one the direct way.
