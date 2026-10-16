"""A balloon drifts in an unknown wind, steering only by which altitude it flies at.

s = (lat, lon, p, r, W)             where it is, which altitude, what resource is left, the wind carrying it
a in {0, 1, 2}                      down one altitude, hold, up one
W(lat, lon, p) = (u, v)             wind in m/s, u eastward and v northward
(lat, lon)' = (lat, lon) + W(s) h   the wind carries it, h hours to a step
p' = p + a - 1, r' = r - |p' - p|   an altitude change spends one unit of resource, and needs r > 0
W = F + e                           the truth is the forecast F the agent is given, plus the error it carries

A plan and a belief are written over (lat, lon, p) alone; r is a scalar the world carries and the
method prices.

Continuous (`SmoothBalloonTransition`): lat, lon and p are real, a is a climb in km, and W is read between records
"""

from zermelo.problems.balloon.field import (
    DriftingWind,
    ForecastPrior,
    GaussianError,
    GEFSError,
    GridWind,
    SmoothWind,
    WindError,
    WindField,
    smoothed,
)
from zermelo.problems.balloon.grid import SphereGrid, Steps, balloon_states
from zermelo.problems.balloon.objective import ColumnPeakSpeed, PointSpeed, StormSearch, Target, balloon_candidates
from zermelo.problems.balloon.readout import BalloonReadout, ColumnWind, PointWind
from zermelo.problems.balloon.transition import Act, Advection, Ascent, BalloonKernel, BalloonTransition, Factor, SmoothBalloonTransition
from zermelo.problems.balloon.world import (
    Highest,
    WindRecord,
    balloon_objective,
    balloon_transition,
    balloon_world,
    load_wind,
    smooth_transition,
)

__all__ = [
    "Act",
    "Advection",
    "Ascent",
    "BalloonKernel",
    "BalloonReadout",
    "BalloonTransition",
    "ColumnPeakSpeed",
    "ColumnWind",
    "DriftingWind",
    "Factor",
    "ForecastPrior",
    "GEFSError",
    "GaussianError",
    "GridWind",
    "Highest",
    "PointSpeed",
    "PointWind",
    "SmoothBalloonTransition",
    "SmoothWind",
    "SphereGrid",
    "Steps",
    "StormSearch",
    "Target",
    "WindError",
    "WindField",
    "WindRecord",
    "balloon_candidates",
    "balloon_objective",
    "balloon_states",
    "balloon_transition",
    "balloon_world",
    "load_wind",
    "smooth_transition",
    "smoothed",
]
