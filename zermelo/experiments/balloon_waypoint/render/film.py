"""One episode as a film: one sheet per move, streamed to a video file"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Int
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure

from zermelo.experiments.balloon_waypoint.render.replay import Replay
from zermelo.experiments.balloon_waypoint.render.style import Style


def schedule(replays: Sequence[Replay], stride: int) -> Int[np.ndarray, " frames"]:
    """Every `stride`-th move up to the longest episode, with the last move and every episode's milestones kept"""
    horizon = max(replay.n_moves for replay in replays)
    paced = np.arange(0, horizon + 1, max(1, stride))
    marks = [replay.milestones() for replay in replays]
    return np.unique(np.concatenate([paced, np.asarray([horizon]), *marks]))


def film(
    fig: Figure, draw: Callable[[Figure, int], None], moves: Int[np.ndarray, " frames"], into: Path, style: Style, *, fps: int
) -> None:
    """`draw` at each of `moves`, written to `into` as video one frame at a time"""
    # drawn once to size the sheet every frame shares, then again into the film
    draw(fig, int(moves[0]))
    writer = FFMpegWriter(fps=fps, codec="h264", metadata={"title": into.stem}, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(into), dpi=style.dpi):
        for frame, move in enumerate(moves):
            fig.clear()
            draw(fig, int(move))
            writer.grab_frame()
            print(f"\rframe {frame + 1} of {moves.size}", end="", flush=True)
    print()
