"""Drawing recorded episodes, as a still or as a film"""

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt  # noqa: E402

from zermelo.experiments.balloon_waypoint.metrics.data import (
    latest_launch,  # noqa: E402
    settings,  # noqa: E402
)
from zermelo.experiments.balloon_waypoint.render.film import film, schedule  # noqa: E402
from zermelo.experiments.balloon_waypoint.render.panels import RASTER_NAMES, compare, contact, detail  # noqa: E402
from zermelo.experiments.balloon_waypoint.render.replay import cells, read  # noqa: E402
from zermelo.experiments.balloon_waypoint.render.style import Style  # noqa: E402
from zermelo.experiments.balloon_waypoint.render.submit import send  # noqa: E402

SWEEPS = Path("results") / __package__.split(".")[-2]
"""Where a sweep named rather than pointed at is looked for: this experiment's own results"""


def arms_of(launch: Path) -> list[str]:
    """Every arm a launch holds, in the order they sort"""
    return sorted({settings(found.parent.name).get("arm", found.parent.name) for found in launch.glob("*/record.npz")})


def lanes(launch: Path, arm: str | None, seed: int | None) -> list[tuple[str | None, int | None]]:
    """The arm and seed each drawing covers, None on an axis laying that axis side by side and both given being one episode"""
    if arm is None:
        return [(None, 0 if seed is None else seed)]
    held = arms_of(launch)
    if arm == "all":
        return [(name, seed) for name in held]
    if arm not in held:
        raise SystemExit(f"{launch.name} holds no arm {arm!r}. It holds:\n  " + "\n  ".join(held))
    return [(arm, seed)]


def draw(chosen: Sequence[Path], into: Path, stem: str, given: argparse.Namespace, style: Style) -> list[Path]:
    """The film or the stills `chosen` asks for, written under `into`"""
    replays = [read(path) for path in chosen]
    alone = len(replays) == 1  # one episode is drawn at full size
    into.mkdir(parents=True, exist_ok=True)
    title = replays[0].label if alone else chosen[0].parent.parent.name.replace("_", " ")
    figure = plt.figure(dpi=style.dpi, facecolor=style.paper)

    def sheet(fig: plt.Figure, move: int) -> None:
        """One sheet at `move`: the full-size panel of a single episode, or one lane for each of many"""
        if alone:
            detail(fig, replays[0], move, style, background=given.background, show_plan=True)
        else:
            compare(fig, replays, move, title, style, background=given.background)

    if given.film:
        written = [into / f"{stem}.mp4"]
        film(figure, sheet, schedule(replays, given.stride), written[0], style, fps=given.fps)
    else:
        last = max(replay.n_moves for replay in replays)
        moves = np.asarray(sorted(int(move) for move in given.moves.split(","))) if given.moves else np.asarray([last])
        if alone and moves.size > 1:  # one episode at many moves has a sheet of its own, laying them in a row
            written = [into / f"sheet-{stem}.png"]
            contact(figure, replays[0], moves, style, background=given.background)
            figure.savefig(written[0], facecolor=style.paper)
        else:
            written = []
            for move in moves:
                figure.clear()
                sheet(figure, int(move))
                written.append(into / f"{stem}-{int(move):05d}.png")
                figure.savefig(written[-1], facecolor=style.paper)
    plt.close(figure)
    return written


def main() -> None:
    """Everything the arguments name, drawn here or as an array on the scheduler"""
    parse = argparse.ArgumentParser(prog="render", description="Draw recorded episodes of a balloon.")
    parse.add_argument("target", type=Path, help="a sweep name, a launch directory, or one cell directory")
    parse.add_argument("--arm", help="lay the seeds of this arm side by side, or of every arm with `all`")
    parse.add_argument("--seed", type=int, help="lay the arms at this seed side by side, 0 by default; with --arm, one episode")
    parse.add_argument("--background", default="truth", choices=RASTER_NAMES, help="what a world panel is shaded by")
    parse.add_argument("--dpi", default="high", choices=("high", "medium", "low"), help="how finely a sheet is drawn")
    parse.add_argument("--stride", type=int, default=10, help="moves between frames of a film; milestones are kept whatever this is")
    parse.add_argument("--fps", type=int, default=2, help="frames a second the film plays at")
    parse.add_argument("--moves", help="stills only: draw exactly these moves, comma separated. The last move by default")
    parse.add_argument("--film", action="store_true", help="write video instead of a still")
    parse.add_argument("--local", action="store_true", help="draw the film here rather than sending it to the scheduler")
    given = parse.parse_args()

    path = given.target if given.target.exists() else SWEEPS / given.target
    if not path.exists():
        raise SystemExit(f"no cell, launch or sweep at {given.target}")
    style = dataclasses.replace(Style(), dpi=Style().resolutions[given.dpi])

    if (path / "record.npz").exists():  # a directory named outright is drawn on its own, whatever else was asked for
        launch, wanted = path, [([path], path / "render", given.background, "")]
    else:
        launch, wanted = latest_launch(path), []
        for arm, seed in lanes(launch, given.arm, given.seed):
            chosen = cells(launch, seed=None if seed is None else str(seed), arm=arm)
            asked = " ".join(part for part in (f"--arm {arm}" if arm else "", f"--seed {seed}" if seed is not None else "") if part)
            if len(chosen) == 1:  # an axis with one value on it is one episode, not a comparison of one
                wanted.append((chosen, chosen[0] / "render", given.background, asked))
            elif arm is None:
                wanted.append((chosen, launch / "render", f"arms-seed-{seed}-{given.background}", asked))
            else:
                wanted.append((chosen, launch / "render", f"seeds-{arm}-{given.background}", asked))

    if given.film and not given.local:
        shared = f"--film --background {given.background} --dpi {given.dpi} --stride {given.stride} --fps {given.fps}"
        # the launch rather than what was asked for: a newer one appearing cannot redirect a task
        print(send(launch, [asked for *_, asked in wanted], shared, wanted[0][1], f"films-{given.background}"))
        return
    for chosen, into, stem, _ in wanted:
        for written in draw(chosen, into, stem, given, style):
            print(written)


if __name__ == "__main__":
    main()
