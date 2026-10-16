"""Sending films to the scheduler, one array task per film"""

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from zermelo.experiments.balloon_waypoint.schema import Resources

RESOURCES_PER_FILM = Resources(cpus=2, mem_gb=8, timeout_min=10, gres=None, partition="dean", constraint="avx2", array_parallelism=8)
"""What one film is given"""

SCRIPT = """#!/bin/bash
#SBATCH --job-name=zermelo-film
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={timeout_min}
#SBATCH --array=0-{last}%{parallelism}
#SBATCH --chdir={repository}
#SBATCH --output={into}/{tag}-%a.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FILM=(
{lines}
)
{python} -m zermelo.experiments.balloon_waypoint.render {target} --local {shared} ${{FILM[$SLURM_ARRAY_TASK_ID]}}
"""
"""The array a submission writes, one task per film"""


def send(target: Path, per_film: Sequence[str], shared: str, into: Path, tag: str) -> str:
    """What the scheduler said, having written `into`/`tag`.sbatch with one task per entry of `per_film` and submitted it"""
    if shutil.which("sbatch") is None:
        raise SystemExit("no sbatch on this machine. Draw the film here instead, with --local")
    into.mkdir(parents=True, exist_ok=True)
    script = into / f"{tag}.sbatch"
    script.write_text(
        SCRIPT.format(
            partition=RESOURCES_PER_FILM.partition,
            cpus=RESOURCES_PER_FILM.cpus,
            mem_gb=RESOURCES_PER_FILM.mem_gb,
            timeout_min=RESOURCES_PER_FILM.timeout_min,
            last=len(per_film) - 1,
            parallelism=RESOURCES_PER_FILM.array_parallelism,
            repository=Path.cwd().resolve(),
            into=into.resolve(),
            tag=tag,
            python=sys.executable,
            target=target.resolve(),
            shared=shared,
            lines="\n".join(f'  "{film}"' for film in per_film),
        )
    )
    said = subprocess.run(["sbatch", str(script)], capture_output=True, text=True, check=True)  # noqa: S603
    return f"{said.stdout.strip()}, {len(per_film)} film{'s' if len(per_film) != 1 else ''} from {script}"
