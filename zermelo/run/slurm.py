"""Slurm submission config settings"""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from omegaconf import II


@dataclass
class ArrayQueueConf(SlurmQueueConf):
    """Submission settings for one array, read off the sweep's `resources` block."""

    account: str = "dean"

    timeout_min: int = II("resources.timeout_min")
    cpus_per_task: int = II("resources.cpus")
    mem_gb: int = II("resources.mem_gb")
    gres: str | None = II("resources.gres")
    partition: str = II("resources.partition")
    constraint: str | None = II("resources.constraint")
    array_parallelism: int = II("resources.array_parallelism")

    wckey: str = ""
    """Empty the submitting library writes a workload key here"""

    additional_parameters: dict[str, Any] = field(default_factory=lambda: {"requeue": True})
    """Requeue asks the scheduler to resubmit an evicted job, which reruns from the start and overwrites itself"""

    setup: list[str] = field(default_factory=lambda: ["export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK"])
    """Shell run inside a job before it starts, holding the thread count to what it was given. Write shell variables without braces"""


ConfigStore.instance().store(group="hydra/launcher", name="slurm", node=ArrayQueueConf, provider="zermelo")
