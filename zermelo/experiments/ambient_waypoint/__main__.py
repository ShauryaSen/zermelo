"""One episode per invocation: `python -m zermelo.experiments.ambient_waypoint +sweep=<name>`"""

import os
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from zermelo.experiments.ambient_waypoint import sweeps  # noqa: F401  registers the arms and the sweeps
from zermelo.experiments.ambient_waypoint.schema import RunConfig, resolve
from zermelo.run import slurm  # noqa: F401  registers the cluster runner

ConfigStore.instance().store(name="run", node=RunConfig)


@hydra.main(config_name="run", config_path=None)
def main(cfg: DictConfig) -> None:
    """One episode at the configured seed, its record written into the directory this invocation was given"""
    import jax  # imported here and not at the top: the submitting process runs no episode and opens no device

    from zermelo.experiments.ambient_waypoint.build import assemble

    if os.environ.get("SLURM_JOB_ID") and cfg.resources.gres and jax.default_backend() == "cpu":
        raise RuntimeError(f"this cell asked for {cfg.resources.gres} and its jax runs on the processor: submit with `pixi run -e cuda`")
    settings = resolve(cfg)  # into dataclasses, which refuse a setting nobody gave
    record = assemble(settings).run()
    record.save(Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
