"""YAML configuration loading and merging via OmegaConf.

Supports a three-layer merge strategy:
    default.yaml  →  experiment override  →  CLI dot-list overrides
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


_DEFAULT_CONFIG_PATH = "/content/drive/MyDrive/AMERS/config/default.yaml"


def load_config(
    default_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    cli_overrides: list[str] | None = None,
) -> DictConfig:
    """Load and merge configuration from YAML files + CLI overrides.

    Args:
        default_path: Path to the base ``default.yaml``.  Falls back to the
            standard Drive location if *None*.
        experiment_path: Optional path to an experiment-specific YAML file
            whose values override the defaults.
        cli_overrides: Optional list of OmegaConf dot-list strings, e.g.
            ``["gan.pretrain_epochs=50", "ppo.num_episodes=5"]``.

    Returns:
        Merged :class:`DictConfig`.
    """
    if default_path is None:
        default_path = _DEFAULT_CONFIG_PATH

    base: DictConfig = OmegaConf.load(str(default_path))  # type: ignore[assignment]

    if experiment_path is not None:
        experiment = OmegaConf.load(str(experiment_path))
        base = OmegaConf.merge(base, experiment)  # type: ignore[assignment]

    if cli_overrides:
        cli = OmegaConf.from_dotlist(cli_overrides)
        base = OmegaConf.merge(base, cli)  # type: ignore[assignment]

    return base


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert an OmegaConf DictConfig to a plain Python dict (recursive)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
