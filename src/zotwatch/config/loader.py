"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Configuration loader with environment variable expansion."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.config_path = self.base_dir / "config" / "config.yaml"

    def load(self) -> dict[str, Any]:
        """Load and parse configuration file."""
        return _load_yaml(self.config_path)

    def get_data_dir(self) -> Path:
        """Get data directory path."""
        return self.base_dir / "data"

    def get_reports_dir(self) -> Path:
        """Get reports directory path."""
        return self.base_dir / "reports"

    def get_templates_dir(self) -> Path:
        """Get templates directory path."""
        return self.base_dir / "templates"


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in configuration."""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    if isinstance(data, str):
        return os.path.expandvars(data)
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file with environment variable expansion."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data = _expand_env_vars(data)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at the top level.")
    return data


__all__ = ["ConfigLoader", "_expand_env_vars", "_load_yaml"]
