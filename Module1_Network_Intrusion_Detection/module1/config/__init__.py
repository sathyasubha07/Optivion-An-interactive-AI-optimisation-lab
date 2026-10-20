from __future__ import annotations

from pathlib import Path

from module1.config.settings import Module1Config, load_config

__all__ = ["Module1Config", "load_config", "CONFIG_DIR", "DEFAULT_CONFIG_PATH"]

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
