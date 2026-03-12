from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def _get_config_path() -> str:
    """Return path to the app config JSON file."""
    return os.path.join(_PROJECT_ROOT, "Data", "app_config.json")


def load_config() -> dict:
    """Load persisted app configuration."""
    path = _get_config_path()
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(config: dict) -> None:
    """Save app configuration to JSON file."""
    path = _get_config_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except OSError:
        logger.warning("Could not save config to %s", path)
