"""Application configuration loading.

Minimal, file+env-var config for settings that must be user-supplied
rather than bundled with BICTool - currently just the two segmentation
model weight paths (see `domain.segmentation.head_ct_unet`/
`aisd_stroke_unet`), which are third-party pretrained weights this project
never ships (see their own module docstrings for why).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_DEFAULT_CONFIG_PATH = Path.home() / ".bictool" / "config.json"

# Environment-variable overrides, applied after the config file - mirrors
# the convention Ashly1991/AISD-HeadCT-Transfer-Learning's own
# `project_paths.py` uses for HEADCT_ROOT/AISD_ROOT.
_ENV_OVERRIDES = {
    "BICTOOL_HEADCT_WEIGHTS": ("segmentation", "anatomy_weights_path"),
    "BICTOOL_STROKE_WEIGHTS": ("segmentation", "stroke_weights_path"),
}


def load_config(path: Optional[str] = None) -> dict:
    """Loads BICTool's user-level configuration.

    Reads a JSON file (`path`, or `~/.bictool/config.json` if not given -
    it being absent entirely is not an error, just an empty starting
    config), then applies any of `_ENV_OVERRIDES`'s environment variables
    on top. Callers should look up nested keys defensively
    (`config.get("segmentation", {}).get(...)`) since any/all of it may be
    absent - there is no bundled default, by design (see module docstring).
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    config: dict = {}
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

    for env_var, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            config.setdefault(section, {})[key] = value

    return config
