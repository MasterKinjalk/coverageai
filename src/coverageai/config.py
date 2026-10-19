"""Load config/spec.yaml once and expose it as a dict."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "config" / "spec.yaml"


@lru_cache(maxsize=1)
def spec() -> dict[str, Any]:
    with open(SPEC_PATH, encoding="utf8") as f:
        return yaml.safe_load(f)


def px_per_mm() -> float:
    return spec()["card"]["dpi"] / 25.4
