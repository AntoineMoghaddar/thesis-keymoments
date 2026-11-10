"""Configuration and folder layout."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Read the default configuration and lay an optional second file and overrides on top."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


@dataclass
class Paths:
    """Folders below the data directory. Nothing in here belongs in the repository."""

    data: Path
    results: Path
    segments: Path

    @classmethod
    def from_config(cls, cfg: dict) -> "Paths":
        data = Path(cfg["data_dir"])
        seg = Path(cfg.get("segments_dir") or ROOT / "configs" / "segments")
        return cls(data=data, results=Path(cfg["results_dir"]), segments=seg)

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def participants(self) -> Path:
        return self.data / "participants"

    @property
    def prepared(self) -> Path:
        return self.data / "prepared"

    @property
    def review(self) -> Path:
        return self.data / "review"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.participants, self.prepared, self.review, self.results, self.segments):
            p.mkdir(parents=True, exist_ok=True)
        return self
