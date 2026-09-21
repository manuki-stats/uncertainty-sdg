"""One experiment cell, fully described by a YAML file in configs/.

R is deliberately NOT part of the config: a cell is a design, and replications
are added to it in chunks. The hash identifies the cell, so all chunks of the
same design land in the same results folder whatever R eventually becomes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    generator: str                               # "oracle" | "ctgan" | "tvae" | ...
    generator_kwargs: dict = field(default_factory=dict)
    dataset: str = "simulation"
    target: str = "population"                   # "population" | "sample"
    n_org: int = 2000
    r: int = 2                                   # n_syn = r * n_org
    m: int = 10                                  # files per fit (Steps 1-2)
    K: int = 10                                  # fits per ensemble (Steps 3-4)
    steps: tuple = (1, 2, 3, 4)
    estimators: tuple = ("T_alt", "T_s", "T_s_ppd", "T_f")
    alpha: float = 0.05
    seed0: int = 20260921

    @property
    def n_syn(self) -> int:
        return self.r * self.n_org

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text())
        for key in ("steps", "estimators"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)
