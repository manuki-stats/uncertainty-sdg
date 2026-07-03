"""Experiment configuration.

Every experiment is fully described by one config (a YAML file in configs/).
A config + a seed must be enough to regenerate any result, with no hidden state.
Fill in the concrete schema when we build the harness; this stub fixes the idea.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    """One experiment cell. Serialise to/from YAML in configs/.

    Fields to populate as the harness takes shape:
      method:     "synthpop" | "ctgan" | "tvae" | "ddpm"
      dataset:    "simulation" | "acs"
      step:       1 (naive) | 2 (m-from-one-fit) | 3 (DGE)
      estimands:  which of E1/E2/E3 to compute
      m, K, R:    #synthetic sets, ensemble size, Monte-Carlo replications
    """
    method: str = ""
    dataset: str = ""
    step: int = 1
    m: int = 1
    K: int = 1
    R: int = 1
    estimands: list[str] = field(default_factory=lambda: ["E1", "E2", "E3"])
    extra: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        """Stable short hash of the config, logged with every result row."""
        import hashlib, json
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]
