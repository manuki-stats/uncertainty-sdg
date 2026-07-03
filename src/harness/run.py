"""The evaluation loop: the reusable core of the whole thesis.

Given ANY generator and a config, run R replications, and in each:
    set the seed  ->  fit the generator  ->  draw m (or K) synthetic sets
    ->  compute E1-E3 on each  ->  pool via the fully-synthetic combining rules
    ->  record coverage / width / bias vs the known truth.

Build this FIRST, test it against the parametric oracle (must give ~95%
coverage), and only then plug in synthpop and the neural generators.
"""
from __future__ import annotations


def run_experiment(config, generator, truth) -> None:
    """Run one experiment cell end to end, appending rows to the results log.

    Skeleton only. Implement once the estimands, metrics, and combining rules
    are in place. Must be idempotent (skip completed (config, seed)) and
    resumable.
    """
    raise NotImplementedError("Wire together seeds -> generate -> estimands -> combine -> metrics -> log.")


if __name__ == "__main__":
    # Entry point for `python -m src.harness.run --config configs/oracle_sim.yaml`
    raise SystemExit("Harness not yet implemented --- next step after the skeleton.")
