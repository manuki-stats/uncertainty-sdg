# results/

The append-only results log lives here as `results.csv` (git-ignored --- it is
regenerable from configs + seeds, and can grow large).

One row per (config_hash, seed, estimand, metric). Never hand-edit it.
Every number in the thesis traces back to a row here, and every row traces back
to a committed config and a logged seed.
