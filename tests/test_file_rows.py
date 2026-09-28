"""Per-file estimates: written beside every chunk, and consistent with it."""
import numpy as np
import pandas as pd

import src.results_io as rio
from src.config import ExperimentConfig
from src.harness.run import run


def test_per_file_rows_match_the_pooled_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(rio, "RAW_ROOT", tmp_path)
    cfg = ExperimentConfig(name="t", generator="oracle", generator_kwargs={"mode": "proper"},
                           n_org=300, m=4, K=3, estimators=("T_alt",))
    run(cfg, 0, 2, 2)
    h = cfg.hash()
    assert rio.chunk_path("t", h, 0, 2).exists() and rio.files_path("t", h, 0, 2).exists()

    main, files = rio.read_cell("t", h), rio.read_cell_files("t", h)
    assert len(files) == 2 * 3 * (1 + cfg.m + cfg.K + cfg.K)    # reps x estimands x files
    assert set(files.columns) >= {"rep", "step", "file", "estimand", "q", "u", "config_hash"}

    for step in (2, 3, 4):
        for (rep, est), g in files[files.step == step].groupby(["rep", "estimand"]):
            row = main[(main.rep == rep) & (main.step == step) & (main.estimand == est)].iloc[0]
            assert np.isclose(g.q.mean(), row.estimate)
            assert np.isclose(g.q.var(ddof=1), row.b)
            assert np.isclose(g.u.mean(), row.u_bar)

    # Step 1 analyses the first file of Step 2
    s1 = files[files.step == 1].set_index(["rep", "estimand"]).q
    s2 = files[(files.step == 2) & (files.file == 0)].set_index(["rep", "estimand"]).q
    pd.testing.assert_series_equal(s1, s2, check_names=False)
