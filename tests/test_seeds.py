from src.seeds import derive_seed


def test_seeds_are_stable_and_distinct():
    a = derive_seed(1, rep=3, step=2, purpose="gen", idx=0)
    assert a == derive_seed(1, rep=3, step=2, purpose="gen", idx=0)
    others = {derive_seed(1, 3, 2, "gen", 1), derive_seed(1, 3, 2, "train", 0),
              derive_seed(1, 3, 3, "gen", 0), derive_seed(1, 4, 2, "gen", 0),
              derive_seed(2, 3, 2, "gen", 0)}
    assert a not in others and len(others) == 5
