"""Integration tests for the controlled synthetic experiments.

These are deterministic (all seeds fixed), so the asserted relationships are reproducible. They
check the *mechanism and honesty* of the controls, not any particular effect size.
"""

from oraclematch.experiments import control_a, control_b


def test_control_a_structure_and_ordering():
    res = control_a(n_seeds=8, generations=10, population=16)
    assert res["backend"] == "mock"
    methods = res["methods"]
    assert set(methods) == {"random", "greedy_single", "ensemble_mean", "oracle_matched"}
    # using two oracles + penalty should not do worse than random selection on the mean metric
    assert methods["oracle_matched"]["grand_mean"] > methods["random"]["grand_mean"]
    for m in methods.values():
        assert m["ci95"][0] <= m["grand_mean"] <= m["ci95"][1]


def test_control_a_penalty_beats_greedy_single_paired():
    res = control_a(n_seeds=8, generations=10, population=16)
    pg = res["verdicts"]["penalty_vs_greedy"]
    # paired mean difference is positive (penalty avoids single-oracle bias exploitation)
    assert pg["mean_diff"] > 0.0


def test_control_a_reports_honest_verdict_strings():
    res = control_a(n_seeds=8, generations=10, population=16)
    for v in res["verdicts"].values():
        assert "verdict" in v and isinstance(v["verdict"], str)
        assert ("CI excludes 0" in v["verdict"]) or ("CI spans 0" in v["verdict"])


def test_control_b_catches_hackers_at_low_correlation():
    res = control_b(shared_fraction=0.5)
    assert res["backend"] == "mock"
    assert res["caught_rate"] > res["fpr"]  # detector discriminates exploiters from clean
    assert res["caught_rate"] > 0.8


def test_control_b_degrades_when_oracles_correlated_kc2():
    """As oracle correlation rises (the KC-2 danger), the disagreement signal weakens and the
    caught-rate drops — empirically demonstrating why KC-2 gates the whole method."""
    low_corr = control_b(shared_fraction=0.5)["caught_rate"]
    high_corr = control_b(shared_fraction=0.9)["caught_rate"]
    assert high_corr < low_corr
