import pytest

from oraclematch.audit import AntiGamingDetector, wilson_interval
from oraclematch.calibration import ScoredResult
from oraclematch.core.protocol import Molecule


def _scored(disagreements):
    return [
        ScoredResult(
            molecule=Molecule(genome=(i,)),
            fitness=0.0,
            fitness_ci_low=0.0,
            fitness_ci_high=0.0,
            mean_affinity=0.0,
            disagreement=float(d),
            mean_confidence=1.0,
        )
        for i, d in enumerate(disagreements)
    ]


def test_wilson_interval_basic():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.5
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.5 < lo < 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_flag_requires_calibration():
    with pytest.raises(RuntimeError):
        AntiGamingDetector().flag(_scored([0.1, 0.2]))


def test_calibrate_sets_operating_point_on_clean():
    det = AntiGamingDetector()
    thr = det.calibrate(_scored([0.0, 0.01, 0.02, 0.03, 0.05]), target_fpr=0.2)
    assert thr > 0.0
    # a clearly-larger disagreement is flagged; a clearly-smaller one is not
    flags = det.flag(_scored([0.5, 0.0]))
    assert flags == [True, False]


def test_calibrate_rejects_empty_and_bad_fpr():
    with pytest.raises(ValueError):
        AntiGamingDetector().calibrate([], target_fpr=0.05)
    with pytest.raises(ValueError):
        AntiGamingDetector().calibrate(_scored([0.1]), target_fpr=1.5)


def test_audit_separates_hackers_from_clean():
    det = AntiGamingDetector(threshold=0.2)
    scored = _scored([0.01, 0.02, 0.5, 0.6])  # two clean (low), two hackers (high)
    is_hacker = [False, False, True, True]
    report = det.audit(scored, is_hacker)
    assert report.caught == 2 and report.caught_rate == 1.0
    assert report.false_positives == 0 and report.fpr == 0.0
    assert report.caught_rate_ci[0] <= 1.0 <= report.caught_rate_ci[1] + 1e-9


def test_audit_length_mismatch_raises():
    det = AntiGamingDetector(threshold=0.2)
    with pytest.raises(ValueError):
        det.audit(_scored([0.1, 0.2]), [True])
