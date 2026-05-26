import numpy as np
import pytest

from oraclematch.core.normalize import normalize, rank_normalize, zscore_normalize


def test_rank_normalize_range_and_endpoints():
    out = rank_normalize(np.array([3.0, 1.0, 2.0, 5.0]))
    assert out.min() == 0.0 and out.max() == 1.0
    assert np.argmin(out) == 1 and np.argmax(out) == 3  # smallest->0, largest->1


def test_rank_normalize_is_monotone():
    x = np.array([10.0, -3.0, 7.5, 2.2, 100.0])
    out = rank_normalize(x)
    assert list(np.argsort(x)) == list(np.argsort(out))


def test_rank_normalize_ties_share_value():
    out = rank_normalize(np.array([5.0, 5.0, 1.0, 9.0]))
    assert out[0] == out[1]  # tied entries get equal normalized rank


def test_rank_normalize_single_element_is_half():
    assert rank_normalize(np.array([42.0])).tolist() == [0.5]


def test_rank_normalize_empty_raises():
    with pytest.raises(ValueError):
        rank_normalize(np.array([]))


def test_rank_normalize_is_scale_invariant():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(rank_normalize(x), rank_normalize(1000.0 * x - 500.0))


def test_zscore_normalize_mean_std():
    out = zscore_normalize(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert abs(out.mean()) < 1e-12
    assert abs(out.std() - 1.0) < 1e-12


def test_zscore_constant_is_zero():
    assert np.allclose(zscore_normalize(np.array([7.0, 7.0, 7.0])), 0.0)


def test_normalize_dispatch_and_unknown_method():
    assert np.allclose(
        normalize(np.array([1.0, 2.0]), "rank"), rank_normalize(np.array([1.0, 2.0]))
    )
    with pytest.raises(ValueError):
        normalize(np.array([1.0]), "nope")
