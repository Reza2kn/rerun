"""Tensor/array helpers underpin every sensor logger, so pin their behaviour."""

from __future__ import annotations

import numpy as np

from genesis_rerun.sensors._arrays import as_numpy, select_env

from .conftest import TorchLikeTensor


def test_as_numpy_passes_through_plain_sequences() -> None:
    out = as_numpy([1.0, 2.0, 3.0])
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_as_numpy_detaches_and_moves_torch_like_tensors_to_cpu() -> None:
    out = as_numpy(TorchLikeTensor([[1, 2], [3, 4]]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [[1, 2], [3, 4]])


def test_select_env_picks_one_row_from_a_batched_array() -> None:
    batched = np.arange(6).reshape(2, 3)  # (n_envs=2, 3)
    np.testing.assert_array_equal(select_env(batched, 0), [0, 1, 2])
    np.testing.assert_array_equal(select_env(batched, 1), [3, 4, 5])


def test_select_env_leaves_unbatched_values_untouched() -> None:
    unbatched = np.array([7.0, 8.0, 9.0])
    np.testing.assert_array_equal(select_env(unbatched, 0), [7.0, 8.0, 9.0])


def test_select_env_falls_back_when_env_index_out_of_range() -> None:
    batched = np.arange(6).reshape(2, 3)
    # env_index 5 does not exist; helper returns the whole array rather than raising.
    np.testing.assert_array_equal(select_env(batched, 5), batched)
