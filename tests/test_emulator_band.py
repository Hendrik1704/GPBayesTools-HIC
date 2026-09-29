"""
Tests for the wrapper of the surmise emulators (gpbayestools/emulator_band.py).

The tests require surmise >= 1.0.0 and are skipped otherwise.
Run with ``python -m pytest tests/test_emulator_band.py``.
"""

import numpy as np
import pytest

surmise = pytest.importorskip("surmise")
if not hasattr(surmise, "set_RNG"):
    pytest.skip("requires surmise >= 1.0.0", allow_module_level=True)

from conftest import N_OBS, true_model
from gpbayestools.emulator_band import EmulatorBAND

METHODS = ["PCGP", "PCSK", "PCGPwImpute", "PCGPwM"]


@pytest.fixture(scope="module", params=METHODS)
def emulator(request, training_file, param_file):
    emu = EmulatorBAND(training_file, param_file, method=request.param, seed=1)
    emu.train_emulator_auto_mask()
    return emu


def test_prediction_accuracy_and_shapes(emulator, test_points):
    mean, cov = emulator.predict(test_points)
    assert mean.shape == (len(test_points), N_OBS)
    assert cov.shape == (len(test_points), N_OBS, N_OBS)
    assert np.abs(mean / true_model(test_points) - 1).mean() < 0.05
    for c in cov:
        np.testing.assert_allclose(c, c.T, atol=1e-12 * np.abs(c).max())
        assert np.all(np.diag(c) > 0)


@pytest.mark.parametrize("method", METHODS)
def test_covariance_contains_variance_of_discarded_pcs(
    training_file, param_file, test_points, method
):
    # with log_trafo, PCGP discards PCs; the diagonal of the covariance must
    # still be the full predictive variance of surmise
    emu = EmulatorBAND(training_file, param_file, method=method, seed=1, log_trafo=True)
    emu.train_emulator_auto_mask()
    _, cov = emu.predict(test_points, include_noise=True)
    x = np.arange(emu.nobs).reshape(-1, 1)
    var = emu.emu_.predict(x=x, theta=test_points).var().T
    np.testing.assert_allclose(np.diagonal(cov, axis1=1, axis2=2), var, rtol=1e-10)


def test_log_transformation(training_file, param_file, test_points):
    emu_log = EmulatorBAND(training_file, param_file, seed=1, log_trafo=True)
    emu_log.train_emulator_auto_mask()
    mean_log, cov_log = emu_log.predict(test_points)
    assert np.abs(np.exp(mean_log) / true_model(test_points) - 1).mean() < 0.05

    emu_exp = EmulatorBAND(
        training_file, param_file, seed=1, log_trafo=True, exp_and_cov_diagonal=True
    )
    emu_exp.train_emulator_auto_mask()
    mean_exp, cov_exp = emu_exp.predict(test_points)
    np.testing.assert_allclose(mean_exp, np.exp(mean_log))
    for c_exp, c_log, m in zip(cov_exp, cov_log, mean_exp):
        np.testing.assert_allclose(np.diag(c_exp), np.diag(c_log) * m**2)

    # a single 1D parameter vector gives the same as a 2D array with one row
    m1, c1 = emu_exp.predict(test_points[0])
    m2, c2 = emu_exp.predict(test_points[:1])
    assert m1.shape == (1, N_OBS) and c1.shape == (1, N_OBS, N_OBS)
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(c1, c2)


def test_seed(training_file, param_file, test_points):
    def predict(seed):
        emu = EmulatorBAND(training_file, param_file, seed=seed)
        emu.train_emulator_auto_mask()
        return emu.predict(test_points)[0]

    np.testing.assert_array_equal(predict(3), predict(3))


def test_validation(training_file, param_file, test_points):
    emu = EmulatorBAND(training_file, param_file, seed=1)
    emu.train_emulator_auto_mask()
    before = emu.predict(test_points)
    pred, pred_err, data, data_err = emu.test_emulator_errors(10)
    assert pred.shape == data.shape == (10, N_OBS)
    assert np.abs(pred / data - 1).mean() < 0.05
    after = emu.predict(test_points)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)

    # the validation restores the random state: training again gives the
    # same emulator as without the validation
    emu.train_emulator_auto_mask()
    ref = EmulatorBAND(training_file, param_file, seed=1)
    ref.train_emulator_auto_mask()
    ref.train_emulator_auto_mask()
    np.testing.assert_array_equal(
        emu.predict(test_points)[0], ref.predict(test_points)[0]
    )


def test_unknown_method(training_file, param_file):
    emu = EmulatorBAND(training_file, param_file, method="GP")
    with pytest.raises(ValueError):
        emu.train_emulator_auto_mask()
