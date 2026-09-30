"""
Tests for the wrapper of the surmise emulators (gpbayestools/emulator_band.py).

Run with ``python -m pytest tests/test_emulator_band.py``.
"""

import pickle

import numpy as np
import pytest
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
    x = np.arange(emu.n_obs).reshape(-1, 1)
    var = emu.emu_.predict(x=x, theta=test_points).var().T
    if method == "PCSK":
        # the noise of the simulations, which surmise does not include
        var = var + np.mean(emu.model_data_err**2, axis=0)
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
    for c_exp, c_log, m in zip(cov_exp, cov_log, mean_exp, strict=True):
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

    # training the same object again gives the same emulator
    emu = EmulatorBAND(training_file, param_file, seed=3)
    emu.train_emulator_auto_mask()
    first = emu.predict(test_points)[0]
    emu.train_emulator_auto_mask()
    np.testing.assert_array_equal(emu.predict(test_points)[0], first)


def test_validation(training_file, param_file, test_points):
    emu = EmulatorBAND(training_file, param_file, seed=1)
    emu.train_emulator_auto_mask()
    before = emu.predict(test_points)
    pred, pred_err, data, data_err = emu.test_emulator_errors(10)
    assert pred.shape == data.shape == (10, N_OBS)
    assert np.abs(pred / data - 1).mean() < 0.05
    after = emu.predict(test_points)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)

    # training again gives the same emulator as a new one with the same seed
    emu.train_emulator_auto_mask()
    ref = EmulatorBAND(training_file, param_file, seed=1)
    ref.train_emulator_auto_mask()
    np.testing.assert_array_equal(
        emu.predict(test_points)[0], ref.predict(test_points)[0]
    )


def test_predict_single_point(emulator, test_points):
    mean, cov = emulator.predict(test_points)
    for x in (test_points[0], list(test_points[0])):
        mean_1, cov_1 = emulator.predict(x)
        np.testing.assert_allclose(mean_1, mean[:1])
        # the surmise variances of the noise-free test data are ill-conditioned
        # and depend at the percent level on the other predicted points
        np.testing.assert_allclose(cov_1, cov[:1], atol=0.02 * np.abs(cov).max())


def test_training_keeps_warning_filters(training_file, param_file):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="test filter")
        filters = list(warnings.filters)
        EmulatorBAND(training_file, param_file, seed=1).train_emulator_auto_mask()
        assert warnings.filters == filters


@pytest.fixture(scope="module")
def missing_file(tmp_path_factory, design):
    """Training data with missing observables in 10 points, one point without
    any finite observable and one with an infinite error."""
    values = true_model(design)
    errors = 0.01 * values
    rng = np.random.default_rng(3)
    for i in range(10):
        values[i, rng.choice(N_OBS, 2, replace=False)] = np.nan
    values[10, :] = np.nan
    errors[11, 0] = np.inf
    data = {
        str(i): {"parameter": design[i], "obs": np.vstack([values[i], errors[i]])}
        for i in range(len(design))
    }
    path = tmp_path_factory.mktemp("missing") / "missing.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


@pytest.mark.parametrize("method", ["PCGPwM", "PCGPwImpute"])
def test_missing_observables(missing_file, param_file, test_points, design, method):
    emu = EmulatorBAND(missing_file, param_file, method=method, seed=1)
    # only the point without any finite observable is discarded
    assert emu.n_ev == len(design) - 1
    missing = np.isnan(emu.model_data)
    assert missing.sum() == 10 * 2 + 1
    np.testing.assert_array_equal(missing, np.isnan(emu.model_data_err))
    emu.train_emulator_auto_mask()
    mean, cov = emu.predict(test_points)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))
    assert np.abs(mean / true_model(test_points) - 1).mean() < 0.05
    # the training is reproducible
    emu2 = EmulatorBAND(missing_file, param_file, method=method, seed=1)
    emu2.train_emulator_auto_mask()
    np.testing.assert_array_equal(mean, emu2.predict(test_points)[0])
    # the validation returns the missing training data as NaN
    pred, pred_err, data, _ = emu.test_emulator_errors(20, random_points=True, seed=1)
    assert np.all(np.isfinite(pred)) and np.all(np.isfinite(pred_err))
    assert np.isnan(data).sum() > 0
    # an observable missing in all training points
    mask = np.zeros(emu.n_ev, dtype=bool)
    mask[:2] = True
    emu.model_data[:2, 0] = np.nan
    with pytest.raises(ValueError, match="missing in all"):
        emu.train_emulator(mask)


def test_missing_observables_discarded_by_other_methods(
    missing_file, param_file, design
):
    emu = EmulatorBAND(missing_file, param_file, method="PCGP")
    assert emu.n_ev == len(design) - 12
    assert np.all(np.isfinite(emu.model_data))


def test_unknown_method(training_file, param_file):
    with pytest.raises(ValueError):
        EmulatorBAND(training_file, param_file, method="GP")
