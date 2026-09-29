"""
Tests for the scikit-learn emulator (gpbayestools/emulator.py) and the functionality
of the emulator base class (gpbayestools/emulator_base.py): loading and filtering the
training data, and the validation functions.

Run with ``python -m pytest tests/test_emulator.py``.
"""

import pickle
import warnings

import numpy as np
import pytest

from conftest import (N_OBS, true_model, write_param_file,
                      write_training_data)
from gpbayestools import parse_model_parameter_file
from gpbayestools.emulator import Emulator

warnings.filterwarnings("ignore", module="sklearn")


@pytest.fixture(scope="module")
def emulator(training_file, param_file):
    emu = Emulator(training_file, param_file, npc=4)
    emu.trainEmulatorAutoMask()
    return emu


# ── Prediction ───────────────────────────────────────────────────────
def test_prediction_accuracy_and_shapes(emulator, test_points):
    mean, cov = emulator.predict(test_points)
    assert mean.shape == (len(test_points), N_OBS)
    assert cov.shape == (len(test_points), N_OBS, N_OBS)
    rel_err = np.abs(mean / true_model(test_points) - 1)
    assert rel_err.mean() < 0.02
    for c in cov:
        np.testing.assert_allclose(c, c.T, atol=1e-12 * np.abs(c).max())
        assert np.all(np.diag(c) > 0)
    np.testing.assert_array_equal(
        emulator.predict(test_points, return_cov=False), mean)


def test_no_pca_covariance_in_observable_units(training_file, param_file,
                                               test_points):
    # the observables differ by a factor 2e4 in scale, the predicted
    # standard deviations must scale accordingly with and without PCA
    std = {}
    for no_pca in (False, True):
        emu = Emulator(training_file, param_file, npc=4,
                       perform_no_PCA=no_pca)
        emu.trainEmulatorAutoMask()
        _, cov = emu.predict(test_points)
        std[no_pca] = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    ratio = std[True] / std[False]
    assert np.all((ratio > 0.1) & (ratio < 10))


@pytest.mark.parametrize("npc, expected", [(2, 2), (0.99, None), (50, N_OBS)])
def test_npc(training_file, param_file, npc, expected):
    emu = Emulator(training_file, param_file, npc=npc)
    emu.trainEmulatorAutoMask()
    if expected is None:
        # smallest number of PCs explaining more than 99% of the variance
        evr = np.cumsum(emu.pca.explained_variance_ratio_)
        expected = np.searchsorted(evr, 0.99, side='right') + 1
    assert emu.npc == expected


@pytest.mark.parametrize("npc", [0, 1.0, -0.5, '3'])
def test_invalid_npc(training_file, param_file, npc):
    with pytest.raises((ValueError, TypeError)):
        Emulator(training_file, param_file, npc=npc)


def test_log_transformation(training_file, param_file, test_points):
    emu_log = Emulator(training_file, param_file, npc=4, logTrafo=True)
    emu_log.trainEmulatorAutoMask()
    mean_log, cov_log = emu_log.predict(test_points)
    # predictions in log space by default
    assert np.abs(np.exp(mean_log) / true_model(test_points) - 1).mean() < 0.02

    emu_exp = Emulator(training_file, param_file, npc=4, logTrafo=True,
                       exp_and_cov_diagonal=True)
    emu_exp.trainEmulatorAutoMask()
    mean_exp, cov_exp = emu_exp.predict(test_points)
    np.testing.assert_allclose(mean_exp, np.exp(mean_log))
    for c_exp, c_log, m in zip(cov_exp, cov_log, mean_exp):
        np.testing.assert_allclose(np.diag(c_exp), np.diag(c_log) * m**2)
        np.testing.assert_array_equal(c_exp, np.diag(np.diag(c_exp)))

    with pytest.raises(ValueError):
        Emulator(training_file, param_file, exp_and_cov_diagonal=True)


def test_sample_y(emulator, test_points):
    samples = emulator.sample_y(test_points, n_samples=4000, random_state=3)
    assert samples.shape == (len(test_points), 4000, N_OBS)
    np.testing.assert_array_equal(
        samples, emulator.sample_y(test_points, n_samples=4000, random_state=3))
    mean, cov = emulator.predict(test_points)
    std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    # sample mean within 5 standard errors of the predicted mean
    assert np.all(np.abs(samples.mean(axis=1) - mean) < 5 * std / np.sqrt(4000))
    # sample variance within 15% of the predicted variance (the prediction
    # contains a small additional term for numerical stability)
    np.testing.assert_allclose(samples.var(axis=1), std**2, rtol=0.15)


def test_output_pca_does_not_change_emulator(emulator, test_points):
    before = emulator.predict(test_points)
    design_points, Z = emulator.outputPCAvsParam()
    assert Z.shape == (emulator.npc, emulator.nev)
    after = emulator.predict(test_points)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_unknown_kernel(training_file, param_file):
    emu = Emulator(training_file, param_file, npc=2)
    with pytest.raises(ValueError):
        emu.trainEmulator(np.ones(emu.nev, dtype=bool), kernel_type='rbf')


def test_seed(training_file, param_file, test_points):
    def predict(seed):
        emu = Emulator(training_file, param_file, npc=3, nrestarts=2,
                       seed=seed)
        emu.trainEmulatorAutoMask()
        return emu.predict(test_points)[0]
    np.testing.assert_array_equal(predict(1), predict(1))


# ── Validation (base class) ──────────────────────────────────────────
def test_validation(emulator, test_points):
    before = emulator.predict(test_points)
    pred, pred_err, data, data_err = emulator.testEmulatorErrors(
        number_test_points=10)
    for arr in (pred, pred_err, data, data_err):
        assert arr.shape == (10, N_OBS)
    np.testing.assert_array_equal(data, emulator.model_data[-10:])
    assert np.abs(pred / data - 1).mean() < 0.05

    train = emulator.testEmulatorErrorsWithTrainingPoints(number_test_points=10)
    assert train[0].shape == (emulator.nev - 10, N_OBS)
    assert np.abs(train[0] / train[2] - 1).mean() < 0.05

    # the validation does not change the trained emulator
    after = emulator.predict(test_points)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_validation_random_points(emulator):
    train_mask, test_mask = emulator._validation_masks(10, True, 5)
    assert test_mask.sum() == 10 and np.all(train_mask == ~test_mask)
    np.testing.assert_array_equal(
        test_mask, emulator._validation_masks(10, True, 5)[1])
    assert not np.array_equal(test_mask,
                              emulator._validation_masks(10, True, 6)[1])
    data = emulator.testEmulatorErrors(10, random_points=True, seed=5)[2]
    np.testing.assert_array_equal(data, emulator.model_data[test_mask])
    with pytest.raises(ValueError):
        emulator.testEmulatorErrors(emulator.nev)


def test_validation_untrained_emulator_stays_untrained(training_file, param_file):
    emu = Emulator(training_file, param_file, npc=3)
    attributes = set(emu.__dict__)
    emu.testEmulatorErrors(5)
    assert set(emu.__dict__) == attributes


# ── Training data (base class) ───────────────────────────────────────
@pytest.fixture
def modified_data(tmp_path, design):
    """Write training data with modified values, returns the file path."""
    def write(modify):
        values = true_model(design)
        errors = 0.01 * values
        modify(values, errors)
        data = {str(i): {'parameter': design[i],
                         'obs': np.vstack([values[i], errors[i]])}
                for i in range(len(design))}
        path = tmp_path / "modified.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return str(path)
    return write


def test_non_finite_points_are_discarded(modified_data, param_file, design):
    def modify(values, errors):
        values[3, 2] = np.nan
        values[5, 1] = np.inf
    emu = Emulator(modified_data(modify), param_file)
    assert emu.nev == len(design) - 2
    assert np.all(np.isfinite(emu.model_data))


def test_relative_error_filter(modified_data, param_file):
    def modify(values, errors):
        errors[9, 3] = 0.5 * values[9, 3]    # 50% relative error
        values[4, 1] = 0.0                   # exactly zero with an error
        errors[4, 1] = 0.5
    path = modified_data(modify)
    # no filter by default
    assert Emulator(path, param_file).nev == 60
    # only the point with the large relative error is discarded, the zero
    # observable has no relative error
    emu = Emulator(path, param_file, max_rel_uncertainty_data=0.1)
    assert emu.nev == 59
    # log transformation requires positive observables
    with pytest.raises(ValueError):
        Emulator(path, param_file, logTrafo=True)


def test_negative_values_with_log_trafo(modified_data, param_file):
    def modify(values, errors):
        values[2, 0] *= -1
    path = modified_data(modify)
    assert Emulator(path, param_file).nev == 60
    with pytest.raises(ValueError):
        Emulator(path, param_file, logTrafo=True)


def test_all_points_discarded(modified_data, param_file):
    def modify(values, errors):
        errors[:] = values
    with pytest.raises(ValueError):
        Emulator(modified_data(modify), param_file,
                 max_rel_uncertainty_data=0.1)


def test_parameter_count_mismatch(training_file, tmp_path):
    with pytest.raises(ValueError):
        Emulator(training_file, write_param_file(tmp_path / "p2.txt", 2))


def test_parse_model_parameter_file(tmp_path):
    path = tmp_path / "par.txt"
    path.write_text("# comment\n\n   \nalpha : a, 0.0, 1.0   # trailing\n"
                    "beta: $\\beta$, -1, 2.5\n")
    assert parse_model_parameter_file(path) == {
        'alpha': ['a', 0.0, 1.0], 'beta': ['$\\beta$', -1.0, 2.5]}


def test_load_emulator_saved_with_old_package_name(emulator, test_points,
                                                    tmp_path):
    # emulators saved with versions < 3.0.0 refer to the module src.emulator
    import sys
    import dill
    import gpbayestools
    import gpbayestools.emulator
    from gpbayestools import load_emulator
    path = tmp_path / "old_emulator.dill"
    cls = gpbayestools.emulator.Emulator
    sys.modules['src'] = gpbayestools
    sys.modules['src.emulator'] = gpbayestools.emulator
    cls.__module__ = 'src.emulator'
    try:
        with open(path, 'wb') as f:
            dill.dump(emulator, f)
    finally:
        cls.__module__ = 'gpbayestools.emulator'
        del sys.modules['src'], sys.modules['src.emulator']
    with pytest.raises(ModuleNotFoundError):
        with open(path, 'rb') as f:
            dill.load(f)
    loaded = load_emulator(path)
    assert type(loaded) is cls
    for a, b in zip(loaded.predict(test_points), emulator.predict(test_points)):
        np.testing.assert_array_equal(a, b)
