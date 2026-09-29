"""
Tests for the scikit-learn emulator (gpbayestools/emulator_sklearn.py) and the
functionality of the emulator base class (gpbayestools/emulator_base.py): loading and
filtering the training data, and the validation functions.

Run with ``python -m pytest tests/test_emulator_sklearn.py``.
"""

import pickle
import warnings

import numpy as np
import pytest
from conftest import N_OBS, OBS_SCALE, true_model, write_param_file

from gpbayestools import parse_model_parameter_file
from gpbayestools.emulator_sklearn import EmulatorSklearn

warnings.filterwarnings("ignore", module="sklearn")


@pytest.fixture(scope="module")
def emulator(training_file, param_file):
    emu = EmulatorSklearn(training_file, param_file, npc=4)
    emu.train_emulator_auto_mask()
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
    np.testing.assert_array_equal(emulator.predict(test_points, return_cov=False), mean)


def test_predict_single_point(emulator, test_points):
    # a 1D array or a list is a single parameter point
    mean, cov = emulator.predict(test_points)
    for x in (test_points[0], list(test_points[0])):
        mean_1, cov_1 = emulator.predict(x)
        np.testing.assert_allclose(mean_1, mean[:1])
        np.testing.assert_allclose(cov_1, cov[:1], atol=1e-6 * np.abs(cov).max())


def test_no_pca_covariance_in_observable_units(training_file, param_file, test_points):
    # the observables differ by a factor 2e4 in scale, the predicted
    # standard deviations must scale accordingly with and without PCA, i.e.
    # relative to the scale of the observables they are of similar size
    for no_pca in (False, True):
        emu = EmulatorSklearn(training_file, param_file, npc=4, perform_no_pca=no_pca)
        emu.train_emulator_auto_mask()
        # without PCA, one GP is trained per observable
        assert emu.npc_ == len(emu.gps_) == (N_OBS if no_pca else 4)
        _, cov = emu.predict(test_points)
        rel_std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2)).mean(axis=0) / OBS_SCALE
        assert rel_std.max() / rel_std.min() < 10


@pytest.mark.parametrize("npc, expected", [(2, 2), (0.99, None), (50, N_OBS)])
def test_npc(training_file, param_file, npc, expected):
    emu = EmulatorSklearn(training_file, param_file, npc=npc)
    emu.train_emulator_auto_mask()
    if expected is None:
        # smallest number of PCs explaining more than 99% of the variance
        evr = np.cumsum(emu.pca_.explained_variance_ratio_)
        expected = np.searchsorted(evr, 0.99, side="right") + 1
    assert emu.npc_ == expected


@pytest.mark.parametrize("npc", [0, 1.0, -0.5, "3"])
def test_invalid_npc(training_file, param_file, npc):
    with pytest.raises((ValueError, TypeError)):
        EmulatorSklearn(training_file, param_file, npc=npc)


def test_log_transformation(training_file, param_file, test_points):
    emu_log = EmulatorSklearn(training_file, param_file, npc=4, log_trafo=True)
    emu_log.train_emulator_auto_mask()
    mean_log, cov_log = emu_log.predict(test_points)
    # predictions in log space by default
    assert np.abs(np.exp(mean_log) / true_model(test_points) - 1).mean() < 0.02

    emu_exp = EmulatorSklearn(
        training_file, param_file, npc=4, log_trafo=True, exp_and_cov_diagonal=True
    )
    emu_exp.train_emulator_auto_mask()
    mean_exp, cov_exp = emu_exp.predict(test_points)
    np.testing.assert_allclose(mean_exp, np.exp(mean_log))
    for c_exp, c_log, m in zip(cov_exp, cov_log, mean_exp, strict=True):
        np.testing.assert_allclose(np.diag(c_exp), np.diag(c_log) * m**2)
        np.testing.assert_array_equal(c_exp, np.diag(np.diag(c_exp)))

    with pytest.raises(ValueError):
        EmulatorSklearn(training_file, param_file, exp_and_cov_diagonal=True)


def test_sample_y(emulator, test_points):
    samples = emulator.sample_y(test_points, n_samples=4000, random_state=3)
    assert samples.shape == (len(test_points), 4000, N_OBS)
    np.testing.assert_array_equal(
        samples, emulator.sample_y(test_points, n_samples=4000, random_state=3)
    )
    mean, cov = emulator.predict(test_points)
    std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    # sample mean within 5 standard errors of the predicted mean
    assert np.all(np.abs(samples.mean(axis=1) - mean) < 5 * std / np.sqrt(4000))
    # sample variance within 15% of the predicted variance (the prediction
    # contains a small additional term for numerical stability)
    np.testing.assert_allclose(samples.var(axis=1), std**2, rtol=0.15)


def test_output_pca_does_not_change_emulator(emulator, test_points):
    before = emulator.predict(test_points)
    design_points, Z = emulator.output_pca_vs_param()
    assert Z.shape == (emulator.npc_, emulator.nev)
    after = emulator.predict(test_points)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)


def test_unknown_kernel(emulator, test_points):
    before = emulator.predict(test_points)
    with pytest.raises(ValueError):
        emulator.train_emulator(np.ones(emulator.nev, dtype=bool), kernel_type="rbf")
    # the trained emulator is not modified
    for a, b in zip(before, emulator.predict(test_points), strict=True):
        np.testing.assert_array_equal(a, b)


def test_seed(training_file, param_file, test_points):
    def predict(seed):
        emu = EmulatorSklearn(training_file, param_file, npc=3, n_restarts=2, seed=seed)
        emu.train_emulator_auto_mask()
        return emu.predict(test_points)[0]

    np.testing.assert_array_equal(predict(1), predict(1))


# ── Validation (base class) ──────────────────────────────────────────
def test_validation(emulator, test_points):
    before = emulator.predict(test_points)
    pred, pred_err, data, data_err = emulator.test_emulator_errors(n_test_points=10)
    for arr in (pred, pred_err, data, data_err):
        assert arr.shape == (10, N_OBS)
    np.testing.assert_array_equal(data, emulator.model_data[-10:])
    assert np.abs(pred / data - 1).mean() < 0.05

    train = emulator.test_emulator_errors_with_training_points(n_test_points=10)
    assert train[0].shape == (emulator.nev - 10, N_OBS)
    assert np.abs(train[0] / train[2] - 1).mean() < 0.05

    # the validation does not change the trained emulator
    after = emulator.predict(test_points)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)


def test_validation_random_points(emulator):
    train_mask, test_mask = emulator._validation_masks(10, True, 5, 1)
    assert test_mask.sum() == 10 and np.all(train_mask == ~test_mask)
    np.testing.assert_array_equal(
        test_mask, emulator._validation_masks(10, True, 5, 1)[1]
    )
    assert not np.array_equal(test_mask, emulator._validation_masks(10, True, 6, 1)[1])
    data = emulator.test_emulator_errors(10, random_points=True, seed=5)[2]
    np.testing.assert_array_equal(data, emulator.model_data[test_mask])
    # at least one test point and two training points are required
    for n_test_points in (0, emulator.nev - 1):
        with pytest.raises(ValueError):
            emulator.test_emulator_errors(n_test_points)
    with pytest.raises(ValueError):
        emulator.test_emulator_errors_with_training_points(emulator.nev - 1)


def test_validation_untrained_emulator_stays_untrained(training_file, param_file):
    emu = EmulatorSklearn(training_file, param_file, npc=3)
    attributes = set(emu.__dict__)
    emu.test_emulator_errors(5)
    assert set(emu.__dict__) == attributes


# ── Training data (base class) ───────────────────────────────────────
@pytest.fixture
def modified_data(tmp_path, design):
    """Write training data with modified values, returns the file path."""

    def write(modify):
        values = true_model(design)
        errors = 0.01 * values
        modify(values, errors)
        data = {
            str(i): {"parameter": design[i], "obs": np.vstack([values[i], errors[i]])}
            for i in range(len(design))
        }
        path = tmp_path / "modified.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return str(path)

    return write


def test_non_finite_points_are_discarded(modified_data, param_file, design):
    def modify(values, errors):
        values[3, 2] = np.nan
        values[5, 1] = np.inf

    emu = EmulatorSklearn(modified_data(modify), param_file)
    assert emu.nev == len(design) - 2
    assert np.all(np.isfinite(emu.model_data))


def test_relative_error_filter(modified_data, param_file):
    def modify(values, errors):
        errors[9, 3] = 0.5 * values[9, 3]  # 50% relative error
        values[4, 1] = 0.0  # exactly zero with an error
        errors[4, 1] = 0.5

    path = modified_data(modify)
    # no filter by default
    assert EmulatorSklearn(path, param_file).nev == 60
    # only the point with the large relative error is discarded, the zero
    # observable has no relative error
    emu = EmulatorSklearn(path, param_file, max_rel_uncertainty_data=0.1)
    assert emu.nev == 59
    # log transformation requires positive observables
    with pytest.raises(ValueError):
        EmulatorSklearn(path, param_file, log_trafo=True)


def test_negative_values_with_log_trafo(modified_data, param_file):
    def modify(values, errors):
        values[2, 0] *= -1

    path = modified_data(modify)
    assert EmulatorSklearn(path, param_file).nev == 60
    with pytest.raises(ValueError):
        EmulatorSklearn(path, param_file, log_trafo=True)

    # a point that is discarded because of its errors does not raise
    def modify_with_errors(values, errors):
        values[2, 0] *= -1
        errors[2, 0] = 10 * abs(values[2, 0])

    path = modified_data(modify_with_errors)
    emu = EmulatorSklearn(
        path, param_file, log_trafo=True, max_rel_uncertainty_data=0.1
    )
    assert emu.nev == 59


def test_all_points_discarded(modified_data, param_file):
    def modify(values, errors):
        errors[:] = values

    with pytest.raises(ValueError):
        EmulatorSklearn(modified_data(modify), param_file, max_rel_uncertainty_data=0.1)


def test_parameter_count_mismatch(training_file, tmp_path):
    with pytest.raises(ValueError):
        EmulatorSklearn(training_file, write_param_file(tmp_path / "p2.txt", 2))


def test_parse_model_parameter_file(tmp_path):
    path = tmp_path / "par.txt"
    path.write_text(
        "# comment\n\n   \nalpha : a, 0.0, 1.0   # trailing\nbeta: $\\beta$, -1, 2.5\n"
    )
    assert parse_model_parameter_file(path) == {
        "alpha": ["a", 0.0, 1.0],
        "beta": ["$\\beta$", -1.0, 2.5],
    }


@pytest.mark.parametrize(
    "line",
    [
        "alpha a, 0, 1",
        "alpha: a, 0",
        "alpha: a, zero, 1",
        "alpha: a, 1, 1",
        "alpha: a, 2, 1",
        "beta: b, 0, 1",
    ],
)
def test_invalid_parameter_file(tmp_path, line):
    path = tmp_path / "par.txt"
    path.write_text(f"beta: b, 0, 1\n{line}\n")
    with pytest.raises(ValueError, match="line 2"):
        parse_model_parameter_file(path)


def test_load_emulator_saved_with_old_package_name(emulator, test_points, tmp_path):
    # emulators saved with versions < 3.0.0 refer to the class
    # src.emulator.Emulator; with pickle protocol 2 the reference is stored
    # as plain text and can be replaced to create such a file
    import dill

    from gpbayestools import load_emulator

    data = dill.dumps(emulator, protocol=2)
    new_ref = b"cgpbayestools.emulator_sklearn\nEmulatorSklearn\n"
    assert new_ref in data
    path = tmp_path / "old_emulator.dill"
    path.write_bytes(data.replace(new_ref, b"csrc.emulator\nEmulator\n"))
    with pytest.raises(ModuleNotFoundError):
        with open(path, "rb") as f:
            dill.load(f)
    loaded = load_emulator(path)
    assert type(loaded) is type(emulator)
    for a, b in zip(
        loaded.predict(test_points), emulator.predict(test_points), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_old_emulator_with_parameter_pca_raises():
    # the parameterTrafoPCA option of versions < 3.0.0 was removed
    state = {"logTrafo_": False, "parameterTrafoPCA_": True}
    with pytest.raises(ValueError, match="parameterTrafoPCA"):
        EmulatorSklearn._migrate_legacy_state(state)
    state["parameterTrafoPCA_"] = False
    assert EmulatorSklearn._migrate_legacy_state(state)["log_trafo"] is False


def test_non_finite_errors_are_set_to_zero(modified_data, param_file):
    def modify(values, errors):
        errors[3, 2] = np.nan
        errors[5, 1] = np.inf

    emu = EmulatorSklearn(modified_data(modify), param_file)
    assert emu.nev == 60
    assert emu.model_data_err[3, 2] == 0 and emu.model_data_err[5, 1] == 0
