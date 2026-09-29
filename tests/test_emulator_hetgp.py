"""
Synthetic end-to-end tests for the hetGPy emulator.

A Latin-Hypercube design in 3 parameters is evaluated with a known
analytical model (sum-of-sines + polynomial) that produces 20 observables
per design point, with small Gaussian noise to mimic statistical errors.
The emulator is trained on these data and tested for

  - the accuracy of the predictions compared to the true model,
  - identical predictions after saving and reloading the emulator,
  - the covariance of the discarded PCs in the predicted covariance,
  - the built-in validation (testEmulatorErrors).

Run with ``python -m pytest tests/test_emulator_hetgp.py``.
"""

import os
import pickle
import sys

import dill
import numpy as np
import pytest

# Resolve the project root (one level up from tests/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)
from gpbayestools.emulator_hetgp import EmulatorHetGP

# ── Configuration ────────────────────────────────────────────────────
N_DESIGN = 80  # number of training design points
N_OBS = 20  # number of observables
N_PARAMS = 3  # number of model parameters
REL_NOISE = 0.01  # relative statistical noise on each observable
SEED = 42
N_TEST = 10  # number of test points for the predictions
MAX_MEAN_REL_ERR = 0.05


# ── Analytical ground-truth model ────────────────────────────────────
def true_model(params, n_obs=N_OBS):
    """
    Map a parameter vector (alpha, beta, gamma) in [0,1]^3
    to *n_obs* observables using a smooth, nonlinear function that
    is easy to emulate.

    f_k(alpha, beta, gamma) =
        (1 + alpha) * sin(pi * k / n_obs * beta)
        + gamma^2 * cos(2*pi * k / n_obs)
        + 0.5 * alpha * beta * k / n_obs

    Parameters
    ----------
    params : array-like, shape (3,)
    n_obs  : int

    Returns
    -------
    values : ndarray, shape (n_obs,)
    """
    alpha, beta, gamma = params
    k = np.arange(n_obs, dtype=float)
    frac = k / n_obs
    values = (
        (1.0 + alpha) * np.sin(np.pi * frac * beta)
        + gamma**2 * np.cos(2.0 * np.pi * frac)
        + 0.5 * alpha * beta * frac
    )
    # Shift so all values are strictly positive (needed for logTrafo)
    values += 3.0
    return values


# ── 1.  Generate a Latin-Hypercube design ────────────────────────────
def latin_hypercube(n_samples, n_dim, rng):
    """Simple random LHD in [0, 1]^n_dim."""
    result = np.zeros((n_samples, n_dim))
    for d in range(n_dim):
        perm = rng.permutation(n_samples)
        result[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
    return result


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def data_files(tmp_path_factory):
    """Write the parameter file and synthetic training data."""
    tmp = tmp_path_factory.mktemp("hetgpy")
    par_file = tmp / "modelDesign_test_3par.txt"
    par_file.write_text(
        "alpha: alpha, 0.0, 1.0\nbeta: beta, 0.0, 1.0\ngamma: gamma, 0.0, 1.0\n"
    )

    rng = np.random.default_rng(SEED)
    design = latin_hypercube(N_DESIGN, N_PARAMS, rng)
    data_dict = {}
    for i in range(N_DESIGN):
        values = true_model(design[i])
        sigma = REL_NOISE * np.abs(values)
        noisy_values = values + sigma * rng.standard_normal(N_OBS)
        # obs has shape (2, n_obs): row 0 = values, row 1 = stat errors
        data_dict[str(i)] = {
            "parameter": design[i],
            "obs": np.vstack([noisy_values, sigma]),
        }
    training_file = tmp / "synthetic_training_data.pickle"
    with open(training_file, "wb") as f:
        pickle.dump(data_dict, f)
    return str(training_file), str(par_file)


def make_emulator(data_files):
    training_file, par_file = data_files
    return EmulatorHetGP(
        training_set_path=training_file,
        parameter_file=par_file,
        logTrafo=False,
        max_rel_uncertainty_data=0.5,
    )


@pytest.fixture(scope="module")
def emulator(data_files):
    emu = make_emulator(data_files)
    emu.trainEmulatorAutoMask()
    return emu


@pytest.fixture(scope="module")
def test_params():
    return np.random.default_rng(SEED + 1).uniform(size=(N_TEST, N_PARAMS))


# ── Tests ────────────────────────────────────────────────────────────
def test_prediction_accuracy(emulator, test_params):
    pred_mean, pred_cov = emulator.predict(test_params, return_cov=True)
    assert pred_mean.shape == (N_TEST, N_OBS)
    assert pred_cov.shape == (N_TEST, N_OBS, N_OBS)

    true_vals = np.array([true_model(p) for p in test_params])
    rel_err = np.abs(pred_mean - true_vals) / np.abs(true_vals)
    assert rel_err.mean() < MAX_MEAN_REL_ERR

    # predicted uncertainties must be positive and of a sensible size
    pred_std = np.sqrt(np.diagonal(pred_cov, axis1=1, axis2=2))
    assert np.all(pred_std > 0)
    rms_pull = np.sqrt(np.mean(((pred_mean - true_vals) / pred_std) ** 2))
    assert 0.1 < rms_pull < 10


def test_pickle_roundtrip_is_exact(emulator, test_params, tmp_path):
    emulator_file = tmp_path / "emulator_synthetic.pkl"
    with open(emulator_file, "wb") as f:
        dill.dump(emulator, f)
    with open(emulator_file, "rb") as f:
        emu_loaded = dill.load(f)

    mean, cov = emulator.predict(test_params)
    mean_loaded, cov_loaded = emu_loaded.predict(test_params)
    np.testing.assert_array_equal(mean, mean_loaded)
    np.testing.assert_array_equal(cov, cov_loaded)


def test_covariance_includes_truncation(emulator, test_params):
    assert emulator.npc < N_OBS
    # by default, the truncation covariance without the noise of the training
    # data is used, with include_noise=True the full truncation covariance
    for include_noise, trunc in (
        (False, emulator._cov_trunc_signal),
        (True, emulator._cov_trunc),
    ):
        _, pred_cov = emulator.predict(test_params, include_noise=include_noise)
        for cov in pred_cov:
            np.testing.assert_allclose(cov, cov.T)
            # the truncation covariance is positive semi-definite, so it can
            # only increase the predicted variances
            assert np.all(np.diag(cov) >= np.diag(trunc) - 1e-12)
            assert np.linalg.eigvalsh(cov).min() > -1e-10 * np.abs(cov).max()


def test_validation(data_files):
    emu = make_emulator(data_files)
    n_test = 5
    emu_pred, emu_pred_err, vali_data, vali_data_err = emu.testEmulatorErrors(
        number_test_points=n_test
    )
    for arr in (emu_pred, emu_pred_err, vali_data, vali_data_err):
        assert arr.shape == (n_test, N_OBS)
    val_rel_err = np.abs(emu_pred - vali_data) / np.abs(vali_data)
    assert val_rel_err.mean() < MAX_MEAN_REL_ERR


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
