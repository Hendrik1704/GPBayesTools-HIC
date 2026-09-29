"""
Tests of the include_noise option of the emulators: by default, predict()
returns the uncertainty of the emulated model function, with
include_noise=True the noise fitted by the GPs is included. The validation
functions compare with noisy simulations and include the noise.

Run with ``python -m pytest tests/test_noise.py``.
"""

import warnings

import numpy as np
import pytest
from conftest import latin_hypercube, true_model, write_training_data

from gpbayestools.emulator_hetgp import EmulatorHetGP
from gpbayestools.emulator_sklearn import EmulatorSklearn
from gpbayestools.emulator_sparse_gp import EmulatorSparseGP

warnings.filterwarnings("ignore", module="sklearn")


def band_emulator(*args, **kwargs):
    from gpbayestools.emulator_band import EmulatorBAND

    return EmulatorBAND(*args, seed=1, **kwargs)


EMULATORS = {
    "EmulatorSklearn": (lambda *a: EmulatorSklearn(*a, npc=4), {}),
    "PCGP": (lambda *a: band_emulator(*a, method="PCGP"), {}),
    "PCSK": (lambda *a: band_emulator(*a, method="PCSK"), {}),
    "hetGPy": (lambda *a: EmulatorHetGP(*a), {}),
    "SparseGP": (
        lambda *a: EmulatorSparseGP(*a, npc=0.999, n_inducing=30),
        {"steps": 1000, "verbose": False},
    ),
}


@pytest.fixture(scope="module")
def noisy_training_file(data_dir):
    """Training data with 3% statistical noise."""
    rng = np.random.default_rng(1)
    design = latin_hypercube(80, 3, rng)
    values = true_model(design)
    noisy = values * (1 + 0.03 * rng.standard_normal(values.shape))
    return write_training_data(data_dir / "noisy.pkl", design, noisy, rel_err=0.03)


@pytest.fixture(scope="module", params=list(EMULATORS))
def trained(request, noisy_training_file, param_file):
    make, train_kwargs = EMULATORS[request.param]
    emu = make(noisy_training_file, param_file)
    emu.train_emulator_auto_mask(**train_kwargs)
    return request.param, emu, train_kwargs


def test_noise_covariance(trained, test_points):
    name, emu, _ = trained
    mean_latent, cov_latent = emu.predict(test_points)
    mean_noise, cov_noise = emu.predict(test_points, include_noise=True)
    np.testing.assert_array_equal(mean_latent, mean_noise)
    noise = cov_noise - cov_latent
    for n, c in zip(noise, cov_noise, strict=True):
        # the noise covariance is positive semi-definite
        assert np.linalg.eigvalsh(n).min() > -1e-8 * np.abs(c).max()
    if name in ("EmulatorSklearn", "hetGPy", "PCGP"):
        # these emulators fit a noise term to the noisy training data
        assert np.all(np.diagonal(noise, axis1=1, axis2=2) > 0)


def test_emulator_noise_is_the_white_kernel(trained, test_points):
    name, emu, _ = trained
    if name != "EmulatorSklearn":
        pytest.skip("only for EmulatorSklearn")
    _, cov_latent = emu.predict(test_points)
    _, cov_noise = emu.predict(test_points, include_noise=True)
    noise_pc = np.array([emu._gp_noise(gp.kernel_) for gp in emu.gps_])
    A = emu._trans_matrix[: emu.npc_]
    # the WhiteKernel noise of the GPs and the noise part of the truncation
    # covariance
    noise_trunc = np.diag(emu._cov_trunc - emu._cov_trunc_signal)
    np.testing.assert_allclose(
        np.diagonal(cov_noise - cov_latent, axis1=1, axis2=2),
        np.tile(noise_pc @ A**2 + noise_trunc, (len(test_points), 1)),
        rtol=1e-8,
    )


def test_truncation_signal():
    from gpbayestools.emulator_base import truncation_signal

    rng = np.random.default_rng(0)
    V = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    # truncation covariance of 3 discarded directions
    trunc = (V[:, :3] * [0.5, 0.2, 0.05]) @ V[:, :3].T
    # isotropic noise: the signal is the truncation minus the noise in each
    # direction, down to zero
    signal = truncation_signal(trunc, 0.1 * np.eye(6))
    np.testing.assert_allclose(
        signal, (V[:, :3] * [0.4, 0.1, 0.0]) @ V[:, :3].T, atol=1e-12
    )
    # for any noise, 0 <= signal <= truncation
    noise = rng.normal(size=(6, 6))
    noise = noise @ noise.T
    signal = truncation_signal(trunc, noise)
    assert np.linalg.eigvalsh(signal).min() > -1e-12
    assert np.linalg.eigvalsh(trunc - signal).min() > -1e-12
    # without noise, the truncation covariance is unchanged
    np.testing.assert_allclose(
        truncation_signal(trunc, np.zeros((6, 6))), trunc, atol=1e-12
    )


def test_validation_includes_noise(trained):
    name, emu, train_kwargs = trained
    pred, pred_err, data, _ = emu.test_emulator_errors(10, **train_kwargs)
    # the same emulator trained without the test points
    mask = np.ones(emu.nev, dtype=bool)
    mask[-10:] = False
    emu_copy = type(emu).__new__(type(emu))
    emu_copy.__dict__.update(emu.__dict__)
    emu_copy.train_emulator(mask, **train_kwargs)
    _, cov = emu_copy.predict(emu.design_points[~mask], include_noise=True)
    np.testing.assert_allclose(pred_err, np.sqrt(np.diagonal(cov, axis1=1, axis2=2)))


def test_emulator_calibration(noisy_training_file, param_file):
    # with noisy training data, the predicted uncertainty of the model
    # function matches the actual errors (it was overestimated with the
    # alpha = 0.1 of versions < 3.0.0)
    emu = EmulatorSklearn(noisy_training_file, param_file, npc=6, n_restarts=2, seed=1)
    emu.train_emulator_auto_mask()
    X = np.random.default_rng(3).uniform(0.2, 0.8, size=(200, 3))
    mean, cov = emu.predict(X)
    z = (mean - true_model(X)) / np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    assert 0.5 < np.sqrt(np.mean(z**2)) < 2


@pytest.mark.parametrize("include_noise", [False, True])
def test_sample_y(trained, test_points, include_noise):
    name, emu, _ = trained
    samples = emu.sample_y(
        test_points, n_samples=4000, random_state=1, include_noise=include_noise
    )
    assert samples.shape == (len(test_points), 4000, emu.nobs)
    np.testing.assert_array_equal(
        samples,
        emu.sample_y(
            test_points, n_samples=4000, random_state=1, include_noise=include_noise
        ),
    )
    mean, cov = emu.predict(test_points, include_noise=include_noise)
    std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    assert np.all(np.abs(samples.mean(axis=1) - mean) < 5 * std / np.sqrt(4000))
    np.testing.assert_allclose(samples.var(axis=1), std**2, rtol=0.15)


def test_sample_y_log_normal(noisy_training_file, param_file, test_points):
    emu = EmulatorHetGP(
        noisy_training_file, param_file, log_trafo=True, exp_and_cov_diagonal=True
    )
    emu.train_emulator_auto_mask()
    samples = emu.sample_y(test_points, n_samples=4000, random_state=1)
    assert np.all(samples > 0)
    # the median of the log-normal samples is exp(mean in log space), which
    # predict() returns with exp_and_cov_diagonal
    mean = emu.predict(test_points)[0]
    np.testing.assert_allclose(np.median(samples, axis=1), mean, rtol=0.01)
    assert emu.exp_and_cov_diagonal


def test_legacy_attribute_names(trained, test_points):
    # emulators saved with versions < 3.0.0 have other attribute names, which
    # are renamed when they are loaded; only the emulators of the published
    # analyses (EmulatorSklearn and EmulatorBAND) can be loaded
    name, emu, _ = trained
    if name not in ("EmulatorSklearn", "PCGP", "PCSK"):
        pytest.skip("old versions can only be loaded for EmulatorSklearn and BAND")
    state = emu.__getstate__() if hasattr(type(emu), "__getstate__") else None
    state = dict(state if state is not None else emu.__dict__)
    renames = [
        ("logTrafo_", "log_trafo"),
        ("max_rel_uncertainty_data_", "max_rel_uncertainty_data"),
        ("exp_and_cov_diagonal_", "exp_and_cov_diagonal"),
    ] + type(emu)._legacy_attributes
    for old, new in reversed(renames):
        if new in state:
            state[old] = state.pop(new)
    legacy = type(emu).__new__(type(emu))
    legacy.__setstate__(state)
    assert "logTrafo_" not in legacy.__dict__
    for a, b in zip(legacy.predict(test_points), emu.predict(test_points), strict=True):
        np.testing.assert_array_equal(a, b)
