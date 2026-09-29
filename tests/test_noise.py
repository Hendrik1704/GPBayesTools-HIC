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
from gpbayestools.emulator import Emulator
from gpbayestools.emulator_hetGPy import EmulatorHETGPy
from gpbayestools.emulator_sparseGP import EmulatorSparseGP

warnings.filterwarnings("ignore", module="sklearn")


def band_emulator(*args, **kwargs):
    surmise = pytest.importorskip("surmise")
    if not hasattr(surmise, "set_RNG"):
        pytest.skip("requires surmise >= 1.0.0")
    from gpbayestools.emulator_BAND import EmulatorBAND

    return EmulatorBAND(*args, seed=1, **kwargs)


EMULATORS = {
    "Emulator": (lambda *a: Emulator(*a, npc=4), {}),
    "PCGP": (lambda *a: band_emulator(*a, method="PCGP"), {}),
    "PCSK": (lambda *a: band_emulator(*a, method="PCSK"), {}),
    "hetGPy": (lambda *a: EmulatorHETGPy(*a), {}),
    "SparseGP": (
        lambda *a: EmulatorSparseGP(*a, npc=0.999, M=30),
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
    emu.trainEmulatorAutoMask(**train_kwargs)
    return request.param, emu, train_kwargs


def test_noise_covariance(trained, test_points):
    name, emu, _ = trained
    mean_latent, cov_latent = emu.predict(test_points)
    mean_noise, cov_noise = emu.predict(test_points, include_noise=True)
    np.testing.assert_array_equal(mean_latent, mean_noise)
    noise = cov_noise - cov_latent
    for n, c in zip(noise, cov_noise):
        # the noise covariance is positive semi-definite
        assert np.linalg.eigvalsh(n).min() > -1e-8 * np.abs(c).max()
    if name in ("Emulator", "hetGPy", "PCGP"):
        # these emulators fit a noise term to the noisy training data
        assert np.all(np.diagonal(noise, axis1=1, axis2=2) > 0)


def test_emulator_noise_is_the_white_kernel(trained, test_points):
    name, emu, _ = trained
    if name != "Emulator":
        pytest.skip("only for Emulator")
    _, cov_latent = emu.predict(test_points)
    _, cov_noise = emu.predict(test_points, include_noise=True)
    noise_pc = np.array([emu._gp_noise(gp.kernel_) for gp in emu.gps])
    A = emu._trans_matrix[: emu.npc]
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
    if name not in ("Emulator", "hetGPy"):
        pytest.skip("deterministic training needed to compare")
    pred, pred_err, data, _ = emu.testEmulatorErrors(10, **train_kwargs)
    # the same emulator trained without the test points
    mask = np.ones(emu.nev, dtype=bool)
    mask[-10:] = False
    emu_copy = type(emu).__new__(type(emu))
    emu_copy.__dict__.update(emu.__dict__)
    emu_copy.trainEmulator(mask)
    _, cov = emu_copy.predict(emu.design_points[~mask], include_noise=True)
    np.testing.assert_allclose(pred_err, np.sqrt(np.diagonal(cov, axis1=1, axis2=2)))


def test_emulator_calibration(noisy_training_file, param_file):
    # with noisy training data, the predicted uncertainty of the model
    # function matches the actual errors (it was overestimated with the
    # alpha = 0.1 of versions < 3.0.0)
    emu = Emulator(noisy_training_file, param_file, npc=6, nrestarts=2, seed=1)
    emu.trainEmulatorAutoMask()
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
    emu = EmulatorHETGPy(
        noisy_training_file, param_file, logTrafo=True, exp_and_cov_diagonal=True
    )
    emu.trainEmulatorAutoMask()
    samples = emu.sample_y(test_points, n_samples=4000, random_state=1)
    assert np.all(samples > 0)
    # the median of the log-normal samples is exp(mean in log space), which
    # predict() returns with exp_and_cov_diagonal
    mean = emu.predict(test_points)[0]
    np.testing.assert_allclose(np.median(samples, axis=1), mean, rtol=0.01)
    assert emu.exp_and_cov_diagonal_
