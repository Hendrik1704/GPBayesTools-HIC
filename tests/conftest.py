"""
Shared fixtures and helpers for the tests.
"""

import os
import pickle
import sys

import numpy as np
import pytest

# Resolve the project root (one level up from tests/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

N_PARAMS = 3
N_DESIGN = 60
N_OBS = 8
# observables on very different scales
OBS_SCALE = np.array([1e3, 1e3, 1.0, 1.0, 1.0, 0.05, 0.05, 0.05])


def true_model(params):
    """Smooth test model mapping (alpha, beta, gamma) in [0, 1]^3 to N_OBS
    positive observables."""
    params = np.atleast_2d(params)
    x = np.linspace(0, 1, N_OBS)
    values = 1 + params[:, :1] * np.sin(3 * x) + params[:, 1:2] * x**2 + params[:, 2:3]
    return values * OBS_SCALE


def write_param_file(path, n_params=N_PARAMS):
    names = ["alpha", "beta", "gamma", "delta", "epsilon"][:n_params]
    with open(path, "w") as f:
        f.write("# test parameters\n\n")
        for name in names:
            f.write("{}: {}, 0.0, 1.0\n".format(name, name))
    return str(path)


def write_training_data(path, design, values, rel_err=0.01):
    data = {
        str(i): {
            "parameter": design[i],
            "obs": np.vstack([values[i], rel_err * np.abs(values[i])]),
        }
        for i in range(len(design))
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return str(path)


def latin_hypercube(n_samples, n_dim, rng):
    """Simple random LHD in [0, 1]^n_dim."""
    result = np.zeros((n_samples, n_dim))
    for d in range(n_dim):
        perm = rng.permutation(n_samples)
        result[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
    return result


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def param_file(data_dir):
    return write_param_file(data_dir / "parameters.txt")


@pytest.fixture(scope="session")
def design():
    return latin_hypercube(N_DESIGN, N_PARAMS, np.random.default_rng(1))


@pytest.fixture(scope="session")
def training_file(data_dir, design):
    """Noise-free training data of the test model."""
    return write_training_data(data_dir / "training.pkl", design, true_model(design))


@pytest.fixture(scope="session")
def test_points():
    return np.random.default_rng(2).uniform(0.1, 0.9, size=(5, N_PARAMS))


class LinearEmulator:
    """Emulator of a linear model y = A x + b with a constant covariance, for
    which the posterior of a Gaussian likelihood is known analytically."""

    def __init__(self, A, b, cov):
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.nobs = len(self.b)

    def predict(self, X, return_cov=True):
        X = np.atleast_2d(X)
        mean = X @ self.A.T + self.b
        if not return_cov:
            return mean
        return mean, np.tile(self.cov, (len(X), 1, 1))
