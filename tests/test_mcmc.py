"""
Tests for the MCMC module (src/mcmc.py).

The emulators are linear models y = A x + b with a constant covariance, and
the experimental data are the model at X_TRUE. The posterior is then a
Gaussian with mean X_TRUE and the covariance inv(A^T S^-1 A), with S the sum
of the experimental and emulator covariances, which the samplers must
reproduce.

Run with ``python -m pytest tests/test_mcmc.py``.
"""

import pickle

import dill
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from conftest import LinearEmulator, write_param_file
from src.mcmc import Chain, mvn_loglike

A = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
B = np.array([1., 2., 3., 4.])
X_TRUE = np.array([0.4, 0.6])
EXP_ERR = 0.05
EMU_VAR = 1e-4
S = (EXP_ERR**2 + EMU_VAR) * np.eye(4)
POST_COV = np.linalg.inv(A.T @ np.linalg.inv(S) @ A)
POST_STD = np.sqrt(np.diag(POST_COV))


def write_exp_data(path, n_sets=1):
    y = A @ X_TRUE + B
    data = {str(i): {'obs': np.vstack([y, EXP_ERR * np.ones(4)])}
            for i in range(n_sets)}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return str(path)


@pytest.fixture
def files(tmp_path):
    """Parameter file, experimental data and two emulator files (the first
    two and the last two observables)."""
    emu_files = []
    for i, rows in enumerate((slice(0, 2), slice(2, 4))):
        emu = LinearEmulator(A[rows], B[rows], EMU_VAR * np.eye(2))
        path = tmp_path / "emu{}.dill".format(i)
        with open(path, 'wb') as f:
            dill.dump(emu, f)
        emu_files.append(str(path))
    return dict(par=write_param_file(tmp_path / "par.txt", 2),
                exp=write_exp_data(tmp_path / "exp.pkl"),
                emus=emu_files,
                mcmc=str(tmp_path / "mcmc" / "chain.pkl"))


@pytest.fixture
def chain(files):
    c = Chain(mcmc_path=files['mcmc'], expdata_path=files['exp'],
              model_parafile=files['par'])
    c.loadEmulator(files['emus'])
    return c


def check_posterior(samples, n_std=4):
    """Compare the samples with the analytic posterior."""
    samples = samples.reshape(-1, 2)
    # the samples are correlated, so allow a few standard deviations of the
    # posterior mean with a reduced effective sample size
    n_eff = len(samples) / 50
    assert np.all(np.abs(samples.mean(axis=0) - X_TRUE)
                  < n_std * POST_STD / np.sqrt(n_eff) + 0.005)
    np.testing.assert_allclose(samples.std(axis=0), POST_STD, rtol=0.2)


# ── Likelihood ───────────────────────────────────────────────────────
def test_mvn_loglike():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4))
    cov = M @ M.T + 4 * np.eye(4)
    y = rng.normal(size=4)
    # mvn_loglike is not normalized
    expected = (multivariate_normal(np.zeros(4), cov).logpdf(y)
                + 2 * np.log(2 * np.pi))
    np.testing.assert_allclose(mvn_loglike(y, cov), expected)
    with pytest.raises(np.linalg.LinAlgError):
        mvn_loglike(np.zeros(2), np.array([[1., 2.], [2., 1.]]))


def test_log_prior_likelihood_posterior(chain):
    X = np.array([[0.4, 0.6], [0.0, 0.5], [1.0, 1.0], [1.2, 0.5]])
    lp = chain.log_prior(X)
    np.testing.assert_allclose(lp[:3], 0.0)       # prior volume 1
    assert lp[3] == -np.inf                       # outside
    ll = chain.log_likelihood(X)
    assert np.all(np.isfinite(ll[:3])) and ll[3] == -np.inf
    assert chain.log_likelihood(X, finite=True)[3] == -1e300
    np.testing.assert_allclose(chain.log_posterior(X)[:3], lp[:3] + ll[:3])
    np.testing.assert_allclose(chain.log_likelihood_point_by_point(X)[:3],
                               ll[:3])
    # the likelihood of the linear model
    y = A @ X[0] + B - (A @ X_TRUE + B)
    np.testing.assert_allclose(ll[0], mvn_loglike(y, S))


# ── Emulators and experimental data ──────────────────────────────────
def test_load_emulator_checks_number_of_observables(chain, files):
    with pytest.raises(ValueError):
        chain.loadEmulator(files['emus'][:1])
    with pytest.raises(ValueError):
        chain.loadEmulator(files['emus'] + files['emus'][:1])
    # loading again replaces the emulators
    chain.loadEmulator(files['emus'])
    assert len(chain.emuList) == 2


def test_exp_data_with_several_sets(files, tmp_path):
    with pytest.raises(ValueError):
        Chain(mcmc_path=files['mcmc'], model_parafile=files['par'],
              expdata_path=write_exp_data(tmp_path / "exp2.pkl", 2))


# ── Samplers ─────────────────────────────────────────────────────────
def test_emcee(chain):
    chain.run_mcmc(nsteps=2000, nburnsteps=400, nwalkers=16, nthin=5, seed=1)
    assert chain.chain.shape == (16, 400, 2)
    check_posterior(chain.chain)

    with open(chain.chain_path('emcee'), 'rb') as f:
        saved = pickle.load(f)
    np.testing.assert_array_equal(saved['chain'], chain.chain)
    assert saved['last_position'].shape == (16, 2)

    # continue the chain
    chain.run_mcmc(nsteps=100, nthin=5, seed=2)
    assert chain.chain.shape == (16, 420, 2)
    with pytest.raises(ValueError):
        chain.run_mcmc(nsteps=100, nwalkers=8)


def test_emcee_seed(files):
    chains = []
    for i in range(2):
        c = Chain(mcmc_path=files['mcmc'].replace('chain', 'chain{}'.format(i)),
                  expdata_path=files['exp'], model_parafile=files['par'])
        c.loadEmulator(files['emus'])
        c.run_mcmc(nsteps=50, nburnsteps=20, nwalkers=8, nthin=1, seed=3)
        chains.append(c.chain)
    np.testing.assert_array_equal(*chains)


def test_ptlmc(chain):
    chain.run_MCMC_PTLMC(nsteps=1000, nwalkers=8, ntemps=6, maxtemp=10,
                         nstartparameters=200, seed=1)
    assert chain.chain.shape == (8, 1000, 2)
    check_posterior(chain.chain)
    first = chain.chain.copy()
    chain.run_MCMC_PTLMC(nsteps=1000, nwalkers=8, ntemps=6, maxtemp=10,
                         nstartparameters=200, seed=1)
    np.testing.assert_array_equal(first, chain.chain)


def test_pocomc_and_log_likelihood_of_chain(chain):
    chain.run_pocoMC(n_effective=512, n_active=256, n_prior=1024,
                     n_total=2000, n_evidence=0, random_state=1)
    assert chain.chain.ndim == 2 and chain.chain.shape[1] == 2
    # pocoMC samples are independent
    samples = chain.chain
    assert np.all(np.abs(samples.mean(axis=0) - X_TRUE)
                  < 5 * POST_STD / np.sqrt(len(samples)) + 0.005)
    np.testing.assert_allclose(samples.std(axis=0), POST_STD, rtol=0.2)
    with open(chain.chain_path('pocoMC'), 'rb') as f:
        assert 'logz' in pickle.load(f)

    # log likelihood of the last chain, written next to the chain file
    chain.compute_log_likelihood_for_chain()
    out = chain.chain_path('pocoMC')
    out = out.with_name(out.stem + '_log_likelihood' + out.suffix)
    with open(out, 'rb') as f:
        ll = pickle.load(f)['log_likelihood']
    assert ll.shape == (len(samples),)
    np.testing.assert_allclose(ll[:10], chain.log_likelihood(samples[:10]))


def test_log_likelihood_of_chain_requires_chain(chain):
    with pytest.raises(ValueError):
        chain.compute_log_likelihood_for_chain()


def test_samplers_write_separate_files(chain):
    chain.run_mcmc(nsteps=20, nburnsteps=10, nwalkers=8, nthin=1, seed=1)
    chain.run_MCMC_PTLMC(nsteps=20, nwalkers=4, ntemps=4, maxtemp=10,
                         nstartparameters=50, seed=1)
    for sampler, shape in (('emcee', (8, 20, 2)), ('PTLMC', (4, 20, 2))):
        with open(chain.chain_path(sampler), 'rb') as f:
            assert pickle.load(f)['chain'].shape == shape
    chain.compute_log_likelihood_for_chain(sampler='emcee')
    assert chain.chain.shape == (8, 20, 2)
