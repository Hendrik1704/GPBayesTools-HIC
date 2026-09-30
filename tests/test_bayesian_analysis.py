"""
Tests for the MCMC module (gpbayestools/bayesian_analysis.py).

The emulators are linear models y = A x + b with a constant covariance, and
the experimental data are the model at X_TRUE. The posterior is then a
Gaussian with mean X_TRUE and the covariance inv(A^T S^-1 A), with S the sum
of the experimental and emulator covariances, which the samplers must
reproduce.

Run with ``python -m pytest tests/test_bayesian_analysis.py``.
"""

import pickle

import dill
import multiprocess.pool
import numpy as np
import pytest
from conftest import LinearEmulator, write_param_file
from scipy.stats import multivariate_normal

from gpbayestools.bayesian_analysis import BayesianAnalysis, mvn_loglike

A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
B = np.array([1.0, 2.0, 3.0, 4.0])
X_TRUE = np.array([0.4, 0.6])
EXP_ERR = 0.05
EMU_VAR = 1e-4
S = (EXP_ERR**2 + EMU_VAR) * np.eye(4)
POST_COV = np.linalg.inv(A.T @ np.linalg.inv(S) @ A)
POST_STD = np.sqrt(np.diag(POST_COV))


def write_exp_data(path, n_sets=1):
    y = A @ X_TRUE + B
    data = {
        str(i): {"obs": np.vstack([y, EXP_ERR * np.ones(4)])} for i in range(n_sets)
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return str(path)


@pytest.fixture
def files(tmp_path):
    """Parameter file, experimental data and two emulator files (the first
    two and the last two observables)."""
    emu_files = []
    for i, rows in enumerate((slice(0, 2), slice(2, 4))):
        emu = LinearEmulator(A[rows], B[rows], EMU_VAR * np.eye(2))
        path = tmp_path / f"emu{i}.dill"
        with open(path, "wb") as f:
            dill.dump(emu, f)
        emu_files.append(str(path))
    return dict(
        par=write_param_file(tmp_path / "par.txt", 2),
        exp=write_exp_data(tmp_path / "exp.pkl"),
        emus=emu_files,
        mcmc=str(tmp_path / "mcmc" / "chain.pkl"),
    )


@pytest.fixture
def analysis(files):
    c = BayesianAnalysis(
        mcmc_path=files["mcmc"], exp_data_path=files["exp"], parameter_file=files["par"]
    )
    c.load_emulators(files["emus"])
    return c


def check_posterior(samples, n_std=4):
    """Compare the samples with the analytic posterior."""
    samples = samples.reshape(-1, 2)
    # the samples are correlated, so allow a few standard deviations of the
    # posterior mean with a reduced effective sample size
    n_eff = len(samples) / 50
    assert np.all(
        np.abs(samples.mean(axis=0) - X_TRUE)
        < n_std * POST_STD / np.sqrt(n_eff) + 0.005
    )
    np.testing.assert_allclose(samples.std(axis=0), POST_STD, rtol=0.2)


# ── Likelihood ───────────────────────────────────────────────────────
def test_mvn_loglike():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4))
    cov = M @ M.T + 4 * np.eye(4)
    y = rng.normal(size=4)
    # mvn_loglike is not normalized
    expected = multivariate_normal(np.zeros(4), cov).logpdf(y) + 2 * np.log(2 * np.pi)
    np.testing.assert_allclose(mvn_loglike(y, cov), expected)
    with pytest.raises(np.linalg.LinAlgError):
        mvn_loglike(np.zeros(2), np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_log_prior_likelihood_posterior(analysis):
    X = np.array([[0.4, 0.6], [0.0, 0.5], [1.0, 1.0], [1.2, 0.5]])
    lp = analysis.log_prior(X)
    np.testing.assert_allclose(lp[:3], 0.0)  # prior volume 1
    assert lp[3] == -np.inf  # outside
    ll = analysis.log_likelihood(X)
    assert np.all(np.isfinite(ll[:3])) and ll[3] == -np.inf
    assert analysis.log_likelihood(X, finite=True)[3] == -1e300
    np.testing.assert_allclose(analysis.log_posterior(X)[:3], lp[:3] + ll[:3])
    np.testing.assert_allclose(analysis.log_likelihood_point_by_point(X)[:3], ll[:3])
    # the likelihood of the linear model
    y = A @ X[0] + B - (A @ X_TRUE + B)
    np.testing.assert_allclose(ll[0], mvn_loglike(y, S))


# ── Emulators and experimental data ──────────────────────────────────
def test_load_emulator_checks_number_of_observables(analysis, files):
    with pytest.raises(ValueError):
        analysis.load_emulators(files["emus"][:1])
    with pytest.raises(ValueError):
        analysis.load_emulators(files["emus"] + files["emus"][:1])
    # loading again replaces the emulators
    analysis.load_emulators(files["emus"])
    assert len(analysis.emulators) == 2


def test_exp_data_with_several_sets(files, tmp_path):
    with pytest.raises(ValueError):
        BayesianAnalysis(
            mcmc_path=files["mcmc"],
            parameter_file=files["par"],
            exp_data_path=write_exp_data(tmp_path / "exp2.pkl", 2),
        )


def test_exp_data_checks(files, tmp_path):
    def analysis_with(values, errors):
        path = tmp_path / "exp_mod.pkl"
        with open(path, "wb") as f:
            pickle.dump({"0": {"obs": np.vstack([values, errors])}}, f)
        return BayesianAnalysis(
            mcmc_path=files["mcmc"], parameter_file=files["par"], exp_data_path=path
        )

    ok = np.ones(4)
    nan = np.array([1, np.nan, 1, 1])
    for values, errors in ((nan, ok), (ok, ok * np.inf), (ok, 0.1 * nan)):
        with pytest.raises(ValueError):
            analysis_with(values, errors)
    analysis = analysis_with(2 * ok, np.array([0.1, 0, 0.1, 0.1]))
    np.testing.assert_array_equal(analysis.exp_data, 2 * ok[np.newaxis])
    np.testing.assert_allclose(np.diag(analysis.exp_data_cov), [0.01, 0, 0.01, 0.01])


# ── Samplers ─────────────────────────────────────────────────────────
def test_emcee(analysis):
    analysis.run_emcee(n_steps=2000, n_burn_steps=400, n_walkers=16, n_thin=5, seed=1)
    assert analysis.chain.shape == (16, 400, 2)
    check_posterior(analysis.chain)

    with open(analysis.chain_path("emcee"), "rb") as f:
        saved = pickle.load(f)
    np.testing.assert_array_equal(saved["chain"], analysis.chain)
    assert saved["last_position"].shape == (16, 2)

    assert saved["n_thin"] == 5

    # continue the chain, with the thinning of the existing chain
    analysis.run_emcee(n_steps=100, seed=2)
    assert analysis.chain.shape == (16, 420, 2)
    with pytest.raises(ValueError):
        analysis.run_emcee(n_steps=100, n_walkers=8)
    with pytest.raises(ValueError):
        analysis.run_emcee(n_steps=100, n_thin=1)


def test_emcee_options_are_checked_before_sampling(analysis, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the sampling started")

    monkeypatch.setattr(analysis, "log_posterior", fail)
    for kwargs in (
        dict(n_steps=0, n_burn_steps=10, n_walkers=8),
        dict(n_steps=10, n_burn_steps=10, n_walkers=8, n_thin=0),
        dict(n_steps=10, n_burn_steps=1, n_walkers=8),
    ):
        with pytest.raises(ValueError):
            analysis.run_emcee(**kwargs)


def test_emcee_seed(files):
    chains = []
    for i in range(2):
        c = BayesianAnalysis(
            mcmc_path=files["mcmc"].replace("chain", f"chain{i}"),
            exp_data_path=files["exp"],
            parameter_file=files["par"],
        )
        c.load_emulators(files["emus"])
        state = np.random.get_state()
        c.run_emcee(n_steps=50, n_burn_steps=20, n_walkers=8, n_thin=1, seed=3)
        # the global random state is not changed by the seed
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])
        chains.append(c.chain)
    np.testing.assert_array_equal(*chains)


def test_ptlmc(analysis):
    analysis.run_ptlmc(
        n_steps=1000,
        n_walkers=8,
        n_temps=6,
        max_temp=10,
        n_start_parameters=200,
        seed=1,
    )
    assert analysis.chain.shape == (8, 1000, 2)
    check_posterior(analysis.chain)
    first = analysis.chain.copy()
    analysis.run_ptlmc(
        n_steps=1000,
        n_walkers=8,
        n_temps=6,
        max_temp=10,
        n_start_parameters=200,
        seed=1,
    )
    np.testing.assert_array_equal(first, analysis.chain)


def test_ptlmc_chains_visit_all_modes():
    # bimodal posterior: every temperature-1 chain must reach both modes,
    # which requires the swaps between the temperature-1 chains
    from gpbayestools import ptlmc

    def log_post(X):
        X = np.atleast_2d(X)
        a = -0.5 * np.sum((X - 0.2) ** 2, axis=1) / 0.04**2
        b = -0.5 * np.sum((X - 0.8) ** 2, axis=1) / 0.04**2
        return np.logaddexp(a, b)

    rng = np.random.default_rng(0)
    result = ptlmc.sampler(
        log_post,
        lambda n: rng.uniform(0, 1, (n, 2)),
        rng,
        2,
        numtemps=6,
        numchain=8,
        sampperchain=500,
        maxtemp=1000,
        nstartparameters=200,
    )
    in_second_mode = result["theta"][:, :, 0] > 0.5
    fraction = in_second_mode.mean(axis=1)
    assert np.all((fraction > 0.2) & (fraction < 0.8))


def test_pocomc_and_log_likelihood_of_chain(analysis):
    analysis.run_pocomc(
        n_effective=512,
        n_active=256,
        n_prior=1024,
        n_total=2000,
        n_evidence=0,
        seed=1,
    )
    assert analysis.chain.ndim == 2 and analysis.chain.shape[1] == 2
    # the resampled pocoMC samples contain duplicates, the effective sample
    # size is n_total
    samples = analysis.chain
    assert np.all(
        np.abs(samples.mean(axis=0) - X_TRUE) < 5 * POST_STD / np.sqrt(2000) + 0.005
    )
    np.testing.assert_allclose(samples.std(axis=0), POST_STD, rtol=0.2)
    with open(analysis.chain_path("pocomc"), "rb") as f:
        assert "logz" in pickle.load(f)

    # log likelihood of the last chain, written next to the chain file
    analysis.compute_log_likelihood_for_chain()
    out = analysis.chain_path("pocomc")
    out = out.with_name(out.stem + "_log_likelihood" + out.suffix)
    with open(out, "rb") as f:
        ll = pickle.load(f)["log_likelihood"]
    assert ll.shape == (len(samples),)
    np.testing.assert_allclose(ll[:10], analysis.log_likelihood(samples[:10]))


def test_pocomc_uses_pool(analysis):
    # a pool object is used for the likelihood evaluations
    class CountingPool:
        n_calls = 0

        def map(self, func, iterable):
            CountingPool.n_calls += 1
            return list(map(func, iterable))

    analysis.run_pocomc(
        n_effective=256,
        n_active=128,
        n_prior=256,
        n_total=256,
        n_evidence=0,
        seed=1,
        pool=CountingPool(),
    )
    assert CountingPool.n_calls > 0
    assert analysis.chain.shape[1] == 2


def test_pocomc_with_process_pool(analysis, monkeypatch):
    # with an integer pool, the analysis is sent to the processes once and
    # only the small worker function with every task
    import gpbayestools.bayesian_analysis as ba

    sizes = []
    map_orig = multiprocess.pool.Pool.map

    def map_recording(self, func, iterable, *args, **kwargs):
        sizes.append(len(dill.dumps(func)))
        return map_orig(self, func, iterable, *args, **kwargs)

    monkeypatch.setattr(multiprocess.pool.Pool, "map", map_recording)
    analysis.run_pocomc(
        n_effective=256,
        n_active=128,
        n_prior=256,
        n_total=512,
        n_evidence=0,
        seed=1,
        pool=2,
    )
    # the bound method with the analysis is much larger
    assert sizes and max(sizes) < len(dill.dumps(analysis._log_likelihood_point)) / 5
    assert ba._worker_analysis is None
    samples = analysis.chain
    assert np.all(np.abs(samples.mean(axis=0) - X_TRUE) < 3 * POST_STD)


def test_log_likelihood_of_chain_requires_chain(analysis):
    with pytest.raises(ValueError):
        analysis.compute_log_likelihood_for_chain()


def test_samplers_write_separate_files(analysis):
    analysis.run_emcee(n_steps=20, n_burn_steps=10, n_walkers=8, n_thin=1, seed=1)
    analysis.run_ptlmc(
        n_steps=20, n_walkers=4, n_temps=4, max_temp=10, n_start_parameters=50, seed=1
    )
    for sampler, shape in (("emcee", (8, 20, 2)), ("ptlmc", (4, 20, 2))):
        with open(analysis.chain_path(sampler), "rb") as f:
            assert pickle.load(f)["chain"].shape == shape
    analysis.compute_log_likelihood_for_chain(sampler="emcee")
    assert analysis.chain.shape == (8, 20, 2)
