"""
Bayesian model calibration with the emulators.

The `BayesianAnalysis` class provides the following samplers:

- ``run_emcee``: affine-invariant ensemble MCMC sampler emcee
- ``run_ptlmc``: parallel tempering Langevin Monte Carlo (PTLMC) from surmise
- ``run_pocomc``: preconditioned Monte Carlo with pocoMC (recommended)
"""

import logging
import os
import pickle
import tempfile
from pathlib import Path

import emcee
import numpy as np
import pocomc
from scipy.linalg import lapack
from scipy.stats import uniform

from . import load_emulator, parse_model_parameter_file, ptlmc

logger = logging.getLogger(__name__)


def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood.

    The log-likelihood of the difference vector `y` and the covariance matrix
    `cov` is::

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC. The
    normalization is const = -n/2*log(2*pi), where n is the dimensionality.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Difference vector (model - experiment). Must have dtype float64.
    cov : ndarray of shape (n, n)
        Covariance matrix. Must have dtype float64.
        The dtypes and shapes of `y` and `cov` are NOT CHECKED.

    Returns
    -------
    float
        The unnormalized log-likelihood.

    Raises
    ------
    ValueError
        If a LAPACK routine reports an illegal argument value.
    numpy.linalg.LinAlgError
        If `cov` is not positive definite.
    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            f"lapack dpotrf error: the {-info}-th argument had an illegal value"
        )
    elif info > 0:
        raise np.linalg.LinAlgError(
            "lapack dpotrf error: "
            f"the leading minor of order {info} is not positive definite"
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            f"lapack dpotrs error: the {-info}-th argument had an illegal value"
        )

    return -0.5 * np.dot(y, alpha) - np.log(L.diagonal()).sum()


def _write_pickle(path, data):
    """
    Write `data` to the pickle file `path` via a temporary file, so that an
    interrupted write does not destroy an existing file.
    """
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp_path, path)
    except BaseException:
        os.remove(tmp_path)
        raise


class LoggingEnsembleSampler(emcee.EnsembleSampler):
    """
    Ensemble sampler of emcee that logs the progress of the run.

    The constructor parameters are those of `emcee.EnsembleSampler`.
    """

    def run_mcmc(self, initial_state, nsteps, status=None, **kwargs):
        """
        Run MCMC and log the acceptance fraction every `status` steps.

        Parameters
        ----------
        initial_state : array_like of shape (nwalkers, ndim) or emcee.State
            Initial positions of the walkers.
        nsteps : int
            Number of steps. Must be at least 1.
        status : int or None, default=None
            Number of steps between log messages. If None, approximately 10% of
            `nsteps` (at least 1). The last step is always logged.
        **kwargs
            Further keyword arguments passed to `emcee.EnsembleSampler.sample`.

        Returns
        -------
        emcee.State
            The state of the last iteration.

        Raises
        ------
        ValueError
            If `nsteps` is smaller than 1.
        """
        if nsteps < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")
        if status is None:
            status = max(nsteps // 10, 1)
        logger.info("Running %d walkers for %d steps ...", self.nwalkers, nsteps)

        # the state of the last iteration is returned
        for n, result in enumerate(  # noqa: B007
            self.sample(initial_state, iterations=nsteps, **kwargs), start=1
        ):
            if n % status == 0 or n == nsteps:
                af = self.acceptance_fraction
                logger.info(
                    "Step %d/%d: acceptance fraction: "
                    "mean %.4f, std %.4f, min %.4f, max %.4f",
                    n,
                    nsteps,
                    af.mean(),
                    af.std(),
                    af.min(),
                    af.max(),
                )

        return result


class BayesianAnalysis:
    """
    High-level interface for running MCMC calibration and accessing results.

    All emulators must use the parameters of the parameter file in the same
    order.

    The experimental data are used as they are given. For emulators that
    return predictions in log space (``log_trafo=True`` and
    ``exp_and_cov_diagonal=False``), the experimental data must be
    log-transformed by the user as well.

    Each sampler writes its chain to its own file, which is derived from
    `mcmc_path` by adding the name of the sampler, e.g. for the default
    ``./mcmc/chain.pkl``: ``./mcmc/chain_emcee.pkl``,
    ``./mcmc/chain_pocomc.pkl`` and ``./mcmc/chain_ptlmc.pkl``
    (see :meth:`chain_path`).

    Parameters
    ----------
    mcmc_path : str or path-like, default="./mcmc/chain.pkl"
        Base path of the chain files. The parent directory is created if it
        does not exist.
    exp_data_path : str or path-like, default="./exp_data.pkl"
        Path of the pickle file with the experimental data. It must contain a
        dictionary with exactly one data set, whose ``"obs"`` entry holds the
        values and the errors of the data points (see
        :meth:`_read_in_exp_data_pickle`).
    parameter_file : str or path-like, default="./model.dat"
        Path of the model parameter file with the label and the range
        (minimum and maximum) of each parameter.

    Attributes
    ----------
    mcmc_path : pathlib.Path
        Base path of the chain files, see `chain_path`.
    pardict : dict
        The parameters of the parameter file, see
        ``parse_model_parameter_file``.
    ndim : int
        Number of parameters.
    labels : list of str
        Labels of the parameters.
    param_min, param_max : ndarray of shape (ndim,)
        Ranges of the parameters (bounds of the uniform prior).
    prior_volume : float
        Volume of the parameter space.
    exp_data : ndarray of shape (1, nobs)
        Values of the experimental data points.
    exp_data_cov : ndarray of shape (nobs, nobs)
        Diagonal covariance matrix of the experimental data points.
    nobs : int
        Number of experimental data points.
    emulators : list
        The emulators loaded with `load_emulators`.
    chain : ndarray or False
        Samples of the last sampler run with this object or loaded by
        `compute_log_likelihood_for_chain`, False before.
    chain_sampler : str or None
        Name of the sampler of `chain`.
    """

    samplers = ("emcee", "pocomc", "ptlmc")

    def __init__(
        self,
        mcmc_path="./mcmc/chain.pkl",
        exp_data_path="./exp_data.pkl",
        parameter_file="./model.dat",
    ):
        self.mcmc_path = Path(mcmc_path)
        self.mcmc_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "The chains of the samplers are saved in {}".format(
                ", ".join(str(self.chain_path(s)) for s in self.samplers)
            )
        )

        # load the model parameter file
        self.pardict = parse_model_parameter_file(parameter_file)
        self.ndim = len(self.pardict.keys())
        self.labels = []
        self.param_min = []
        self.param_max = []
        for val in self.pardict.values():
            self.labels.append(val[0])
            self.param_min.append(val[1])
            self.param_max.append(val[2])
        self.param_min = np.array(self.param_min)
        self.param_max = np.array(self.param_max)

        # the volume of the uniform prior
        diff = self.param_max - self.param_min
        self.prior_volume = np.prod(diff)
        logger.info(f"Loaded {self.ndim} model parameters from {parameter_file}")

        # load the experimental data to be fit
        self.exp_data, self.exp_data_cov = self._read_in_exp_data_pickle(exp_data_path)
        self.nobs = self.exp_data.shape[1]
        self.emulators = []
        self.chain = False
        # sampler that generated self.chain
        self.chain_sampler = None

    def load_emulators(self, emulator_path_list):
        """
        Load the emulators from files.

        The loaded emulators replace previously loaded emulators. The order of
        the emulators must be the order of the observables in the experimental
        data, and their numbers of observables must add up to the number of
        experimental data points.

        Parameters
        ----------
        emulator_path_list : list of str or path-like
            Paths of the emulator files.

        Raises
        ------
        ValueError
            If the total number of observables of the emulators differs from
            the number of experimental data points.
        """
        emu_list = [load_emulator(emu_path) for emu_path in emulator_path_list]
        nobs_emu = [emu.nobs for emu in emu_list]
        if sum(nobs_emu) != self.nobs:
            raise ValueError(
                "The emulators have {} observables in total ({}), but the "
                "experimental data have {} data points".format(
                    sum(nobs_emu), ", ".join(map(str, nobs_emu)), self.nobs
                )
            )
        self.emulators = emu_list
        logger.info(
            "Loaded {} emulators with {} observables in total ({})".format(
                len(emu_list), self.nobs, ", ".join(map(str, nobs_emu))
            )
        )

    def _predict(self, X):
        """
        Predict the mean and covariance of all observables at the points `X`.

        The predictions of the emulators are concatenated, and the covariance
        is block diagonal with one block per emulator. Returns arrays of shape
        (n, nobs) and (n, nobs, nobs), and raises a ValueError if no emulators
        are loaded or if they do not predict `nobs` observables in total.
        """
        if not self.emulators:
            raise ValueError("No emulators are loaded, call load_emulators first")
        n_preds = X.shape[0]
        model_pred = np.zeros([n_preds, self.nobs])
        model_pred_cov = np.zeros([n_preds, self.nobs, self.nobs])
        curr_idx = 0
        for emu_i in self.emulators:
            model_Y, model_cov = emu_i.predict(X, return_cov=True)
            nobs_i = model_Y.shape[1]
            model_pred[:, curr_idx : curr_idx + nobs_i] = model_Y
            model_pred_cov[
                :, curr_idx : curr_idx + nobs_i, curr_idx : curr_idx + nobs_i
            ] = model_cov
            curr_idx += nobs_i
        if curr_idx != self.nobs:
            raise ValueError(
                f"The emulators predict {curr_idx} observables, but the experimental "
                f"data have {self.nobs} data points"
            )
        return model_pred, model_pred_cov

    def _inside(self, X):
        """
        Return True for the points in `X` inside the parameter ranges.

        The boundaries are included, as in the uniform prior of pocoMC.
        """
        return np.all((X >= self.param_min) & (X <= self.param_max), axis=-1)

    def log_prior(self, X):
        """
        Evaluate the (normalized, uniform) log prior at `X`.

        Parameters
        ----------
        X : array_like of shape (n, ndim) or (ndim,)
            Points in parameter space.

        Returns
        -------
        ndarray of shape (n,)
            Log prior, ``-log(prior_volume)`` inside the parameter ranges
            (boundaries included) and ``-inf`` outside.
        """
        X = np.atleast_2d(np.asarray(X))
        lp = np.log(np.ones(X.shape[0]) / self.prior_volume)
        lp[~self._inside(X)] = -np.inf
        return lp

    def log_likelihood(self, X, finite=False):
        """
        Evaluate the log-likelihood at `X`.

        The covariance is the sum of the emulator covariance and the
        experimental covariance. The log-likelihood is not normalized (see
        :func:`mvn_loglike`).

        Parameters
        ----------
        X : array_like of shape (n, ndim) or (ndim,)
            Points in parameter space.
        finite : bool, default=False
            If True, points outside the parameter ranges get the finite value
            -1e300 instead of ``-inf`` (used for pocoMC).

        Returns
        -------
        ndarray of shape (n,)
            Log-likelihood at each point.

        Raises
        ------
        ValueError
            If no emulators are loaded, or if they do not predict the number
            of experimental data points.
        numpy.linalg.LinAlgError
            If the covariance matrix at a point is not positive definite.
        """
        X = np.atleast_2d(np.asarray(X))
        lp = np.zeros(X.shape[0])
        inside = self._inside(X)
        lp[~inside] = -1e300 if finite else -np.inf

        if np.any(inside):
            model_Y, model_cov = self._predict(X[inside])
            # difference (model - experiment) and the sum of the emulator and
            # the experimental covariance
            dY = model_Y - self.exp_data
            cov = model_cov + self.exp_data_cov
            lp[inside] += list(map(mvn_loglike, dY, cov))
        return lp

    def _log_likelihood_point(self, x, finite=False):
        """Log-likelihood at the single point `x` as a float."""
        return float(self.log_likelihood(x, finite=finite)[0])

    def log_likelihood_point_by_point(self, X):
        """
        Evaluate the log-likelihood at `X` point by point.

        This is used to compute the log-likelihood for each point of an
        already generated chain. The progress is logged about 10 times.

        Parameters
        ----------
        X : ndarray of shape (n, ndim)
            Points in parameter space.

        Returns
        -------
        ndarray of shape (n,)
            Log-likelihood at each point, ``-inf`` outside the parameter
            ranges.
        """
        X = np.atleast_2d(np.asarray(X))
        n_points = X.shape[0]
        lp = np.zeros(n_points)
        log_every = max(n_points // 10, 1)

        for k in range(n_points):
            if k % log_every == 0:
                logger.info(
                    f"Evaluating the log-likelihood at point {k + 1}/{n_points} ..."
                )
            lp[k] = self.log_likelihood(X[k])[0]
        return lp

    def log_posterior(self, X):
        """
        Evaluate the log posterior at `X`.

        The log posterior is the sum of the log prior and the log-likelihood.

        Parameters
        ----------
        X : array_like of shape (n, ndim) or (ndim,)
            Points in parameter space.

        Returns
        -------
        ndarray of shape (n,)
            Log posterior at each point, ``-inf`` outside the parameter ranges.
        """
        return self.log_prior(X) + self.log_likelihood(X)

    def _read_in_exp_data_pickle(self, filepath):
        """
        Read the experimental data and compute their covariance matrix.

        The pickle file must contain a dictionary with exactly one entry,
        whose ``"obs"`` array holds the values in the first and the errors in
        the second row. The covariance matrix is diagonal with the squared
        errors (NaN errors are set to 0 with a warning). Returns the data of
        shape (1, nobs) and the covariance of shape (nobs, nobs). Raises a
        ValueError for non-finite values or infinite errors.
        """
        model_data = []
        model_data_err = []

        with open(filepath, "rb") as fp:
            data_dict = pickle.load(fp)
        if len(data_dict) != 1:
            raise ValueError(
                f"The experimental data file {filepath} must contain exactly one data "
                f"set, but contains {len(data_dict)}"
            )

        for event_id in data_dict.keys():
            temp_data = data_dict[event_id]["obs"].transpose()
            model_data.append(temp_data[:, 0])
            model_data_err.append(temp_data[:, 1])
        logger.info(
            f"Loaded {model_data[0].shape[0]} experimental data points from {filepath}"
        )
        model_data = np.array(model_data)
        model_data_err = np.abs(np.array(model_data_err))
        if not np.all(np.isfinite(model_data)):
            raise ValueError(f"The experimental data in {filepath} are not finite")
        if np.any(np.isinf(model_data_err)):
            raise ValueError(
                f"The experimental data in {filepath} have infinite errors"
            )
        n_nan = int(np.sum(np.isnan(model_data_err)))
        if n_nan > 0:
            logger.warning(f"Setting {n_nan} NaN errors of the experimental data to 0")
            model_data_err = np.nan_to_num(model_data_err)
        nobs = model_data.shape[1]

        data_cov = np.zeros((nobs, nobs))
        model_data_err = model_data_err.flatten()
        np.fill_diagonal(data_cov, (model_data_err) ** 2)

        return model_data, data_cov

    def random_pos(self, n=1):
        """
        Generate random positions in parameter space.

        The positions are drawn uniformly within the parameter ranges using
        numpy's global random number generator.

        Parameters
        ----------
        n : int, default=1
            Number of positions.

        Returns
        -------
        ndarray of shape (n, ndim)
            Random positions.
        """
        return np.random.uniform(self.param_min, self.param_max, (n, self.ndim))

    def chain_path(self, sampler):
        """
        Return the path of the chain file of a sampler.

        The path is `mcmc_path` with ``_<sampler>`` added to the file name,
        e.g. ``./mcmc/chain_emcee.pkl``.

        Parameters
        ----------
        sampler : {'emcee', 'pocomc', 'ptlmc'}
            Name of the sampler.

        Returns
        -------
        pathlib.Path
            Path of the chain file.

        Raises
        ------
        ValueError
            If `sampler` is not a known sampler.
        """
        if sampler not in self.samplers:
            raise ValueError(f"Unknown sampler '{sampler}', use one of {self.samplers}")
        return self.mcmc_path.with_name(
            f"{self.mcmc_path.stem}_{sampler}{self.mcmc_path.suffix}"
        )

    def _warn_overwrite(self, sampler):
        """Warn if the chain file of `sampler` exists and is overwritten."""
        if self.chain_path(sampler).exists():
            logger.warning(
                f"Overwriting the existing chain file {self.chain_path(sampler)}"
            )

    def run_emcee(
        self,
        n_steps=500,
        n_burn_steps=None,
        n_walkers=None,
        status=None,
        n_thin=None,
        skip_initial_state_check=False,
        seed=None,
    ):
        """
        Run MCMC model calibration with emcee.

        Markov chain Monte Carlo model calibration using the `affine-invariant
        ensemble sampler (emcee) <http://dfm.io/emcee>`_.

        If the chain file ``chain_path("emcee")`` already contains a chain,
        continue from its last walker positions. Otherwise, run a burn-in and
        start a new chain. The burn-in is run in two halves: after the first
        half, the walkers are moved to the most likely distinct points found
        so far. The production steps are thinned by `n_thin` and appended to
        the chain file. ``self.chain`` is the whole chain of the file, with
        shape (n_walkers, n_saved, ndim), where n_saved is the total number of
        saved (thinned) steps.

        Parameters
        ----------
        n_steps : int, default=500
            Number of production steps.
        n_burn_steps : int or None, default=None
            Number of burn-in steps. Must be at least 2. Required to start a
            new chain, ignored when continuing an existing chain.
        n_walkers : int or None, default=None
            Number of walkers. Required to start a new chain. When continuing
            an existing chain, it defaults to the number of walkers of that
            chain and must match it.
        status : int or None, default=None
            Number of steps between progress log messages (see
            :meth:`LoggingEnsembleSampler.run_mcmc`).
        n_thin : int or None, default=None
            Thinning of the production chain, only every `n_thin`-th step is
            stored. None uses 10 for a new chain and the thinning of the
            existing chain when continuing it. The thinning is stored in the
            chain file (``"n_thin"``) and must be the same for all runs of a
            chain.
        skip_initial_state_check : bool, default=False
            Passed to emcee. If True, do not check that the initial walker
            positions are linearly independent.
        seed : int or None, default=None
            Seed of the random numbers of the initial positions and of emcee,
            which makes the chain reproducible. The state of numpy's global
            random number generator, which emcee copies, is restored
            afterwards. If None, the global random number generator is used.

        Raises
        ------
        ValueError
            If `n_steps` or `n_thin` is smaller than 1, if `n_burn_steps` or
            `n_walkers` is missing for a new chain, if `n_burn_steps` is
            smaller than 2, if the existing chain was not generated with
            emcee, or if `n_walkers`, `n_thin` or the number of parameters
            does not match the existing chain.
        """
        # checked before the (long) sampling
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if n_thin is not None and n_thin < 1:
            raise ValueError(f"n_thin must be >= 1, got {n_thin}")
        chain_file = self.chain_path("emcee")
        chain_data = {}
        try:
            with open(chain_file, "rb") as f:
                chain_data = pickle.load(f)
        except FileNotFoundError:
            pass

        burn_in = "chain" not in chain_data

        if burn_in:
            if n_burn_steps is None or n_walkers is None:
                raise ValueError(
                    "n_burn_steps and n_walkers are required to start a new chain"
                )
            if n_burn_steps < 2:
                raise ValueError(
                    "n_burn_steps must be >= 2, the burn-in is run in two halves"
                )
            if n_thin is None:
                n_thin = 10
        else:
            # emcee chains have shape (n_walkers, n_steps, ndim), pocoMC samples
            # (nsamples, ndim)
            if chain_data["chain"].ndim != 3:
                raise ValueError(
                    f"The chain in {chain_file} was not generated with emcee and "
                    "cannot be continued, use a different mcmc_path"
                )
            if chain_data["chain"].shape[2] != self.ndim:
                raise ValueError(
                    f"The chain in {chain_file} has {chain_data['chain'].shape[2]} "
                    f"parameters, but the parameter file has {self.ndim}"
                )
            if n_walkers is None:
                n_walkers = chain_data["chain"].shape[0]
            elif n_walkers != chain_data["chain"].shape[0]:
                raise ValueError(
                    "The existing chain has {} walkers, but n_walkers = {}".format(
                        chain_data["chain"].shape[0], n_walkers
                    )
                )
            # chains saved with older versions do not contain the thinning
            n_thin_chain = chain_data.get("n_thin")
            if n_thin is None:
                n_thin = 10 if n_thin_chain is None else n_thin_chain
            elif n_thin_chain is not None and n_thin != n_thin_chain:
                raise ValueError(
                    f"The existing chain was thinned with n_thin = {n_thin_chain}, "
                    f"but n_thin = {n_thin}"
                )
            if n_burn_steps is not None:
                logger.info("Continuing the existing chain, n_burn_steps is ignored")
        if n_steps % n_thin != 0:
            logger.warning(
                f"n_steps = {n_steps} is not a multiple of n_thin = {n_thin}, the "
                "thinned samples are not equally spaced where the chain is continued"
            )

        if burn_in:
            logger.info(
                f"Running emcee with {n_walkers} walkers for {n_burn_steps} burn-in "
                f"and {n_steps} production steps ..."
            )
        else:
            logger.info(
                f"Continuing the emcee chain in {chain_file} with {n_walkers} "
                f"walkers for {n_steps} steps ..."
            )
        if seed is not None:
            # emcee initializes its random number generator from numpy's
            # global state, which is restored after the sampler is created
            global_random_state = np.random.get_state()
            np.random.seed(seed)
        if burn_in:
            # drawn before the sampler is created, which copies the state of
            # numpy's global random number generator
            initial_state = self.random_pos(n_walkers)
        sampler = LoggingEnsembleSampler(
            n_walkers, self.ndim, self.log_posterior, vectorize=True
        )
        if seed is not None:
            np.random.set_state(global_random_state)

        if burn_in:
            logger.info("Starting the burn-in from random positions ...")

            # Run first half of burn-in starting from random positions.
            n_burn_first = n_burn_steps // 2
            state = sampler.run_mcmc(
                initial_state,
                n_burn_first,
                status=status,
                skip_initial_state_check=skip_initial_state_check,
            )
            logger.info(
                "Restarting the walkers at the most probable points of the first "
                "half of the burn-in ..."
            )
            # Reposition walkers to the most likely points in the chain,
            # then run the second half of burn-in.  This significantly
            # accelerates burn-in and helps prevent stuck walkers.
            log_prob = sampler.get_log_prob(flat=True)
            # indices of the distinct log-probabilities in ascending order
            idx = np.unique(log_prob, return_index=True)[1]
            idx = idx[np.isfinite(log_prob[idx])]
            if len(idx) >= n_walkers:
                initial_state = sampler.get_chain(flat=True)[idx[-n_walkers:]]
            else:
                logger.warning(
                    f"Only {len(idx)} distinct points with finite probability in the "
                    "first half of the burn-in, continuing from the current "
                    "walker positions"
                )
                initial_state = state.coords
            sampler.reset()
            initial_state = sampler.run_mcmc(
                initial_state,
                n_burn_steps - n_burn_first,
                status=status,
                skip_initial_state_check=skip_initial_state_check,
            )
            sampler.reset()
            logger.info("Burn-in finished, starting the production run ...")
        else:
            # the last walker positions of the previous run, or for chains
            # saved with older versions, the last thinned sample
            initial_state = chain_data.get(
                "last_position", chain_data["chain"][:, -1, :]
            )

        state = sampler.run_mcmc(
            initial_state,
            n_steps,
            status=status,
            skip_initial_state_check=skip_initial_state_check,
        )
        chain_data["last_position"] = state.coords
        chain_data["n_thin"] = n_thin

        # shape (n_walkers, n_steps, ndim)
        thinned_chain = np.swapaxes(sampler.get_chain(), 0, 1)[:, ::n_thin, :]
        if "chain" in chain_data:
            chain_data["chain"] = np.concatenate(
                (chain_data["chain"], thinned_chain), axis=1
            )
            self.chain = chain_data["chain"]
        else:
            chain_data["chain"] = thinned_chain
            self.chain = thinned_chain

        self.chain_sampler = "emcee"

        # write the whole chain, including the steps of previous runs
        logger.info(
            f"Writing the chain with {self.chain.shape[1]} samples per walker to "
            f"{chain_file}"
        )
        _write_pickle(chain_file, chain_data)

    def run_ptlmc(
        self,
        n_steps=500,
        n_walkers=16,
        n_temps=50,
        max_temp=100,
        n_start_parameters=1000,
        seed=None,
    ):
        """
        Run parallel tempering ensemble MCMC with Langevin Monte Carlo.

        This function wraps the PTLMC sampler adapted from surmise (see
        :func:`gpbayestools.ptlmc.sampler`). The initial points are drawn
        uniformly within the parameter ranges, and ``n_temps + n_walkers`` of
        them are optimized with L-BFGS-B before the sampling. The first
        ``2 * n_steps`` steps tune the step size and are discarded. The
        samples of the temperature-1 chains are stored in ``self.chain`` with
        shape (n_walkers, n_steps, ndim) and written to
        ``chain_path("ptlmc")``, overwriting an existing file.

        Parameters
        ----------
        n_steps : int, default=500
            Number of samples per chain.
        n_walkers : int, default=16
            Number of chains of temperature 1.
        n_temps : int, default=50
            Number of chains of varying temperature.
        max_temp : float, default=100
            Maximum temperature used in parallel tempering.
        n_start_parameters : int, default=1000
            Number of initial random draws from the parameter space.
        seed : int or None, default=None
            Seed of the random number generator of the sampler, which makes
            the chain reproducible.
        """
        rng = np.random.default_rng(seed)
        chain_data = {}

        def draw_func(n):
            return rng.uniform(self.param_min, self.param_max, (n, self.ndim))

        self._warn_overwrite("ptlmc")
        logger.info(
            f"Running PTLMC with {n_walkers} chains and {n_temps} temperatures "
            f"(maximum {max_temp}) for {n_steps} samples per chain ..."
        )
        result_dict = ptlmc.sampler(
            logpostfunc=self.log_posterior,
            draw_func=draw_func,
            rng=rng,
            ndim=self.ndim,
            theta0=None,
            numtemps=n_temps,
            numchain=n_walkers,
            sampperchain=n_steps,
            maxtemp=max_temp,
            nstartparameters=n_start_parameters,
        )

        # shape (n_walkers, n_steps, ndim)
        self.chain = result_dict["theta"]

        self.chain_sampler = "ptlmc"

        # Write the chain to file (n_walkers, n_steps, self.ndim)
        chain_data["chain"] = self.chain
        logger.info(f"Writing the PTLMC chains to {self.chain_path('ptlmc')}")
        _write_pickle(self.chain_path("ptlmc"), chain_data)

    def compute_log_likelihood_for_chain(self, sampler=None, output_path=None):
        """
        Compute the log-likelihood for each point in a chain and save it.

        The result is stored in a new pickle file as a dictionary with the key
        ``"log_likelihood"``, with the shape of the chain without the last
        (parameter) axis.

        Parameters
        ----------
        sampler : {'emcee', 'pocomc', 'ptlmc'} or None, default=None
            Sampler whose chain is loaded from its chain file (see
            :meth:`chain_path`). If None, the chain of the last sampler run
            with this object is used.
        output_path : str or path-like or None, default=None
            Path of the output file. If None, the output is written next to
            the chain file with the suffix ``_log_likelihood``, e.g.
            ``./mcmc/chain_emcee_log_likelihood.pkl``. The parent directory is
            created if it does not exist.

        Raises
        ------
        ValueError
            If `sampler` is None and no chain has been run with this object,
            or if `sampler` is not a known sampler.
        FileNotFoundError
            If the chain file of `sampler` does not exist.
        """
        if sampler is not None:
            logger.info(f"Loading the {sampler} chain from {self.chain_path(sampler)}")
            with open(self.chain_path(sampler), "rb") as f:
                chain_data = pickle.load(f)
            self.chain = chain_data["chain"]
            self.chain_sampler = sampler
        elif self.chain is False:
            raise ValueError(
                "No chain has been run with this object, specify "
                "the sampler of the chain to load"
            )
        if output_path is None:
            chain_file = self.chain_path(self.chain_sampler)
            output_path = chain_file.with_name(
                chain_file.stem + "_log_likelihood" + chain_file.suffix
            )
        # create the output directory before the (expensive) computation
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(output_path).exists():
            logger.warning(f"Overwriting the existing file {output_path}")
        logger.info(
            f"Computing the log-likelihood of the {self.chain_sampler} chain ..."
        )
        points = self.chain.reshape(-1, self.ndim)
        n_points = len(points)
        # the emulators predict many points at once much faster than single
        # points, so the log-likelihood is computed in batches
        batch_size = 1000
        n_batches = -(-n_points // batch_size)
        log_every = max(n_batches // 10, 1)
        log_like = np.empty(n_points)
        for i, start in enumerate(range(0, n_points, batch_size)):
            if i % log_every == 0:
                logger.info(
                    "Evaluating the log-likelihood at points "
                    f"{start + 1}-{min(start + batch_size, n_points)}/{n_points} ..."
                )
            log_like[start : start + batch_size] = self.log_likelihood(
                points[start : start + batch_size]
            )
        # emcee/PTLMC chains have shape (n_walkers, n_steps, ndim),
        # pocoMC samples have shape (n_samples, ndim)
        log_like = log_like.reshape(self.chain.shape[:-1])

        logger.info(f"Writing the log-likelihood of the chain to {output_path}")
        _write_pickle(output_path, {"log_likelihood": log_like})

    def run_pocomc(
        self,
        n_effective=1000,
        n_active=250,
        n_prior=2000,
        sample="tpcn",
        n_max_steps=200,
        seed=None,
        n_total=5000,
        n_evidence=5000,
        n_ndim_steps=2,
        pool=None,
        prior=None,
    ):
        """
        Run preconditioned Monte Carlo with pocoMC.

        This function uses the pocoMC package (version 1.2.6, as required by
        the package). pocoMC is a Preconditioned Monte Carlo (PMC) sampler
        that uses normalizing flows to precondition the target distribution.

        The resampled posterior samples are stored in ``self.chain`` with
        shape (nsamples, ndim). They are written to ``chain_path("pocomc")``
        together with the log-likelihood (``"logl"``), the log prior
        (``"logp"``), the log evidence (``"logz"``) and its error
        (``"logz_err"``). Points outside the parameter ranges get the finite
        log-likelihood -1e300.

        Parameters
        ----------
        n_effective : int, default=1000
            Effective sample size maintained during the run.
        n_active : int, default=250
            Number of active particles. It should be smaller than
            `n_effective`.
        n_prior : int, default=2000
            Number of prior samples to draw (pocoMC's own default is
            ``2*(n_effective//n_active)*n_active``).
        sample : str, default="tpcn"
            Type of MCMC sampler to use. Options are ``"tpcn"``
            (t-preconditioned Crank-Nicolson) or ``"rwm"`` (random-walk
            Metropolis). t-preconditioned Crank-Nicolson is the recommended
            sampler for PMC, as it is more efficient and scales better with
            the number of parameters.
        n_max_steps : int, default=200
            Maximum number of MCMC steps per iteration (pocoMC's own default
            is ``10 * n_steps``, see `n_ndim_steps`).
        seed : int or None, default=None
            Random seed, passed to pocoMC as ``random_state``, which makes the
            run reproducible. pocoMC sets it as the seed of numpy's global
            random number generator (``np.random.seed``) and of torch, which
            also affects later code that uses these generators. None does not
            change the global random state.
        n_total : int, default=5000
            Total number of effectively independent samples to be collected.
        n_evidence : int, default=5000
            Number of importance samples used to estimate the evidence. If
            ``n_evidence=0``, the evidence is not estimated using importance
            sampling and the SMC estimate is used instead.
        n_ndim_steps : int, default=2
            Number of MCMC steps per parameter after the log-probability
            plateau, passed to pocoMC as ``n_steps = n_ndim_steps * ndim``. It
            controls the early stopping of the MCMC steps of each iteration.
        pool : int, pool object or None, default=None
            Parallelization of the likelihood evaluations. If None, the
            likelihood is evaluated for all particles at once (vectorized). If
            `pool` is an integer greater than 1, a ``multiprocess`` pool with
            this number of processes is created and closed after the run (1
            is the same as None); a
            pool object with a ``map`` method (e.g. of mpi4py) is used
            directly. With a pool, the likelihood is evaluated point by point
            in the processes of the pool.
        prior : object or None, default=None
            Prior distribution implementing the ``logpdf`` and ``rvs`` methods
            and the ``dim`` and ``bounds`` attributes. If None, a uniform
            prior within the parameter ranges is used. For more information on
            customizing the prior, see the pocoMC documentation.

        Raises
        ------
        ValueError
            If ``prior.dim`` does not match the dimension of the parameter
            space.

        Notes
        -----
        When experiencing issues with the fork() function, set the environment
        variable ``export RDMAV_FORK_SAFE=1``.
        """
        if prior is None:
            logger.info("Using a uniform prior for all parameters")
            prior_distributions = []
            for i in range(self.ndim):
                prior_distributions.append(
                    uniform(self.param_min[i], self.param_max[i] - self.param_min[i])
                )
            prior = pocomc.Prior(prior_distributions)
        else:
            logger.info("Using the given prior")
            # Check the dimensions of the prior
            if self.ndim != prior.dim:
                raise ValueError(
                    f"prior.dim = {prior.dim} does not match the {self.ndim} "
                    "model parameters"
                )

        self._warn_overwrite("pocomc")
        if isinstance(pool, int) and pool <= 1:
            # pocoMC creates a pool only for more than one process
            pool = None
        # pocoMC uses the pool only for a likelihood that is not vectorized
        vectorize = pool is None
        logger.info(
            f"Running pocoMC with n_effective={n_effective} "
            f"({'vectorized' if vectorize else 'with a pool'}) ..."
        )
        sampler = pocomc.Sampler(
            prior=prior,
            likelihood=self.log_likelihood if vectorize else self._log_likelihood_point,
            likelihood_kwargs={"finite": True},
            n_effective=n_effective,
            n_active=n_active,
            n_prior=n_prior,
            sample=sample,
            n_max_steps=n_max_steps,
            n_steps=n_ndim_steps * self.ndim,
            random_state=seed,
            vectorize=vectorize,
            pool=pool,
        )
        try:
            sampler.run(n_total=n_total, n_evidence=n_evidence)
        finally:
            # pocoMC creates a pool for an integer `pool` but does not close it
            if isinstance(pool, int) and sampler.pool is not None:
                sampler.pool.close()
                sampler.pool.join()

        samples, logl, logp = sampler.posterior(resample=True)
        logz, logz_err = sampler.evidence()
        # pocoMC gives no error of the evidence with n_evidence=0
        logz_err_str = "" if logz_err is None else f" +- {logz_err:.4f}"
        logger.info(
            f"pocoMC finished with {len(samples)} posterior samples, "
            f"log evidence = {logz:.4f}{logz_err_str}"
        )

        self.chain = samples
        self.chain_sampler = "pocomc"
        chain_data = {
            "chain": samples,
            "logl": logl,
            "logp": logp,
            "logz": logz,
            "logz_err": logz_err,
        }
        logger.info(f"Writing the pocoMC samples to {self.chain_path('pocomc')}")
        _write_pickle(self.chain_path("pocomc"), chain_data)
