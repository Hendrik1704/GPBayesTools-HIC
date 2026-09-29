"""
Bayesian model calibration with the emulators.

The `BayesianAnalysis` class provides the following samplers:

- ``run_emcee``: affine-invariant ensemble MCMC sampler emcee
- ``run_ptlmc``: parallel tempering Langevin Monte Carlo (PTLMC) from surmise
- ``run_pocomc``: preconditioned Monte Carlo with pocoMC (recommended)
"""

import logging
import pickle
from pathlib import Path

import emcee
import numpy as np
import pocomc
import scipy.optimize as spo
from scipy.linalg import lapack
from scipy.stats import uniform

from . import load_emulator, parse_model_parameter_file

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


class LoggingEnsembleSampler(emcee.EnsembleSampler):
    """
    Ensemble sampler of emcee that logs the progress of the run.

    The constructor parameters are those of `emcee.EnsembleSampler`.
    """

    def run_mcmc(self, X0, nsteps, status=None, **kwargs):
        """
        Run MCMC and log the acceptance fraction every `status` steps.

        Parameters
        ----------
        X0 : array_like of shape (nwalkers, ndim) or emcee.State
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
        logger.info("running %d walkers for %d steps", self.nwalkers, nsteps)

        if nsteps < 1:
            raise ValueError("nsteps must be >= 1")
        if status is None:
            status = max(nsteps // 10, 1)

        # the state of the last iteration is returned
        for n, result in enumerate(  # noqa: B007
            self.sample(X0, iterations=nsteps, **kwargs), start=1
        ):
            if n % status == 0 or n == nsteps:
                af = self.acceptance_fraction
                logger.info(
                    "step %d: acceptance fraction: "
                    "mean %.4f, std %.4f, min %.4f, max %.4f",
                    n,
                    af.mean(),
                    af.std(),
                    af.min(),
                    af.max(),
                )

        return result


class BayesianAnalysis:
    """
    High-level interface for running MCMC calibration and accessing results.

    Currently all design parameters except for the normalizations are required
    to be the same at all beam energies. It is assumed (NOT checked) that all
    system designs have the same parameters and ranges (except for the norms).

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
    expdata_path : str or path-like, default="./exp_data.dat"
        Path of the pickle file with the experimental data. It must contain a
        dictionary with exactly one data set, whose ``"obs"`` entry holds the
        values and the errors of the data points (see
        :meth:`_read_in_exp_data_pickle`).
    model_parafile : str or path-like, default="./model.dat"
        Path of the model parameter file with the label and the range
        (minimum and maximum) of each parameter.
    """

    samplers = ("emcee", "pocomc", "ptlmc")

    def __init__(
        self,
        mcmc_path="./mcmc/chain.pkl",
        expdata_path="./exp_data.dat",
        model_parafile="./model.dat",
    ):
        logger.info("Initializing MCMC ...")
        self.mcmc_path = Path(mcmc_path)
        self.mcmc_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Final Markov chain results will be saved in {}".format(
                ", ".join(str(self.chain_path(s)) for s in self.samplers)
            )
        )

        # load the model parameter file
        logger.info(f"Loading the model parameters space from {model_parafile} ...")
        self.pardict = parse_model_parameter_file(model_parafile)
        self.ndim = len(self.pardict.keys())
        self.label = []
        self.min = []
        self.max = []
        for val in self.pardict.values():
            self.label.append(val[0])
            self.min.append(val[1])
            self.max.append(val[2])
        self.min = np.array(self.min)
        self.max = np.array(self.max)

        # the volume of the uniform prior
        diff = self.max - self.min
        self.prior_volume = np.prod(diff)

        logger.info("Run MCMC with emcee...")
        # load the experimental data to be fit
        logger.info(f"Loading the experiment data from {expdata_path} ...")
        self.expdata, self.expdata_cov = self._read_in_exp_data_pickle(expdata_path)
        self.nobs = self.expdata.shape[1]
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
        logger.info(f"Number of Emulators: {len(self.emulators)}")

    def _predict(self, X):
        """
        Predict the mean and covariance of all observables at the points `X`.

        The predictions of the emulators are concatenated, and the covariance
        is block diagonal with one block per emulator. Returns arrays of shape
        (n, nobs) and (n, nobs, nobs), and raises a ValueError if the emulators
        do not predict `nobs` observables in total.
        """
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
        return np.all((X >= self.min) & (X <= self.max), axis=-1)

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
        """
        X = np.atleast_2d(np.asarray(X))
        lp = np.zeros(X.shape[0])
        inside = self._inside(X)
        if not finite:
            lp[~inside] = -np.inf
        elif finite:
            lp[~inside] = -1e300

        nsamples = np.count_nonzero(inside)
        if nsamples > 0:
            model_Y, model_cov = self._predict(X[inside])

            # allocate difference (model - experiment) and covariance arrays
            dY = np.empty([nsamples, self.nobs])
            cov = np.empty([nsamples, self.nobs, self.nobs])
            dY = model_Y - self.expdata
            # add experiment cov to model cov
            cov = model_cov + self.expdata_cov

            # compute log likelihood at each point
            lp[inside] += list(map(mvn_loglike, dY, cov))
        return lp

    def log_likelihood_point_by_point(self, X):
        """
        Evaluate the log-likelihood at `X` point by point.

        This is used to compute the log-likelihood for each point of an
        already generated chain. The progress is logged every 100 points.

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
        lp = np.zeros(X.shape[0])

        for k in range(X.shape[0]):
            if k % 100 == 0:
                logger.info(f"Evaluating log_likelihood at point {k}")
            Xk = np.atleast_2d(np.asarray(X[k]))
            inside = bool(self._inside(Xk)[0])
            lp[k] = -np.inf if not inside else 0.0

            nsamples = 1 if inside else 0
            if nsamples > 0:
                model_Y, model_cov = self._predict(Xk)

                # allocate difference (model - experiment) and covariance arrays
                dY = np.empty([nsamples, self.nobs])
                cov = np.empty([nsamples, self.nobs, self.nobs])
                dY = model_Y - self.expdata
                # add experiment cov to model cov
                cov = model_cov + self.expdata_cov

                # compute log likelihood at this point
                lp[k] += mvn_loglike(dY[0], cov[0])
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
        errors (NaN errors are set to 0). Returns the data of shape (1, nobs)
        and the covariance of shape (nobs, nobs).
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
        logger.info(f"Experimental dataset size: {model_data[0].shape[0]}")
        model_data = np.array(model_data)
        model_data_err = np.nan_to_num(np.abs(np.array(model_data_err)))
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
        return np.random.uniform(self.min, self.max, (n, self.ndim))

    @staticmethod
    def map(f, args):
        """
        Apply `f` to `args` in a single call.

        Dummy function so that this object can be used as a 'pool' for
        :class:`emcee.EnsembleSampler`, which then evaluates the vectorized
        log posterior once for all walkers.

        Parameters
        ----------
        f : callable
            Function to apply.
        args : object
            Argument passed to `f`.

        Returns
        -------
        object
            The result of ``f(args)``.
        """
        return f(args)

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

    def run_emcee(
        self,
        n_steps=500,
        n_burn_steps=None,
        n_walkers=None,
        status=None,
        n_thin=10,
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
        so far. The thinned chain is appended to the chain file and stored in
        ``self.chain`` with shape (n_walkers, n_steps, ndim).

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
        n_thin : int, default=10
            Thinning of the production chain, only every `n_thin`-th step is
            stored.
        skip_initial_state_check : bool, default=False
            Passed to emcee. If True, do not check that the initial walker
            positions are linearly independent.
        seed : int or None, default=None
            If given, numpy's global random number generator is seeded with it
            before the run, which makes the chain reproducible.

        Raises
        ------
        ValueError
            If `n_burn_steps` or `n_walkers` is missing for a new chain, if
            `n_burn_steps` is smaller than 2, if the existing chain was not
            generated with emcee, or if `n_walkers` does not match the existing
            chain.
        """
        if seed is not None:
            # emcee initializes its random number generator from numpy's
            # global state
            np.random.seed(seed)
        chain_file = self.chain_path("emcee")
        chain_data = {}
        try:
            with open(chain_file, "rb") as f:
                chain_data = pickle.load(f)
        except FileNotFoundError:
            pass

        if "chain" not in chain_data:
            burn_in = True
        else:
            burn_in = False

        if burn_in:
            if n_burn_steps is None or n_walkers is None:
                raise ValueError(
                    "must specify n_burn_steps and n_walkers to start chain"
                )
        else:
            # emcee chains have shape (n_walkers, n_steps, ndim), pocoMC samples
            # (nsamples, ndim)
            if chain_data["chain"].ndim != 3:
                raise ValueError(
                    f"the chain in {chain_file} was not generated with emcee and "
                    "cannot be continued, use a different mcmc_path"
                )
            if n_walkers is None:
                n_walkers = chain_data["chain"].shape[0]
            elif n_walkers != chain_data["chain"].shape[0]:
                raise ValueError(
                    "the existing chain has {} walkers, but n_walkers = {}".format(
                        chain_data["chain"].shape[0], n_walkers
                    )
                )

        logger.info("Starting MCMC ...")
        sampler = LoggingEnsembleSampler(
            n_walkers, self.ndim, self.log_posterior, pool=self
        )

        if burn_in:
            logger.info("no existing chain found, starting initial burn-in")
            if n_burn_steps < 2:
                raise ValueError(
                    "n_burn_steps must be >= 2, the burn-in is run in two halves"
                )

            # Run first half of burn-in starting from random positions.
            nburn0 = n_burn_steps // 2
            state = sampler.run_mcmc(
                self.random_pos(n_walkers),
                nburn0,
                status=status,
                skip_initial_state_check=skip_initial_state_check,
            )
            logger.info("resampling walker positions")
            # Reposition walkers to the most likely points in the chain,
            # then run the second half of burn-in.  This significantly
            # accelerates burn-in and helps prevent stuck walkers.
            lnprob = sampler.flatlnprobability
            # indices of the distinct log-probabilities in ascending order
            idx = np.unique(lnprob, return_index=True)[1]
            idx = idx[np.isfinite(lnprob[idx])]
            if len(idx) >= n_walkers:
                X0 = sampler.flatchain[idx[-n_walkers:]]
            else:
                logger.warning(
                    f"only {len(idx)} distinct points with finite probability in the "
                    "first half of the burn-in, continuing from the current "
                    "walker positions"
                )
                X0 = state.coords
            sampler.reset()
            X0 = sampler.run_mcmc(
                X0,
                n_burn_steps - nburn0,
                status=status,
                skip_initial_state_check=skip_initial_state_check,
            )
            sampler.reset()
            logger.info("burn-in complete, starting production")
        else:
            logger.info("restarting from last point of existing chain")
            # the last walker positions of the previous run, or for chains
            # saved with older versions, the last thinned sample
            X0 = chain_data.get("last_position", chain_data["chain"][:, -1, :])

        state = sampler.run_mcmc(
            X0,
            n_steps,
            status=status,
            skip_initial_state_check=skip_initial_state_check,
        )
        chain_data["last_position"] = state.coords

        thinned_chain = sampler.chain[:, ::n_thin, :]
        if "chain" in chain_data:
            chain_data["chain"] = np.concatenate(
                (chain_data["chain"], thinned_chain), axis=1
            )
            self.chain = chain_data["chain"]
        else:
            chain_data["chain"] = thinned_chain
            self.chain = thinned_chain

        self.chain_sampler = "emcee"

        # Append the new data to the existing file
        logger.info(f"writing chain to {chain_file}")
        with open(chain_file, "wb") as file:
            pickle.dump(chain_data, file)

    # This function is taken from the surmise package (version 1.0.0) and
    # modified: the number of initial draws is set by nstartparameters, the
    # tuning phase is longer (fractunning = 2), progress is logged, and the
    # unflattened chains of the temperature-1 walkers are returned.
    def _sampler_ptlmc(
        self,
        logpostfunc,
        draw_func,
        rng,
        theta0=None,
        numtemps=32,
        numchain=16,
        sampperchain=400,
        maxtemp=30,
        nstartparameters=1000,
    ):
        """
        Run parallel-tempering ensemble MCMC based on Langevin Monte Carlo.

        Before sampling, the starting points are optimized with L-BFGS-B and
        then moved slightly off the optima. The first ``2 * sampperchain``
        steps tune the step size and are discarded.

        Parameters
        ----------
        logpostfunc : callable
            Function that evaluates the log of the posterior distribution.
            Without gradient, it takes an m by p numpy array of parameters
            and returns a length m numpy array of log posterior evaluations.
            With gradient, it returns a tuple whose first element is as above
            and whose second element is an m by p array of gradients of the
            log posterior.
        draw_func : callable
            Function that produces approximate draws from the distribution,
            ``draw_func(n)`` returns an n by p array. Used to initialize the
            points.
        rng : numpy.random.Generator
            Random number generator used for all random numbers of the sampler.
        theta0 : ndarray of shape (n, p) or None, default=None
            A long list of parameters to start from. If None or with fewer
            than ``max(numtemps + numchain, 10 * ndim)`` points, the points
            are drawn with `draw_func`.
        numtemps : int, default=32
            Number of chains of varying temperature to run simultaneously.
        numchain : int, default=16
            Number of chains of temperature 1 to run simultaneously.
        sampperchain : int, default=400
            Number of samples saved for each chain.
        maxtemp : float, default=30
            Maximum temperature used in parallel tempering, larger than 1.
        nstartparameters : int, default=1000
            Number of initial draws from `draw_func` if `theta0` is not given
            or too small.

        Returns
        -------
        dict
            Dictionary with the samples of the temperature-1 chains in the
            key ``'theta'``, with shape (numchain, sampperchain, p).

        Raises
        ------
        ValueError
            If `logpostfunc` returns a tuple with more than 2 elements or a
            gradient of the wrong shape.
        """
        # Need at least one starting point per chain and enough points to
        # estimate their spread. If we do not get enough parameters to start,
        # draw at least nstartparameters.
        nmin = max(numtemps + numchain, 10 * self.ndim)
        if theta0 is None or theta0.shape[0] < nmin:
            theta0 = draw_func(max(nstartparameters, nmin))
        # Setting up some default parameters
        fractunning = 2.0  # samples spent tuning, relative to sampperchain
        # define the number of samples for tuning
        samptunning = np.ceil(sampperchain * fractunning).astype("int")
        # defining the total number of chains
        totnumchain = numtemps + numchain
        # space out the temperature vector to go from maxtemp to 1, and then
        # repeat 1 for the number of non-tempered chains
        temps = np.concatenate(
            (
                np.exp(
                    np.linspace(
                        np.log(maxtemp), np.log(maxtemp) / (numtemps + 1), numtemps
                    )
                ),
                np.ones(numchain),
            )
        )  # ratio idea tend from emcee
        tempsc = temps[:, np.newaxis]  # for broadcasting against (chain, p) arrays

        # number of optimization at each chain before starting
        numopt = temps.shape[0]
        # before beginning, let's test out the given logpdf function
        testout = logpostfunc(theta0[0:2, :])
        if type(testout) is tuple:
            if len(testout) > 2:
                raise ValueError("log density does not return 1 or 2 elements")
            if testout[1].shape[1] != theta0.shape[1]:
                raise ValueError("derivative appears to be the wrong shape")

            def logpostf(thetain):  # canonical shapes: (m,) and (m, p)
                f, df = logpostfunc(thetain)
                f = np.asarray(f, dtype=float).ravel()
                df = np.asarray(df, dtype=float).reshape(f.shape[0], -1)
                return f, df

            def logpostf_grad(thetain):
                return logpostf(thetain)[1]

            try:
                testout = logpostfunc(theta0[10, :], return_grad=False)
                if (
                    type(testout) is tuple
                ):  # make sure that return_grad functionality works
                    raise ValueError("Cannot stop returning a grad")

                def logpostf_nograd(theta):
                    return np.asarray(
                        logpostfunc(theta, return_grad=False), dtype=float
                    ).ravel()
            except Exception:

                def logpostf_nograd(theta):  # if not, do not use return_grad key
                    return np.asarray(logpostfunc(theta)[0], dtype=float).ravel()
        else:
            logpostf_grad = None  # sometimes no derivative is given

            def logpostf_nograd(theta):
                return np.asarray(logpostfunc(theta), dtype=float).ravel()

            logpostf = logpostf_nograd

        if logpostf_grad is None:  # standard target acceptance rates
            taracc = 0.25  # close to theoretical result 0.234
        else:
            taracc = 0.60  # close to theoretical result in LMC paper
        # begin preoptimizer
        logger.info("Begin PTLMC pre-optimization ...")
        # order the existing initial theta's by log pdf
        ord1 = np.argsort(
            -logpostf_nograd(theta0)
            + (theta0.shape[1] * rng.standard_normal(size=theta0.shape[0]) ** 2)
        )
        theta0 = theta0[ord1[0:totnumchain], :]
        # begin optimizing at each chain
        thetacen = np.mean(theta0, 0)
        thetas = np.maximum(np.std(theta0, 0), 10 ** (-8) * np.std(theta0))

        # rescale the input to make it easier to optimize
        def neglogpostf_nograd(thetap):
            theta = thetacen + thetas * thetap
            return -logpostf_nograd(theta.reshape((1, len(theta))))[0]

        if logpostf_grad is not None:

            def neglogpostf_grad(thetap):
                theta = thetacen + thetas * thetap
                return -thetas * logpostf_grad(theta.reshape((1, len(theta)))).ravel()

        boundL = np.maximum(
            -10 * np.ones(theta0.shape[1]), np.min((theta0 - thetacen) / thetas, 0)
        )
        boundU = np.minimum(
            10 * np.ones(theta0.shape[1]), np.max((theta0 - thetacen) / thetas, 0)
        )
        bounds = spo.Bounds(boundL, boundU)
        thetaop = theta0
        # now we are ready to optimize for each chain
        logger.info("Begin PTLMC chain optimization ...")
        for k in range(0, numopt):
            if k % 10 == 0:
                logger.info(f"Currently working on optimization of k = {k}")
            if logpostf_grad is None:
                opval = spo.minimize(
                    neglogpostf_nograd,
                    (thetaop[k, :] - thetacen) / thetas,
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                thetaop[k, :] = thetacen + thetas * opval.x
            else:
                opval = spo.minimize(
                    neglogpostf_nograd,
                    (thetaop[k, :] - thetacen) / thetas,
                    method="L-BFGS-B",
                    jac=neglogpostf_grad,
                    bounds=bounds,
                )
                thetaop[k, :] = thetacen + thetas * opval.x
            # use these as starting locations
            # try to move off optimized value to stop it from devolving
            W, V = np.linalg.eigh(opval.hess_inv @ np.eye(thetacen.shape[0]))
            notmoved = True
            if k == 0:
                notmoved = False
            stepadj = 4
            l0 = neglogpostf_nograd(opval.x)
            while notmoved:
                if (W > 0).all():
                    r = (V.T * np.sqrt(W)) @ (
                        V @ rng.standard_normal(size=thetacen.shape[0])
                    )
                else:
                    stepadj /= 2
                    if stepadj < 1 / 16:
                        thetaop[k, :] = thetacen + thetas * opval.x
                        notmoved = False
                    continue

                if (
                    neglogpostf_nograd(stepadj * r + opval.x) - l0
                ) < 3 * thetacen.shape[0]:
                    thetaop[k, :] = thetacen + thetas * (stepadj * r + opval.x)
                    notmoved = False
                else:
                    stepadj /= 2
        # end preoptimizer
        # initialize the starting point
        logger.info("Initialize PTLMC starting point ...")
        thetac = thetaop
        if logpostf_grad is not None:
            fval, dfval = logpostf(thetac)
            fval = fval / temps
            dfval = dfval / tempsc
        else:
            fval = logpostf_nograd(thetac) / temps

        # preallocate the saving matrix
        thetasave = np.zeros((numchain, sampperchain, thetac.shape[1]))
        # try to start the covariance matrix
        covmat0 = np.cov(thetac.T)
        if thetac.shape[1] > 1:
            covmat0 = 0.9 * covmat0 + 0.1 * np.diag(
                np.diag(covmat0)
            )  # add a diagonal part to prevent any non-moving issues
            W, V = np.linalg.eigh(covmat0)
            hc = V @ np.diag(np.sqrt(W)) @ V.T
        else:
            hc = np.sqrt(covmat0)
            hc = hc.reshape(1, 1)
            covmat0 = covmat0.reshape(1, 1)
        # parameter initialization
        tau = -1
        rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
        adjrho = rho * temps ** (
            1 / 3
        )  # this adjusts rho across different temperatures
        adjrhoc = adjrho[:, np.newaxis]
        numtimes = 0  # accumulated acceptance rate, reset after each tuning update
        logger.info("Run over all PTLMC chains and tune ...")
        for k in range(0, samptunning + sampperchain):  # loop over all chains
            if k % 100 == 0:
                logger.info(f"Currently working on {k}")
            rvalo = rng.standard_normal(size=thetac.shape)
            rval = (np.sqrt(2) * adjrho * np.squeeze(rvalo @ hc).T).T
            if thetac.shape[1] > 1:
                thetap = thetac + rval
            elif thetac.shape[1] == 1:
                thetap = thetac + rval[:, np.newaxis]
            if logpostf_grad is not None:
                # calculate the elements to move if there is a gradient
                diffval = (adjrhoc**2) * (dfval @ covmat0)
                thetap += diffval
                fvalp, dfvalp = logpostf(thetap)  # thetap : no chain x dimension
                fvalp = fvalp / temps  # to flatten the posterior
                dfvalp = dfvalp / tempsc
                term1 = rvalo / np.sqrt(2)
                term2 = (adjrhoc / 2) * ((dfval + dfvalp) @ hc)
                qadj = -(2 * np.sum(term1 * term2, 1) + np.sum(term2**2, 1))
            else:
                # calculate the elements to move if there is no gradient
                fvalp = logpostf_nograd(thetap) / temps  # thetap : no chain x dimension
                qadj = np.zeros(fvalp.shape)
            swaprnd = np.log(rng.uniform(size=fval.shape[0]))
            whereswap = np.where(
                np.squeeze(swaprnd) < np.squeeze(fvalp - fval) + np.squeeze(qadj)
            )[0]  # MH step to find which of the chains to swap
            if whereswap.shape[0] > 0:  # if we swap, do it where needed
                numtimes = numtimes + np.sum(whereswap > -1) / totnumchain
                thetac[whereswap] = np.copy(thetap[whereswap])
                fval[whereswap] = np.copy(fvalp[whereswap])
                if logpostf_grad is not None:
                    dfval[whereswap] = np.copy(dfvalp[whereswap])
            # do some swaps along the temperatures
            fvaln = fval * temps
            # go through 5 times, swapping where needed
            orderprop = self._temp_exchange(fvaln, temps, iters=5, rng=rng)
            fval = fvaln[orderprop] / temps
            thetac = thetac[orderprop, :]
            if logpostf_grad is not None:
                dfvaln = tempsc * dfval
                dfval = (1 / tempsc) * dfvaln[orderprop, :]
            # if we have to tune, move tau up or down, which gives bigger or
            # smaller jumps
            if (k < samptunning) and (k % 10 == 0):  # if not done with tuning
                tau = tau + 1 / np.sqrt(1 + k / 10) * ((numtimes / 10) - taracc)
                rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
                adjrho = rho * (temps ** (1 / 3))  # adjusting rho across the chain
                adjrhoc = adjrho[:, np.newaxis]
                numtimes = 0
            elif k >= samptunning:  # if done with tuning
                thetasave[:, k - samptunning, :] = 1 * thetac[numtemps:,]
        # return the unflattened values of the temp=1 chains
        sampler_info = {"theta": thetasave}
        return sampler_info

    # This function is taken from the surmise package (version 1.0.0) and
    # modified to skip swaps between chains with the same temperature
    def _temp_exchange(self, lpostf, temps, iters=1, rng=None):
        """
        Propose swaps of the chains between neighboring temperatures.

        Given the log pdf values `lpostf` of the chains and their temperatures
        `temps`, random swaps are proposed `iters` times and accepted with the
        parallel tempering rule. Returns the (random) revised order of the
        chains. `rng` (a numpy.random.Generator) is required.
        """
        assert rng is not None

        order = np.arange(0, lpostf.shape[0])  # initializing
        for _ in range(iters):
            # choose random values to check for swapping
            rtv = rng.choice(range(1, lpostf.shape[0]), lpostf.shape[0])
            for rt in rtv:
                rhoh = 1 / temps[rt - 1] - 1 / temps[rt]
                if rhoh == 0:
                    # chains with the same temperature (e.g. the temperature-1
                    # chains) would always be swapped, which only mixes up the
                    # walkers without changing the sampled distribution
                    continue
                if (lpostf[order[rt]] - lpostf[order[rt - 1]]) * rhoh > np.log(
                    rng.uniform()
                ):  # swap via the PT rule
                    temporder = order[rt - 1]
                    order[rt - 1] = 1 * order[rt]
                    order[rt] = 1 * temporder
        return order

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

        This function wraps the PTLMC sampler (adapted from surmise, see
        :meth:`_sampler_ptlmc`). The initial points are drawn uniformly within
        the parameter ranges. The samples of the temperature-1 chains are
        stored in ``self.chain`` with shape (n_walkers, n_steps, ndim) and
        written to ``chain_path("ptlmc")``, overwriting an existing file.

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
            return rng.uniform(self.min, self.max, (n, self.ndim))

        logger.info("Starting MCMC ...")
        result_dict = self._sampler_ptlmc(
            logpostfunc=self.log_posterior,
            draw_func=draw_func,
            rng=rng,
            theta0=None,
            numtemps=n_temps,
            numchain=n_walkers,
            sampperchain=n_steps,
            maxtemp=max_temp,
            nstartparameters=n_start_parameters,
        )

        self.chain = result_dict["theta"]
        # This reshape should not be necessary, it is just done to match the
        # format of the other MCMC samplers
        self.chain = self.chain.reshape((n_walkers, n_steps, self.ndim))

        self.chain_sampler = "ptlmc"

        # Write the chain to file (n_walkers, n_steps, self.ndim)
        chain_data["chain"] = self.chain
        logger.info("Writing MCMC chains to {}".format(self.chain_path("ptlmc")))
        with open(self.chain_path("ptlmc"), "wb") as file:
            pickle.dump(chain_data, file)

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
            If `sampler` is None and no chain has been run with this object.
        """
        if sampler is not None:
            logger.info(f"Loading chain from {self.chain_path(sampler)}")
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
        logger.info("Computing log likelihood for the chain...")
        reshape_chain = self.chain.reshape(-1, self.ndim)
        likelihood = self.log_likelihood_point_by_point(reshape_chain)
        # emcee/PTLMC chains have shape (nwalkers, nsteps, ndim),
        # pocoMC samples have shape (nsamples, ndim)
        likelihood = likelihood.reshape(self.chain.shape[:-1])

        # Write the log_likelihood to file
        logger.info(f"Writing log_likelihood for chains to {output_path}")
        likelihood_data = {"log_likelihood": likelihood}
        with open(output_path, "wb") as file:
            pickle.dump(likelihood_data, file)

    def run_pocomc(
        self,
        n_effective=1000,
        n_active=250,
        n_prior=2000,
        sample="tpcn",
        n_max_steps=200,
        random_state=42,
        n_total=5000,
        n_evidence=5000,
        n_ndim_steps=2,
        pool=None,
        prior=None,
    ):
        """
        Run preconditioned Monte Carlo with pocoMC.

        This function is based on the pocoMC package (version 1.2.6). It works
        with versions of pocoMC >= 1.2.2 and is tested up to 1.2.6. pocoMC is
        a Preconditioned Monte Carlo (PMC) sampler that uses normalizing flows
        to precondition the target distribution.

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
            Number of active particles. It must be smaller than `n_effective`.
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
            Maximum number of MCMC steps (pocoMC's own default is
            ``10*n_dim``).
        random_state : int or None, default=42
            Initial random seed.
        n_total : int, default=5000
            Total number of effectively independent samples to be collected.
        n_evidence : int, default=5000
            Number of importance samples used to estimate the evidence. If
            ``n_evidence=0``, the evidence is not estimated using importance
            sampling and the SMC estimate is used instead. If
            ``preconditioned=False``, the evidence is estimated using SMC and
            `n_evidence` is ignored.
        n_ndim_steps : int, default=2
            Number of MCMC steps in beta per dimension, pocoMC runs
            ``n_ndim_steps * ndim`` steps.
        pool : int or None, default=None
            Number of processes to use for parallelization. If `pool` is an
            integer greater than 1, a ``multiprocessing`` pool is created with
            the specified number of processes.
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
        logger.info("Generate the prior class for pocoMC ...")
        if prior is None:
            logger.info("Using uniform prior for all parameters ...")
            prior_distributions = []
            for i in range(self.ndim):
                prior_distributions.append(
                    uniform(self.min[i], self.max[i] - self.min[i])
                )
            prior = pocomc.Prior(prior_distributions)
        else:
            logger.info("Using custom prior ...")
            # Check the dimensions of the prior
            if self.ndim != prior.dim:
                logger.error("prior.dim does not match the model parameter space")
                raise ValueError("prior.dim does not match the model parameter space")

        logger.info("Starting pocoMC ...")
        sampler = pocomc.Sampler(
            prior=prior,
            likelihood=self.log_likelihood,
            likelihood_kwargs={"finite": True},
            n_effective=n_effective,
            n_active=n_active,
            n_prior=n_prior,
            sample=sample,
            n_max_steps=n_max_steps,
            n_steps=n_ndim_steps * self.ndim,
            random_state=random_state,
            vectorize=True,
            pool=pool,
        )
        sampler.run(n_total=n_total, n_evidence=n_evidence)

        logger.info("Generate the posterior samples ...")
        samples, logl, logp = sampler.posterior(resample=True)

        logger.info("Generate the evidence ...")
        logz, logz_err = sampler.evidence()
        logger.info(f"Log evidence: {logz}")
        logger.info(f"Log evidence error: {logz_err}")

        self.chain = samples
        self.chain_sampler = "pocomc"
        chain_data = {
            "chain": samples,
            "logl": logl,
            "logp": logp,
            "logz": logz,
            "logz_err": logz_err,
        }
        logger.info("Writing pocoMC chains to {}".format(self.chain_path("pocomc")))
        with open(self.chain_path("pocomc"), "wb") as file:
            pickle.dump(chain_data, file)
