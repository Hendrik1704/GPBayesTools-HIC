"""
Markov chain Monte Carlo model calibration. The following methods are available
- run_mcmc: run MCMC model calibration with emcee
- run_MCMC_ptemcee: run MCMC model calibration with parallel tempering ptemcee
- run_MCMC_PTLMC: run MCMC model calibration with PTLMC sampler
- run_pocoMC: run MCMC model calibration with pocoMC sampler (recommended)
"""
import logging
import pickle

from pathlib import Path
import emcee
import numpy as np
from scipy.linalg import lapack
import dill

from . import workdir, parse_model_parameter_file
import scipy.optimize as spo
import pocomc
from scipy.stats import uniform


def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.  The
    normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info > 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )

    return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()


class LoggingEnsembleSampler(emcee.EnsembleSampler):
    def run_mcmc(self, X0, nsteps, status=None, **kwargs):
        """
        Run MCMC with logging every 'status' steps (default: approx 10% of
        nsteps).

        """
        logging.info('running %d walkers for %d steps', self.nwalkers, nsteps)

        if nsteps < 1:
            raise ValueError('nsteps must be >= 1')
        if status is None:
            status = max(nsteps // 10, 1)

        for n, result in enumerate(
                self.sample(X0, iterations=nsteps, **kwargs),
                start=1
        ):
            if n % status == 0 or n == nsteps:
                af = self.acceptance_fraction
                logging.info(
                    'step %d: acceptance fraction: '
                    'mean %.4f, std %.4f, min %.4f, max %.4f',
                    n, af.mean(), af.std(), af.min(), af.max()
                )

        return result


class Chain:
    """
    High-level interface for running MCMC calibration and accessing results.

    Currently all design parameters except for the normalizations are required
    to be the same at all beam energies.  It is assumed (NOT checked) that all
    system designs have the same parameters and ranges (except for the norms).

    The experimental data are used as they are given. For emulators that
    return predictions in log space (``logTrafo=True`` and
    ``exp_and_cov_diagonal=False``), the experimental data must be
    log-transformed by the user as well.

    Each sampler writes its chain to its own file, which is derived from
    `mcmc_path` by adding the name of the sampler, e.g. for the default
    ``./mcmc/chain.pkl``: ``./mcmc/chain_emcee.pkl``,
    ``./mcmc/chain_pocoMC.pkl`` and ``./mcmc/chain_PTLMC.pkl``
    (see :meth:`chain_path`).

    """
    samplers = ('emcee', 'pocoMC', 'PTLMC')

    def __init__(self, mcmc_path="./mcmc/chain.pkl",
                 expdata_path="./exp_data.dat",
                 model_parafile="./model.dat"
    ):
        logging.info('Initializing MCMC ...')
        self.mcmc_path = Path(mcmc_path)
        self.mcmc_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info('Final Markov Chain results will be saved in {}'.format(
            ', '.join(str(self.chain_path(s)) for s in self.samplers))
        )

        # load the model parameter file
        logging.info('Loading the model parameters space from {} ...'.format(
            model_parafile)
        )
        self.pardict = parse_model_parameter_file(model_parafile)
        self.ndim = len(self.pardict.keys())
        self.label = []
        self.min = []
        self.max = []
        for par, val in self.pardict.items():
            self.label.append(val[0])
            self.min.append(val[1])
            self.max.append(val[2])
        self.min = np.array(self.min)
        self.max = np.array(self.max)

        #the volume of the uniform prior
        diff =  self.max - self.min
        self.prior_volume_ = np.prod( diff )

        logging.info("Run MCMC with emcee...")
        # load the experimental data to be fit
        logging.info(
            'Loading the experiment data from {} ...'.format(expdata_path))
        self.expdata, self.expdata_cov = self._read_in_exp_data_pickle(expdata_path)
        self.nobs = self.expdata.shape[1]
        self.emuList = []
        self.chain = False
        # sampler that generated self.chain
        self.chain_sampler = None


    def loadEmulator(self, emulatorPathList):
        """
        Load the emulators from the files in `emulatorPathList`, replacing
        previously loaded emulators. The order of the emulators must be the
        order of the observables in the experimental data, and their numbers
        of observables must add up to the number of experimental data points.
        """
        emuList = []
        for emuPath in emulatorPathList:
            with open(emuPath, 'rb') as f:
                emuList.append(dill.load(f))
        nobs_emu = [emu.nobs for emu in emuList]
        if sum(nobs_emu) != self.nobs:
            raise ValueError(
                'The emulators have {} observables in total ({}), but the '
                'experimental data have {} data points'.format(
                    sum(nobs_emu), ', '.join(map(str, nobs_emu)), self.nobs))
        self.emuList = emuList
        logging.info("Number of Emulators: {}".format(len(self.emuList)))


    def _predict(self, X):
        nPreds = X.shape[0]
        modelPred = np.zeros([nPreds, self.nobs])
        modelPredCov = np.zeros([nPreds, self.nobs, self.nobs])
        currIdx = 0
        for i, emu_i in enumerate(self.emuList):
            model_Y, model_cov = emu_i.predict(X, return_cov=True)
            nobs_i = model_Y.shape[1]
            modelPred[:, currIdx:currIdx+nobs_i] = model_Y
            modelPredCov[:, currIdx:currIdx+nobs_i, currIdx:currIdx+nobs_i] = model_cov
            currIdx += nobs_i
        if currIdx != self.nobs:
            raise ValueError(
                'The emulators predict {} observables, but the experimental '
                'data have {} data points'.format(currIdx, self.nobs))
        return modelPred, modelPredCov


    def _inside(self, X):
        """True for the points in X inside the parameter ranges (including
        the boundaries, as the uniform prior of pocoMC)."""
        return np.all((X >= self.min) & (X <= self.max), axis=-1)


    def log_prior(self, X):
        """
        Evaluate the (normalized, uniform) prior at `X`.

        """
        X = np.atleast_2d(np.asarray(X))
        lp = np.log( np.ones(X.shape[0]) / self.prior_volume_ )
        lp[~self._inside(X)] = -np.inf
        return lp


    def log_likelihood(self, X, finite=False):
        """
        Evaluate the likelihood at `X`.
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
        Evaluate the likelihood at `X` point by point.
        This is used for the log_likelihood computation when the chain is already
        generated and the likelihood is computed for each point in the chain.
        """
        lp = np.zeros(X.shape[0])
        
        for k in range(X.shape[0]):
            if k % 100 == 0:
                logging.info("Evaluating log_likelihood at point {}".format(k))
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
        Evaluate the posterior at `X`, the sum of the log prior and the log
        likelihood.
        """
        return self.log_prior(X) + self.log_likelihood(X)


    def _read_in_exp_data_pickle(self, filepath):
        """This function reads in exp data and compute the covariance matrix"""
        model_data = []
        model_data_err = []
        
        with open(filepath, "rb") as fp:
            dataDict = pickle.load(fp)
        if len(dataDict) != 1:
            raise ValueError(
                'The experimental data file {} must contain exactly one data '
                'set, but contains {}'.format(filepath, len(dataDict)))

        for event_id in dataDict.keys():
            temp_data = dataDict[event_id]["obs"].transpose()
            model_data.append(temp_data[:, 0])
            model_data_err.append(temp_data[:, 1])
        logging.info("Experimental dataset size: {}".format(model_data[0].shape[0]))
        model_data = np.array(model_data)
        model_data_err = np.nan_to_num(
                np.abs(np.array(model_data_err)))
        nobs = model_data.shape[1]
        
        data_cov = np.zeros((nobs, nobs))
        model_data_err = model_data_err.flatten()
        np.fill_diagonal(data_cov, (model_data_err)** 2)
      
        return model_data, data_cov


    def random_pos(self, n=1):
        """
        Generate `n` random positions in parameter space.

        """
        return np.random.uniform(self.min, self.max, (n, self.ndim))


    @staticmethod
    def map(f, args):
        """
        Dummy function so that this object can be used as a 'pool' for
        :meth:`emcee.EnsembleSampler`.

        """
        return f(args)


    def chain_path(self, sampler):
        """
        Path of the chain file of `sampler` ('emcee', 'pocoMC' or 'PTLMC').
        """
        if sampler not in self.samplers:
            raise ValueError("Unknown sampler '{}', use one of {}".format(
                sampler, self.samplers))
        return self.mcmc_path.with_name('{}_{}{}'.format(
            self.mcmc_path.stem, sampler, self.mcmc_path.suffix))


    def run_mcmc(self, nsteps=500, nburnsteps=None, nwalkers=None,
                 status=None, nthin=10, skip_initial_state_check=False,
                 seed=None):
        """
        Markov chain Monte Carlo model calibration using the `affine-invariant 
        ensemble sampler (emcee) <http://dfm.io/emcee>`_.
        
        Run MCMC model calibration. If the chain already exists, continue from
        the last point, otherwise burn-in and start the chain.

        If `seed` is given, numpy's global random number generator is seeded
        with it before the run, which makes the chain reproducible.
        """
        if seed is not None:
            # emcee initializes its random number generator from numpy's
            # global state
            np.random.seed(seed)
        chain_file = self.chain_path('emcee')
        chain_data = {}
        try:
            with open(chain_file, 'rb') as f:
                chain_data = pickle.load(f)
        except FileNotFoundError:
            pass

        if 'chain' not in chain_data:
            burnFlag = True
        else:
            burnFlag = False

        if burnFlag:
            if nburnsteps is None or nwalkers is None:
                raise ValueError(
                    'must specify nburnsteps and nwalkers to start chain')
        else:
            # emcee chains have shape (nwalkers, nsteps, ndim), pocoMC samples
            # (nsamples, ndim)
            if chain_data['chain'].ndim != 3:
                raise ValueError(
                    'the chain in {} was not generated with emcee and cannot '
                    'be continued, use a different mcmc_path'.format(
                        chain_file))
            if nwalkers is None:
                nwalkers = chain_data['chain'].shape[0]
            elif nwalkers != chain_data['chain'].shape[0]:
                raise ValueError(
                    'the existing chain has {} walkers, but nwalkers = {}'
                    .format(chain_data['chain'].shape[0], nwalkers))

        logging.info('Starting MCMC ...')
        sampler = LoggingEnsembleSampler(
            nwalkers, self.ndim, self.log_posterior, pool=self
        )

        if burnFlag:
            logging.info(
                    'no existing chain found, starting initial burn-in')
            if nburnsteps < 2:
                raise ValueError('nburnsteps must be >= 2, the burn-in is '
                                 'run in two halves')

            # Run first half of burn-in starting from random positions.
            nburn0 = nburnsteps // 2
            state = sampler.run_mcmc(
                self.random_pos(nwalkers),
                nburn0,
                status=status,
                skip_initial_state_check=skip_initial_state_check
            )
            logging.info('resampling walker positions')
            # Reposition walkers to the most likely points in the chain,
            # then run the second half of burn-in.  This significantly
            # accelerates burn-in and helps prevent stuck walkers.
            lnprob = sampler.flatlnprobability
            # indices of the distinct log-probabilities in ascending order
            idx = np.unique(lnprob, return_index=True)[1]
            idx = idx[np.isfinite(lnprob[idx])]
            if len(idx) >= nwalkers:
                X0 = sampler.flatchain[idx[-nwalkers:]]
            else:
                logging.warning(
                    'only {} distinct points with finite probability in the '
                    'first half of the burn-in, continuing from the current '
                    'walker positions'.format(len(idx)))
                X0 = state.coords
            sampler.reset()
            X0 = sampler.run_mcmc(
                X0,
                nburnsteps - nburn0,
                status=status,
                skip_initial_state_check=skip_initial_state_check
            )
            sampler.reset()
            logging.info('burn-in complete, starting production')
        else:
            logging.info('restarting from last point of existing chain')
            # the last walker positions of the previous run, or for chains
            # saved with older versions, the last thinned sample
            X0 = chain_data.get('last_position', chain_data['chain'][:, -1, :])

        state = sampler.run_mcmc(X0, nsteps, status=status,
                                 skip_initial_state_check=skip_initial_state_check)
        chain_data['last_position'] = state.coords

        thinedChain = sampler.chain[:, ::nthin, :]
        if 'chain' in chain_data:
            chain_data['chain'] = np.concatenate((chain_data['chain'], 
                                                  thinedChain), axis=1)
            self.chain = chain_data['chain']
        else:
            chain_data['chain'] = thinedChain
            self.chain = thinedChain

        self.chain_sampler = 'emcee'

        # Append the new data to the existing file
        logging.info('writing chain to {}'.format(chain_file))
        with open(chain_file, 'wb') as file:
            pickle.dump(chain_data, file)


    # This function is taken from the surmise package (version 1.0.0) and
    # modified: the number of initial draws is set by nstartparameters, the
    # tuning phase is longer (fractunning = 2), progress is logged, and the
    # unflattened chains of the temperature-1 walkers are returned.
    def samplerPTLMC(self, logpostfunc,
                     draw_func,
                     rng,
                     theta0=None,
                     numtemps=32,
                     numchain=16,
                     sampperchain=400,
                     maxtemp=30,
                     nstartparameters=1000):
        """
        Parallel-Tempering Ensemble MCMC based on Langevin Monte Carlo.

        Parameters
        ----------
        logpostfunc : function
            A function call describing the log of the posterior distribution.
                If no gradient, logpostfunc should take a value of an m by p numpy
                array of parameters and theta and return
                a length m numpy array of log posterior evaluations.
                If gradient, logpostfunc should return a tuple.  The first element
                in the tuple should be as listed above.
                The second element in the tuple should be an m by p matrix of
                gradients of the log posterior.
        draw_func : function, required
            A function that produces approximate draws from the distribution.  Can be used to initialize points.
        rng : numpy.random.Generator
            Random number generator used for all random numbers of the sampler.
        theta0 : n by p numpy array, optional
            This should contain a long list of original parameters to start from. The default is None.
        numtemps : integer, optional
            A positive integer that controls how many chains of varying temperature to run simultaneously. The default is
            32.
        numchain : integer, optional
            A positive integer that controls how many chains of fixed temperature to run simultaneously. The default is 16.
        sampperchain : integer, optional
            A positive integer that controls how many samples should be done for each chain. The default is 400.
        maxtemp : double, optional
            A positive number, larger than 1, that gives the maximum temperature used in parallel tempering. The default
            is 30.
        nstartparameters : integer, optional
            Number of initial draws from draw_func if theta0 is not given or
            too small. The default is 1000.

        Raises
        ------
        ValueError
            Indicates that something was not entered right, please check documentation.

        Returns
        -------
        dictionary
            A dictionary that contains the sampled values of the temperature-1
            chains in the key 'theta', with shape (numchain, sampperchain, p).
        """
        # Need at least one starting point per chain and enough points to
        # estimate their spread. If we do not get enough parameters to start,
        # draw at least nstartparameters
        nmin = max(numtemps + numchain, 10 * self.ndim)
        if theta0 is None or theta0.shape[0] < nmin:
            theta0 = draw_func(max(nstartparameters, nmin))
        # Setting up some default parameters
        fractunning = 2.0  # number of samples spent tunning the sampler
        # define the number of samples for tunning
        samptunning = np.ceil(sampperchain*fractunning).astype('int')
        # defining the total number of chains
        totnumchain = numtemps+numchain
        # spacing out the temperature vector to go from maxtemp to 1, and  then replacating 1 the number of
        # non-temperatured chains
        temps = np.concatenate((np.exp(np.linspace(np.log(maxtemp),
                                                   np.log(maxtemp)/(numtemps+1),
                                                   numtemps)),
                                np.ones(numchain)))  # ratio idea tend from emcee
        tempsc = temps[:, np.newaxis]  # for broadcasting against (chain, p) arrays

        # number of optimization at each chain before starting
        numopt = temps.shape[0]
        # before beginning, let's test out the given logpdf function
        testout = logpostfunc(theta0[0:2, :])
        if type(testout) is tuple:
            if len(testout) > 2:
                raise ValueError('log density does not return 1 or 2 elements')
            if testout[1].shape[1] != theta0.shape[1]:
                raise ValueError('derivative appears to be the wrong shape')

            def logpostf(thetain):  # canonical shapes: (m,) and (m, p)
                f, df = logpostfunc(thetain)
                f = np.asarray(f, dtype=float).ravel()
                df = np.asarray(df, dtype=float).reshape(f.shape[0], -1)
                return f, df

            def logpostf_grad(thetain):
                return logpostf(thetain)[1]
            try:
                testout = logpostfunc(theta0[10, :], return_grad=False)
                if type(testout) is tuple:  # make sure that return_grad functionality works
                    raise ValueError('Cannot stop returning a grad')

                def logpostf_nograd(theta):
                    return np.asarray(logpostfunc(theta, return_grad=False),
                                      dtype=float).ravel()
            except Exception:
                def logpostf_nograd(theta):  # if not, do not use return_grad key
                    return np.asarray(logpostfunc(theta)[0], dtype=float).ravel()
        else:
            logpostf_grad = None  # sometimes no derivative is given

            def logpostf_nograd(theta):
                return np.asarray(logpostfunc(theta), dtype=float).ravel()
            logpostf = logpostf_nograd

        if logpostf_grad is None:  # these are standard parameters if there is
            taracc = 0.25  # close to theoretical result 0.234
        else:
            taracc = 0.60  # close to theoretical result in LMC paper
        # begin preoptimizer
        logging.info('Begin PTLMC pre-optimization ...')
        # order the existing initial theta's by log pdf
        ord1 = np.argsort(-logpostf_nograd(theta0) +
                          (theta0.shape[1] *
                           rng.standard_normal(size=theta0.shape[0])**2))
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
        boundL = np.maximum(-10*np.ones(theta0.shape[1]),
                            np.min((theta0 - thetacen)/thetas, 0))
        boundU = np.minimum(10*np.ones(theta0.shape[1]),
                            np.max((theta0 - thetacen)/thetas, 0))
        bounds = spo.Bounds(boundL, boundU)
        thetaop = theta0
        # now we are ready to optimize for each chain
        logging.info('Begin PTLMC chain optimization ...')
        for k in range(0, numopt):
            if k % 10 == 0:
                logging.info(f"Currently working on optimization of k = {k}")
            if logpostf_grad is None:
                opval = spo.minimize(neglogpostf_nograd,
                                     (thetaop[k, :] - thetacen) / thetas,
                                     method='L-BFGS-B',
                                     bounds=bounds)
                thetaop[k, :] = thetacen + thetas * opval.x
            else:
                opval = spo.minimize(neglogpostf_nograd,
                                     (thetaop[k, :] - thetacen) / thetas,
                                     method='L-BFGS-B',
                                     jac=neglogpostf_grad,
                                     bounds=bounds)
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
                    r = (V.T*np.sqrt(W)) @ (V @ rng.standard_normal(size=thetacen.shape[0]))
                else:
                    stepadj /= 2
                    if stepadj < 1/16:
                        thetaop[k, :] = thetacen + thetas * opval.x
                        notmoved = False
                    continue

                if (neglogpostf_nograd(stepadj * r + opval.x) -
                        l0) < 3*thetacen.shape[0]:
                    thetaop[k, :] = thetacen + thetas * (stepadj * r + opval.x)
                    notmoved = False
                else:
                    stepadj /= 2
        # end preoptimizer
        # initialize the starting point
        logging.info('Initialize PTLMC starting point ...')
        thetac = thetaop
        if logpostf_grad is not None:
            fval, dfval = logpostf(thetac)
            fval = fval / temps
            dfval = dfval / tempsc
        else:
            fval = logpostf_nograd(thetac) / temps

        # preallocate the saving matrix
        thetasave = np.zeros((numchain,
                              sampperchain,
                              thetac.shape[1]))
        # try to start the covariance matrix
        covmat0 = np.cov(thetac.T)
        if thetac.shape[1] > 1:
            covmat0 = 0.9*covmat0 + 0.1*np.diag(np.diag(covmat0))  # add a diagonal part to prevent any non-moving issues
            W, V = np.linalg.eigh(covmat0)
            hc = V @ np.diag(np.sqrt(W)) @ V.T
        else:
            hc = np.sqrt(covmat0)
            hc = hc.reshape(1, 1)
            covmat0 = covmat0.reshape(1, 1)
        # Parameter initilzation
        tau = -1
        rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
        adjrho = rho*temps**(1/3)  # this adjusts rho across different temperatures
        adjrhoc = adjrho[:, np.newaxis]
        numtimes = 0  # number of times we reject, just to star
        logging.info('Run over all PTLMC chains and tune ...')
        for k in range(0, samptunning+sampperchain):  # loop over all chains
            if k % 100 == 0:
                logging.info(f"Currently working on {k}")
            rvalo = rng.standard_normal(size=thetac.shape)
            rval = (np.sqrt(2) * adjrho * np.squeeze(rvalo @ hc).T).T
            if thetac.shape[1] > 1:
                thetap = thetac + rval
            elif thetac.shape[1] == 1:
                thetap = thetac + rval[:, np.newaxis]
            if logpostf_grad is not None:
                # calculate the elements to move if there is a gradiant
                diffval = (adjrhoc ** 2) * (dfval @ covmat0)
                thetap += diffval
                fvalp, dfvalp = logpostf(thetap)  # thetap : no chain x dimension
                fvalp = fvalp / temps  # to flatten the posterior
                dfvalp = dfvalp / tempsc
                term1 = rvalo / np.sqrt(2)
                term2 = (adjrhoc / 2) * ((dfval + dfvalp) @ hc)
                qadj = -(2 * np.sum(term1 * term2, 1) + np.sum(term2**2, 1))
            else:
                # calculate the elements to move if there is not a gradiant
                fvalp = logpostf_nograd(thetap) / temps  # thetap : no chain x dimension
                qadj = np.zeros(fvalp.shape)
            swaprnd = np.log(rng.uniform(size=fval.shape[0]))
            whereswap = np.where(np.squeeze(swaprnd)
                                 < np.squeeze(fvalp - fval)
                                 + np.squeeze(qadj))[0]  # MH step to find which of the chains to swap
            if whereswap.shape[0] > 0:  # if we swap, do it where needed
                numtimes = numtimes + np.sum(whereswap > -1)/totnumchain
                thetac[whereswap] = np.copy(thetap[whereswap])
                fval[whereswap] = np.copy(fvalp[whereswap])
                if logpostf_grad is not None:
                    dfval[whereswap] = np.copy(dfvalp[whereswap])
            # do some swaps along the temperatures
            fvaln = fval * temps
            # go through 5 times, swapping where needed
            orderprop = self.tempexchange(fvaln, temps, iters=5, rng=rng)
            fval = fvaln[orderprop] / temps
            thetac = thetac[orderprop, :]
            if logpostf_grad is not None:
                dfvaln = tempsc * dfval
                dfval = (1 / tempsc) * dfvaln[orderprop, :]
            # if we have to tune, let's move tau up or down which gives bigger or smaller jumps
            if (k < samptunning) and (k % 10 == 0):  # if not done with tuning
                tau = tau + 1 / np.sqrt(1 + k/10) * \
                      ((numtimes / 10) - taracc)
                rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
                adjrho = rho*(temps**(1/3))  # adjusting rho across the chain
                adjrhoc = adjrho[:, np.newaxis]
                numtimes = 0
            elif k >= samptunning:  # if done with tuning
                thetasave[:, k-samptunning, :] = 1 * thetac[numtemps:, ]
        # return the unflattened values of the temp=1 chains
        sampler_info = {'theta': thetasave}
        return sampler_info


    # This function is taken from the surmise package (version 1.0.0) and
    # modified to skip swaps between chains with the same temperature
    def tempexchange(self, lpostf, temps, iters=1, rng=None):
        # This function will swap values along the chain given the log pdf values in an
        # array lpostf with temperature array temps. It will do it iters number of times.
        # It returns the (random) revised order.
        assert rng is not None

        order = np.arange(0, lpostf.shape[0])  # initializing
        for k in range(0, iters):
            # choose random values to check for swapping
            rtv = rng.choice(range(1, lpostf.shape[0]), lpostf.shape[0])
            for rt in rtv:
                rhoh = (1/temps[rt-1] - 1 / temps[rt])
                if rhoh == 0:
                    # chains with the same temperature (e.g. the temperature-1
                    # chains) would always be swapped, which only mixes up the
                    # walkers without changing the sampled distribution
                    continue
                if ((lpostf[order[rt]]-lpostf[order[rt - 1]]) * rhoh >
                        np.log(rng.uniform())):  # swap via the PT rule
                    temporder = order[rt - 1]
                    order[rt-1] = 1*order[rt]
                    order[rt] = 1 * temporder
        return order


    def run_MCMC_PTLMC(self, nsteps=500, nwalkers=16, ntemps=50, maxtemp=100, 
                       nstartparameters=1000, seed=None):
        """
        This function wrapps the PTLMC package to run the parallel tempering 
        ensemble MCMC with Langevin Monte Carlo

        `seed` is the seed of the random number generator of the sampler,
        which makes the chain reproducible.
        """
        rng = np.random.default_rng(seed)
        chain_data = {}

        def draw_func(n):
            return rng.uniform(self.min, self.max, (n, self.ndim))

        logging.info('Starting MCMC ...')
        result_dict = self.samplerPTLMC(logpostfunc=self.log_posterior,
                                   draw_func=draw_func,
                                   rng=rng,
                                   theta0=None,
                                   numtemps=ntemps,
                                   numchain=nwalkers,
                                   sampperchain=nsteps,
                                   maxtemp=maxtemp,
                                   nstartparameters=nstartparameters
                                   )

        self.chain = result_dict['theta']
        # This reshape should not be necessary, just done to match the format of the other MCMC
        self.chain = self.chain.reshape((nwalkers, nsteps, self.ndim))

        self.chain_sampler = 'PTLMC'

        # Write the chain to file (nwalkers, nsteps, self.ndim)
        chain_data['chain'] = self.chain
        logging.info('Writing MCMC chains to {}'.format(self.chain_path('PTLMC')))
        with open(self.chain_path('PTLMC'), 'wb') as file:
            pickle.dump(chain_data, file)


    def compute_log_likelihood_for_chain(self, sampler=None, output_path=None):
        """
        This function computes the log likelihood for each point in a chain and
        stores it in a new pkl file.

        The chain of `sampler` ('emcee', 'pocoMC' or 'PTLMC') is loaded from
        its chain file. If `sampler` is None, the chain of the last sampler run
        with this object is used. By default, the output is written next to the
        chain file with the suffix ``_log_likelihood``, e.g.
        ``./mcmc/chain_emcee_log_likelihood.pkl``.
        """
        if sampler is not None:
            logging.info('Loading chain from {}'.format(self.chain_path(sampler)))
            with open(self.chain_path(sampler), 'rb') as f:
                chain_data = pickle.load(f)
            self.chain = chain_data['chain']
            self.chain_sampler = sampler
        elif self.chain is False:
            raise ValueError('No chain has been run with this object, specify '
                             'the sampler of the chain to load')
        if output_path is None:
            chain_file = self.chain_path(self.chain_sampler)
            output_path = chain_file.with_name(
                chain_file.stem + '_log_likelihood' + chain_file.suffix)
        # create the output directory before the (expensive) computation
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logging.info('Computing log likelihood for the chain...')
        reshape_chain = self.chain.reshape(-1, self.ndim)
        likelihood = self.log_likelihood_point_by_point(reshape_chain)
        # emcee/PTLMC chains have shape (nwalkers, nsteps, ndim),
        # pocoMC samples have shape (nsamples, ndim)
        likelihood = likelihood.reshape(self.chain.shape[:-1])

        # Write the log_likelihood to file
        logging.info('Writing log_likelihood for chains to {}'.format(output_path))
        likelihood_data = {'log_likelihood': likelihood}
        with open(output_path, 'wb') as file:
            pickle.dump(likelihood_data, file)


    def run_pocoMC(self,n_effective=1000,n_active=250,n_prior=2000,
                   sample="tpcn",n_max_steps=200,random_state=42,
                   n_total=5000,n_evidence=5000,n_ndim_steps=2,pool=None,prior=None):
        """
        This function is based on PocoMC package (version 1.2.6).
        It works with versions of pocomc >= 1.2.2 and is tested up to 1.2.6.
        pocoMC is a Preconditioned Monte Carlo (PMC) sampler that uses 
        normalizing flows to precondition the target distribution.

        n_effective (int) – The effective sample size maintained during the run (default is n_ess=1000).
        n_active (int) – The number of active particles (default is n_active=250). It must be smaller than n_ess.
        n_prior (int) – Number of prior samples to draw (default is n_prior=2*(n_effective//n_active)*n_active).
        sample (str) – Type of MCMC sampler to use (default is sample="pcn"). 
            Options are ``"pcn"`` (t-preconditioned Crank-Nicolson) or ``"rwm"`` (Random-walk Metropolis).
            t-preconditioned Crank-Nicolson is the default and recommended sampler for PMC as it is more efficient and scales better with the number of parameters.
        n_max_steps (int) – Maximum number of MCMC steps (default is max_steps=10*n_dim).
        random_state (int or None) – Initial random seed.

        n_total (int) – The total number of effectively independent samples to be collected (default is n_total=5000).
        n_evidence (int) – The number of importance samples used to estimate the evidence (default is n_evidence=5000). 
                            If n_evidence=0, the evidence is not estimated using importance sampling and the SMC estimate is used instead. 
                            If preconditioned=False, the evidence is estimated using SMC and n_evidence is ignored.
        n_ndim_steps (int) – Number of MCMC steps in beta per dimension (default is n_ndim_steps=2).

        pool (int) – Number of processes to use for parallelisation (default is ``pool=None``). 
            If ``pool`` is an integer greater than 1, a ``multiprocessing`` pool is created with the specified number of processes.
        prior (class) – Prior distribution class implementing logpdf, rvs functions and dim, bounds attributes (default is None).

        When experiencing issues with the fork() function, set the environment variable ``export RDMAV_FORK_SAFE=1``.
        For more information on customizing the prior, see the PocoMC documentation.
        """
        logging.info('Generate the prior class for pocoMC ...')
        if prior is None:
            logging.info('Using uniform prior for all parameters ...')
            prior_distributions = []
            for i in range(self.ndim):
                prior_distributions.append(uniform(self.min[i], 
                                               self.max[i] - self.min[i]))
            prior = pocomc.Prior(prior_distributions)
        else:
            logging.info('Using custom prior ...')
            # Check the dimensions of the prior
            if self.ndim != prior.dim:
                logging.error('prior.dim does not match the model parameter space')
                raise ValueError('prior.dim does not match the model parameter space')

        logging.info('Starting pocoMC ...')
        sampler = pocomc.Sampler(prior=prior, likelihood=self.log_likelihood, 
                                likelihood_kwargs={'finite': True}, 
                                n_effective=n_effective, n_active=n_active, 
                                n_prior=n_prior, sample=sample, 
                                n_max_steps=n_max_steps, 
                                n_steps=n_ndim_steps*self.ndim,
                                random_state=random_state, vectorize=True, 
                                pool=pool)
        sampler.run(n_total=n_total, n_evidence=n_evidence)

        logging.info('Generate the posterior samples ...')
        samples, logl, logp = sampler.posterior(resample=True)

        logging.info('Generate the evidence ...')
        logz, logz_err = sampler.evidence()
        logging.info('Log evidence: {}'.format(logz))
        logging.info('Log evidence error: {}'.format(logz_err))

        self.chain = samples
        self.chain_sampler = 'pocoMC'
        chain_data = {'chain': samples, 'logl': logl,
                        'logp': logp, 'logz': logz, 'logz_err': logz_err}
        logging.info('Writing pocoMC chains to {}'.format(self.chain_path('pocoMC')))
        with open(self.chain_path('pocoMC'), 'wb') as file:
            pickle.dump(chain_data, file)
