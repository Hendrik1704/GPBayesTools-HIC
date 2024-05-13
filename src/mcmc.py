"""
Markov chain Monte Carlo model calibration using the `affine-invariant ensemble
sampler (emcee) <http://dfm.io/emcee>`_.

This module must be run explicitly to create the posterior distribution.
Run ``python -m src.mcmc --help`` for complete usage information.

On first run, the number of walkers and burn-in steps must be specified, e.g.
::

    python -m src.mcmc --nwalkers 500 --nburnsteps 100 200

would run 500 walkers for 100 burn-in steps followed by 200 production steps.
This will create the HDF5 file :file:`mcmc/chain.hdf` (default path).

On subsequent runs, the chain resumes from the last point and the number of
walkers is inferred from the chain, so only the number of production steps is
required, e.g. ::

    python -m src.mcmc 300

would run an additional 300 production steps (total of 500).

To restart the chain, delete (or rename) the chain HDF5 file.
"""

import argparse
import logging
import pickle

from pathlib import Path
import emcee
import numpy as np
from scipy.linalg import lapack
import dill

from . import workdir, parse_model_parameter_file
from .emulator import Emulator
from .emulator_BAND import EmulatorBAND
import scipy.optimize as spo
from .ptemcee_modified.sampler import Sampler as PTemceeSampler

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
    elif info < 0:
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

        if status is None:
            status = nsteps // 10

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

    """
    def __init__(self, mcmc_path="./mcmc/chain.pkl",
                 expdata_path="./exp_data.dat",
                 model_parafile="./model.dat"
    ):
        logging.info('Initializing MCMC ...')
        self.mcmc_path = Path(mcmc_path)
        self.mcmc_path.parent.mkdir(exist_ok=True)
        logging.info('Final Markov Chain results will be saved in {}'.format(
            self.mcmc_path)
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
        self.closureTestFalg = False
        self.emuList = []
        self.chain = False


    def trainEmulator(self, model_parafile="./model.dat",
                      training_data_path="./training_data", npc=10):
        # setup the emulator
        logging.info('Initializing emulators for the training model ...')
        self.emuList.append(
                Emulator(training_set_path=training_data_path,
                         parameter_file=model_parafile,
                         npc=npc)
        )


    def loadEmulator(self, emulatorPathList):
        for i, emuPath in enumerate(emulatorPathList):
            with open(emuPath, 'rb') as f:
                emu_i = dill.load(f)
                self.emuList.append(emu_i)
        logging.info("Number of Emulators: {}".format(len(self.emuList)))


    def _predict(self, X, extra_std=0.0):
        nPreds = X.shape[0]
        modelPred = np.zeros([nPreds, self.nobs])
        modelPredCov = np.zeros([nPreds, self.nobs, self.nobs])
        extra_std_arr = extra_std*X[:, -1]
        currIdx = 0
        for i, emu_i in enumerate(self.emuList):
            model_Y, model_cov = emu_i.predict(
                X, return_cov=True, extra_std=extra_std_arr)
            nobs_i = model_Y.shape[1]
            modelPred[:, currIdx:currIdx+nobs_i] = model_Y
            modelPredCov[:, currIdx:currIdx+nobs_i, currIdx:currIdx+nobs_i] = model_cov
            currIdx += nobs_i
        return modelPred, modelPredCov


    def log_prior(self, X):
        """
        Evaluate the prior at `X`.

        """
        X = np.array(X, copy=False, ndmin=2)

        #not normalized
        #lp = np.zeros(X.shape[0])

        #normalize the prior
        lp = np.log( np.ones(X.shape[0]) / self.prior_volume_ )

        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        return lp


    def log_likelihood(self, X, extra_std_prior_scale=0.001):
        """
        Evaluate the likelihood at `X`.
        """
        X = np.array(X, copy=False, ndmin=2)
        lp = np.zeros(X.shape[0])
        inside = np.all( (X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        extra_std = X[inside, -1]

        nsamples = np.count_nonzero(inside)
        if nsamples > 0:
            # not sure why to use the last parameter for extra std
            extra_std = 0.0*X[inside, -1]

            model_Y, model_cov = self._predict(X[inside], extra_std)

            # allocate difference (model - experiment) and covariance arrays
            dY = np.empty([nsamples, self.nobs])
            cov = np.empty([nsamples, self.nobs, self.nobs])
            dY = model_Y - self.expdata
            # add experiment cov to model cov
            cov = model_cov + self.expdata_cov

            # compute log likelihood at each point
            lp[inside] += list(map(mvn_loglike, dY, cov))

            # add prior for extra_std (model sys error)
            lp[inside] += (2*np.log(extra_std + 1e-16)
                           - extra_std/extra_std_prior_scale)
        return lp

    def log_likelihood_point_by_point(self, X, extra_std_prior_scale=0.001):
        """
        Evaluate the likelihood at `X` point by point.
        This is used for the log_likelihood computation when the chain is already
        generated and the likelihood is computed for each point in the chain.
        """
        lp = np.zeros(X.shape[0])
        
        for k in range(X.shape[0]):
            if k % 100 == 0:
                logging.info("Evaluating log_likelihood at point {}".format(k))
            Xk = np.array(X[k], copy=False, ndmin=2)
            inside = np.all( (Xk > self.min) & (Xk < self.max))
            lp[k] = -np.inf if not inside else 0.0

            nsamples = 1 if inside else 0
            if nsamples > 0:
                extra_std = 0.0*Xk[0,-1]
                model_Y, model_cov = self._predict(Xk, extra_std)

                # allocate difference (model - experiment) and covariance arrays
                dY = np.empty([nsamples, self.nobs])
                cov = np.empty([nsamples, self.nobs, self.nobs])
                dY = model_Y - self.expdata
                # add experiment cov to model cov
                cov = model_cov + self.expdata_cov

                # compute log likelihood at each point
                lp[k] += list(map(mvn_loglike, dY, cov))

                # add prior for extra_std (model sys error)
                lp[k] += (2*np.log(extra_std + 1e-16)
                               - extra_std/extra_std_prior_scale)
        return lp

    def log_posterior(self, X, extra_std_prior_scale=.05):
        """
        Evaluate the posterior at `X`.

        `extra_std_prior_scale` is the scale parameter for the prior
        distribution on the model sys error parameter:

            prior ~ sigma^2 * exp(-sigma/scale)

        """
        X = np.array(X, copy=False, ndmin=2)

        lp = np.zeros(X.shape[0])

        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        nsamples = np.count_nonzero(inside)
        if nsamples > 0:
            # not sure why to use the last parameter for extra std
            extra_std = 0.0*X[inside, -1]

            #model_Y, model_cov = self.emu.predict(
            #    X[inside], return_cov=True, extra_std=extra_std
            #)
            model_Y, model_cov = self._predict(X[inside], extra_std)

            # allocate difference (model - expt) and covariance arrays
            dY = np.empty([nsamples, self.nobs])
            cov = np.empty([nsamples, self.nobs, self.nobs])
            dY = model_Y - self.expdata
            # add expt cov to model cov
            cov = model_cov + self.expdata_cov

            # compute log likelihood at each point
            lp[inside] += list(map(mvn_loglike, dY, cov))

            # add prior for extra_std (model sys error)
            lp[inside] += (2*np.log(extra_std + 1e-16)
                           - extra_std/extra_std_prior_scale)

        return lp


    def _read_in_exp_data_pickle(self, filepath):
        """This function reads in exp data and compute the covariance matrix"""
        model_data = []
        model_data_err = []
        
        with open(filepath, "rb") as fp:
            dataDict = pickle.load(fp)

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


    def set_closure_test_truth(self, filename):
        self.closureTestFalg = True
        self.trueParams = []
        with open(filename, "r") as parfile:
            for line in parfile:
                line = line.split()
                self.trueParams.append(float(line[1]))
        self.trueParams = np.array(self.trueParams)


    @staticmethod
    def map(f, args):
        """
        Dummy function so that this object can be used as a 'pool' for
        :meth:`emcee.EnsembleSampler`.

        """
        return f(args)


    def run_mcmc(self, nsteps=500, nburnsteps=None, nwalkers=None,
                 status=None, nthin=10, skip_initial_state_check=False):
        """
        Run MCMC model calibration. If the chain already exists, continue from
        the last point, otherwise burn-in and start the chain.
        """
        chain_data = {}
        try:
            with open(self.mcmc_path, 'rb') as f:
                chain_data = pickle.load(f)
        except FileNotFoundError:
            pass

        if 'chain' not in chain_data:
            burnFlag = True
        else:
            burnFlag = False

        if nburnsteps is None or nwalkers is None:
                logging.error(
                    'must specify nburnsteps and nwalkers to start chain')
                return

        logging.info('Starting MCMC ...')
        sampler = LoggingEnsembleSampler(
            nwalkers, self.ndim, self.log_posterior, pool=self
        )

        if burnFlag:
            logging.info(
                    'no existing chain found, starting initial burn-in')

            # Run first half of burn-in starting from random positions.
            nburn0 = nburnsteps // 2
            sampler.run_mcmc(
                self.random_pos(nwalkers),
                nburn0,
                status=status,
                skip_initial_state_check=skip_initial_state_check
            )
            logging.info('resampling walker positions')
            # Reposition walkers to the most likely points in the chain,
            # then run the second half of burn-in.  This significantly
            # accelerates burn-in and helps prevent stuck walkers.
            X0 = sampler.flatchain[
                np.unique(
                    sampler.flatlnprobability,
                    return_index=True
                )[1][-nwalkers:]
            ]
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
            X0 = chain_data['chain'][:, -1, :]

        sampler.run_mcmc(X0, nsteps, status=status, 
                         skip_initial_state_check=skip_initial_state_check)

        thinedChain = sampler.chain[:, ::nthin, :]
        if 'chain' in chain_data:
            chain_data['chain'] = np.concatenate((chain_data['chain'], thinedChain), axis=1)
            self.chain = chain_data['chain']
        else:
            chain_data['chain'] = thinedChain
            self.chain = thinedChain

        # Append the new data to the existing file
        logging.info('writing chain to file')
        with open(self.mcmc_path, 'wb') as file:
            pickle.dump(chain_data, file)


    # This function is taken from the surmise package (version 0.2.1) and slightly modified.
    # The ptemcee package source code is copied and modified to work with more modern versions of numpy.
    def sampler_PTemcee_wrapper(self,draw_func,
            log_likelihood,
            log_prior,
            nburnin=100,
            ndim=15,
            niterations=200,
            ntemps=50,
            nthin=1,
            nwalkers=200,
            nthreads=10,
            Tmax=np.inf,
            verbose=False):
        """
        Parameters
        ----------
        logpostfunc: function,
            Not used in PTMC sampler. It uses log_likelihood and log_prior instead.

        draw_func: function, required
            A function that produces approximate draws from the prior distribution.
            This is used to initialize MCMC chains.
        log_likelihood: function, required
            Log of the likelihood.
        log_prior: function, required
            Log of the prior.
        nburnin:
            Number of burnin samples.
        ndim:
            Dimension of the model parameter space.
        niterations:
            Number of MCMC samples for each chain after burnin.
        nthin:
            Thinning applied to MCMC chains. The default is 1, which is no thinning.
        nwalkers:
            Number of chains.
        nthreads:
            Number of threads for parallel computation.
        ntemps: integer, optional
            A positive integer that controls how many chains of varying temperature to run simultaneously.
            The default is 50.
        Tmax: double, optional
            A number larger than 1 that gives the maximum temperature used in parallel tempering.
            The default is inf.
        verbose: bool, optional
            Boolean flag to control output printing.  The default is False (do not print).

        Raises
        ------
        ValueError
            Indicates that something was not entered right, please check the documentation.

        Returns
        -------
        dictionary
            A dictionary that contains the sampled values in the key 'theta' and the acceptance rate in the key 'acc_rate'.
        """
        nburnin = int(nburnin)
        ndim = int(ndim)
        niterations = int(niterations)
        ntemps = int(ntemps)
        nthin = int(nthin)
        nwalkers = int(nwalkers)
        nthreads = int(nthreads)
        Tmax = float(Tmax)
        global log_like
        def log_like(x): return log_likelihood(x.reshape(-1, ndim))
        global log_prior_fix
        def log_prior_fix(x): return log_prior(x.reshape(-1, ndim))
        ptsampler_ex = PTemceeSampler(nwalkers=nwalkers, dim=ndim, logl=log_like, logp=log_prior_fix, ntemps=ntemps, threads=nthreads, Tmax=Tmax)

        pos0 = np.array([draw_func(nwalkers) for n in range(0, ntemps)])
        if verbose:
            print("Running burn-in phase")
        for p, lnprob, lnlike in ptsampler_ex.sample(pos0, iterations=nburnin, adapt=True):
            pass
        ptsampler_ex.reset()  # Discard previous samples from the chain, but keep the position

        if verbose:
            print("Running MCMC chains")
        # Now we sample for nwalkers*niterations, recording every nthin-th sample
        for p, lnprob, lnlike in ptsampler_ex.sample(p, iterations=niterations, thin=nthin, adapt=True):
            pass

        if verbose:
            print('Done MCMC')

        mean_acc_frac = np.mean(ptsampler_ex.acceptance_fraction)

        if verbose:
            print(f"Mean acceptance fraction: {mean_acc_frac:.3f}",
                f"(in total {nwalkers*niterations} steps)")

        # We only analyze the zero temperature MCMC samples
        samples = ptsampler_ex.chain[0, :, :, :].reshape((-1, ndim))

        sampler_info = {'theta': samples, 'acc_rate': mean_acc_frac}
        return sampler_info


    def run_MCMC_ptemcee(self, nsteps=500, nburnsteps=None, nwalkers=None,
                 status=None, nthin=10, ntemps=50, nthreads=1):
        """
        This function wrapps the ptemcee package to run the parallel tempering 
        MCMC.
        The parameter nthreads should be used with caution. The Pool object is
        pckling and unpickling all the data and this can be very slow.
        It might be faster to run in sequential mode. There is a print out around
        the Pool initialization, if you still want to try it and check the speed.
        """
        # Check that nsteps modulo nthin is zero
        if nsteps % nthin != 0:
            raise ValueError('nsteps must be divisible by nthin')

        chain_data = {}

        # Run the MCMC
        logging.info('Starting MCMC ...')
        result_dict = self.sampler_PTemcee_wrapper(draw_func=self.random_pos,
                                log_likelihood=self.log_likelihood,
                                log_prior=self.log_prior,
                                nburnin=nburnsteps,
                                ndim=self.ndim,
                                niterations=nsteps,
                                ntemps=ntemps,
                                nthin=nthin,
                                nwalkers=nwalkers,
                                nthreads=nthreads,
                                verbose=status)
        
        # This is the thinned chain already at 0 temperature
        chain_data['chain'] = result_dict['theta']

        # Reshape the chain to have the shape (nwalkers, nsteps//nthin, ndim)
        # This format is similar to the other MCMC implemented in this file
        old_shape = (nwalkers, nsteps//nthin, self.ndim)
        chain_data['chain'] = chain_data['chain'].reshape(old_shape)
        self.chain = chain_data['chain']

        # Write the chain to file
        logging.info('Writing MCMC chains to file...')
        with open(self.mcmc_path, 'wb') as file:
            pickle.dump(chain_data, file)


    # This function is taken from the surmise package (version 0.2.1) and slightly modified
    # to match with our definition of logpostfunc and the format of the chain output
    def samplerPTLMC(self,logpostfunc,
                draw_func,
                theta0=None,
                numtemps=32,
                numchain=16,
                sampperchain=400,
                maxtemp=30,
                nstartparameters=1000):
        """

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

        Raises
        ------
        ValueError
            Indicates that something was not entered right, please check documentation.

        Returns
        -------
        dictionary
            A dictionary that contains the sampled values in the key 'theta'.
        """
        # If we do not get parameters to start, draw nstartparameters
        if theta0 is None:
            theta0 = draw_func(nstartparameters)
        # Need to make sure the initial draws are sufficent to continue
        if theta0.shape[0] < 10*theta0.shape[1]:
            theta0 = draw_func(nstartparameters)
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
        temps = np.array(temps, ndmin=2).T
        # number of optimization at each chain before starting
        numopt = temps.shape[0]
        # before beginning, let's test out the given logpdf function
        testout = logpostfunc(theta0[0:2, :])
        if type(testout) is tuple:
            if len(testout) != 2:
                raise ValueError('log density does not return 1 or 2 elements')
            if testout[1].shape[1] is not theta0.shape[1]:
                raise ValueError('derivative appears to be the wrong shape')
            logpostf = logpostfunc

            def logpostf_grad(thetain):
                return logpostfunc(thetain)[1]
            try:
                testout = logpostfunc(theta0[10, :], return_grad=False)
                if type(testout) is tuple:  # make sure that return_grad functionality works
                    raise ValueError('Cannot stop returning a grad')

                def logpostf_nograd(theta):
                    return logpostfunc(theta, return_grad=False)
            except Exception:
                def logpostf_nograd(theta):  # if not, do not use return_grad key
                    return logpostfunc(theta)[0]
        else:
            logpostf_grad = None  # sometimes no derivative is given
            def logpostf(theta):
                return np.array(logpostfunc(theta), ndmin=2).T
            def logpostf_nograd(theta):
                return np.array(logpostfunc(theta), ndmin=2).T
        if logpostf_grad is None:  # these are standard parameters if there is
            taracc = 0.25  # close to theoretical result 0.234
        else:
            taracc = 0.60  # close to theoretical result in LMC paper
        # begin preoptimizer
        logging.info('Begin PTLMC pre-optimization ...')
        # order the existing initial theta's by log pdf
        ord1 = np.argsort(-np.squeeze(logpostf_nograd(theta0)) +
                        (theta0.shape[1] *
                        np.random.standard_normal(size=theta0.shape[0])**2))
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
                return -thetas * logpostf_grad(theta.reshape((1, len(theta))))
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
                r = (V.T*np.sqrt(W)) @ (V @ np.random.standard_normal(size=thetacen.shape[0]))

                if (neglogpostf_nograd((stepadj * r + opval.x)) -
                        l0) < 3*thetacen.shape[0]:
                    thetaop[k, :] = thetacen + thetas * (stepadj * r + opval.x)
                    notmoved = False
                else:
                    stepadj /= 2
                if stepadj < 1/16:
                    thetaop[k, :] = thetacen + thetas * opval.x
                    notmoved = False
        # end preoptimizer
        # initialize the starting point
        logging.info('Initialize PTLMC starting point ...')
        thetac = thetaop
        if logpostf_grad is not None:
            fval, dfval = logpostf(thetac)
            fval = fval/temps
            dfval = dfval/temps
        else:
            fval = logpostf_nograd(thetac)
            fval = fval/temps
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
        numtimes = 0  # number of times we reject, just to star
        logging.info('Run over all PTLMC chains and tune ...')
        for k in range(0, samptunning+sampperchain):  # loop over all chains
            if k % 100 == 0:
                logging.info(f"Currently working on {k}")
            rvalo = np.random.normal(0, 1, thetac.shape)
            rval = np.sqrt(2) * adjrho * (rvalo @ hc)
            thetap = thetac + rval
            if logpostf_grad is not None:
                # calculate the elements to move if there is a gradiant
                diffval = (adjrho ** 2) * (dfval @ covmat0)
                thetap += diffval
                fvalp, dfvalp = logpostf(thetap)  # thetap : no chain x dimension
                fvalp = fvalp / temps  # to flatten the posterior
                dfvalp = dfvalp / temps
                term1 = rvalo / np.sqrt(2)
                term2 = (adjrho / 2) * ((dfval + dfvalp) @ hc)
                qadj = -(2 * np.sum(term1 * term2, 1) + np.sum(term2**2, 1))
            else:
                # calculate the elements to move if there is not a gradiant
                fvalp = logpostf_nograd(thetap)  # thetap : no chain x dimension
                fvalp = fvalp / temps
                qadj = np.zeros(fvalp.shape)
            swaprnd = np.log(np.random.uniform(size=fval.shape[0]))
            whereswap = np.where(np.squeeze(swaprnd)
                                < np.squeeze(fvalp - fval)
                                + np.squeeze(qadj))[0]  # MH step to find which of the chains to swap
            if whereswap.shape[0] > 0:  # if we swap, do it where needed
                numtimes = numtimes + np.sum(whereswap > -1)/totnumchain
                thetac[whereswap, :] = 1*thetap[whereswap, :]
                fval[whereswap] = 1*fvalp[whereswap]
                if logpostf_grad is not None:
                    dfval[whereswap, :] = 1*dfvalp[whereswap, :]
            # do some swaps along the temperatures
            fvaln = fval*temps
            orderprop = self.tempexchange(fvaln, temps, iters=5)  # go through 5 times, swapping where needed
            fval = fvaln[orderprop] / temps
            thetac = thetac[orderprop, :]
            if logpostf_grad is not None:
                dfvaln = temps * dfval
                dfval = (1 / temps) * dfvaln[orderprop, :]
            # if we have to tune, let's move tau up or down which gives bigger or smaller jumps
            if (k < samptunning) and (k % 10 == 0):  # if not done with tuning
                tau = tau + 1 / np.sqrt(1 + k/10) * \
                    ((numtimes / 10) - taracc)
                rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
                adjrho = rho*(temps**(1/3))  # adjusting rho across the chain
                numtimes = 0
            elif k >= samptunning:  # if done with tuning
                thetasave[:, k-samptunning, :] = 1 * thetac[numtemps:, ]
        # save the theta values in the temp=1 chains, squeezing flattening the values of all chains
        theta = thetasave
        # store this in a dictionary
        sampler_info = {'theta': theta}
        return sampler_info


    # This function is taken from the surmise package (vesion 0.2.1)
    def tempexchange(self, lpostf, temps, iters=1):
        # This function will swap values along the chain given the log pdf values in an
        # array lpostf with temperature array temps. It will do it iters number of times.
        # It returns the (random) revised order.
        order = np.arange(0, lpostf.shape[0])  # initializing
        for k in range(0, iters):
            rtv = np.random.choice(range(1, lpostf.shape[0]), lpostf.shape[0])  # choose random values to check for swapping
            for rt in rtv:
                rhoh = (1/temps[rt-1] - 1 / temps[rt])
                if ((lpostf[order[rt]]-lpostf[order[rt - 1]]) * rhoh >
                        np.log(np.random.uniform(size=1))):  # swap via the PT rule
                    temporder = order[rt - 1]
                    order[rt-1] = 1*order[rt]
                    order[rt] = 1 * temporder
        return order


    def run_MCMC_PTLMC(self, nsteps=500, nwalkers=16, ntemps=50, maxtemp=100, nstartparameters=1000):
        """
        This function wrapps the PTLMC package to run the parallel tempering 
        ensemble MCMC with Langevin Monte Carlo
        """
        chain_data = {}

        logging.info('Starting MCMC ...')
        result_dict = self.samplerPTLMC(logpostfunc=self.log_posterior,
                                   draw_func=self.random_pos,
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

        # Write the chain to file (nwalkers, nsteps, self.ndim)
        logging.info('Writing MCMC chains to file ...')
        chain_data['chain'] = self.chain

        # Write the chain to file
        logging.info('Writing MCMC chains to file...')
        with open(self.mcmc_path, 'wb') as file:
            pickle.dump(chain_data, file)


    def compute_log_likelihood_for_chain(self, output_path="./mcmc/log_likelihood.pkl"):
        """
        This function computes the log likelihood for the loaded chain.
        The log likelihood is computed for each point in the chain and stored
        in a new pkl file.
        """
        if self.chain is False:
            logging.error('Load chain before computing log likelihood')
            with open(self.mcmc_path, 'rb') as f:
                chain_data = pickle.load(f)
            self.chain = chain_data['chain']
        logging.info('Computing log likelihood for the chain...')
        reshape_chain = self.chain.reshape(-1, self.ndim)
        likelihood = self.log_likelihood_point_by_point(reshape_chain)
        likelihood = likelihood.reshape((self.chain.shape[0], self.chain.shape[1]))

        # Write the log_likelihood to file
        logging.info('Writing log_likelihood for chains to file...')
        likelihood_data = {'log_likelihood': likelihood}
        with open(output_path, 'wb') as file:
            pickle.dump(likelihood_data, file)


def main():
    parser = argparse.ArgumentParser(
            description='Markov Chain Monte Carlo',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--nsteps', type=int, default=500,
        help='number of steps'
    )
    parser.add_argument(
        '--nwalkers', type=int, default=100,
        help='number of walkers'
    )
    parser.add_argument(
        '--nburnsteps', type=int, default=200,
        help='number of burn-in steps'
    )
    parser.add_argument(
        '--status', type=int,
        help='number of steps between logging status'
    )
    parser.add_argument(
        '--exp', type=str, default='./exp_data.dat',
        help="experimental data"
    )
    parser.add_argument(
        '--model_design', type=str,
        default='model_parameter_dict_examples/ABCD.txt',
        help="model parameter filename"
    )
    parser.add_argument(
        '--training_set', type=str,
        default='./training_dataset',
        help="model training set parameters"
    )
    args = parser.parse_args()

    mymcmc = Chain(expdata_path=args.exp, model_parafile=args.model_design,
                   training_data_path=args.training_set)
    mymcmc.run_mcmc(nsteps=args.nsteps, nburnsteps=args.nburnsteps,
                    nwalkers=args.nwalkers, status=args.status)


if __name__ == '__main__':
    main()
