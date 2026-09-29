"""
Parallel tempering Langevin Monte Carlo (PTLMC) sampler.

The functions are taken from the surmise package (version 1.0.0) and modified
for this package. They are used by BayesianAnalysis.run_ptlmc.
"""

import logging

import numpy as np
import scipy.optimize as spo

logger = logging.getLogger(__name__)


# This function is taken from the surmise package (version 1.0.0) and
# modified: the number of initial draws is set by nstartparameters, the
# tuning phase is longer (fractunning = 2), progress is logged, the
# unflattened chains of the temperature-1 walkers are returned, and the
# perturbation of the optimized starting points uses the inverse Hessian as
# covariance and stops after a few step reductions.
def sampler(
    logpostfunc,
    draw_func,
    rng,
    ndim,
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
    ndim : int
        Number of parameters, used for the minimum number of initial draws.
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
    nmin = max(numtemps + numchain, 10 * ndim)
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
                np.linspace(np.log(maxtemp), np.log(maxtemp) / (numtemps + 1), numtemps)
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
            if type(testout) is tuple:  # make sure that return_grad functionality works
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
    logger.info(f"Optimizing the starting points of the {numopt} PTLMC chains ...")
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
    for k in range(0, numopt):
        logger.debug(f"Optimizing the starting point {k + 1}/{numopt} ...")
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
            if stepadj < 1 / 16:
                # no acceptable perturbation, keep the optimized point
                thetaop[k, :] = thetacen + thetas * opval.x
                break
            if not (W > 0).all():
                stepadj /= 2
                continue
            # random step with the inverse Hessian V diag(W) V^T as covariance
            r = V @ (np.sqrt(W) * rng.standard_normal(size=thetacen.shape[0]))

            if (neglogpostf_nograd(stepadj * r + opval.x) - l0) < 3 * thetacen.shape[0]:
                thetaop[k, :] = thetacen + thetas * (stepadj * r + opval.x)
                notmoved = False
            else:
                stepadj /= 2
    # end preoptimizer
    # initialize the starting point
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
    adjrho = rho * temps ** (1 / 3)  # this adjusts rho across different temperatures
    adjrhoc = adjrho[:, np.newaxis]
    numtimes = 0  # accumulated acceptance rate, reset after each tuning update
    n_total = samptunning + sampperchain
    log_every = max(n_total // 10, 1)
    logger.info(
        f"Running {samptunning} tuning and {sampperchain} sampling steps of the "
        "PTLMC chains ..."
    )
    for k in range(0, n_total):  # loop over all chains
        if k % log_every == 0:
            logger.info(f"PTLMC step {k + 1}/{n_total} ...")
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
        orderprop = temp_exchange(fvaln, temps, iters=5, rng=rng)
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
    logger.info("PTLMC sampling finished")
    # return the unflattened values of the temp=1 chains
    sampler_info = {"theta": thetasave}
    return sampler_info


# This function is taken from the surmise package (version 1.0.0) and
# modified to skip swaps between chains with the same temperature
def temp_exchange(lpostf, temps, iters=1, rng=None):
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
