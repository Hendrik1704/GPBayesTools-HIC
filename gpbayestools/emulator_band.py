"""
Emulator of the model outputs with the GP emulators of surmise (training,
prediction and validation).

Uses the `Gaussian process regression
<https://surmise.readthedocs.io/en/latest/index.html>`_ implemented by the BAND
collaboration.
"""

import logging
import warnings

import numpy as np
import surmise
from surmise.emulation import emulator

from .emulator_base import EmulatorBase

logger = logging.getLogger(__name__)


class EmulatorBAND(EmulatorBase):
    """
    Multidimensional Gaussian process emulator wrapper for the GP emulators of
    the BAND collaboration (surmise).

    The number of principal components is chosen by surmise, so there is no
    npc argument. surmise keeps all PCs whose squared singular value of the
    standardized training data exceeds a small threshold (0.1 for PCGP, 0.001
    for the other methods, out of a total of n_ev * n_obs), i.e. in practice
    all PCs except exactly degenerate directions. PCSK additionally drops PCs
    whose mean ratio of noise to signal exceeds 8. The hyperparameters are
    optimized on a random subset of the training points (at most 20 points
    per parameter for PCGP, PCGPwM and PCGPwImpute and 25 for PCSK), the GPs
    are then conditioned on all training points.

    PCGPwM and PCGPwImpute handle missing observables: for these methods,
    non-finite observables of a training point (or observables with
    non-finite errors) are passed to surmise as missing (NaN), and only
    points without any finite observable are discarded. PCGPwM models the
    missing values in the GPs of the PCs, PCGPwImpute imputes them before
    the training. Without missing values, both behave like PCGP with the PC
    threshold of 0.001. The other methods discard training points with
    non-finite values (see `EmulatorBase`).

    Parameters
    ----------
    training_set_path : str or path-like
        Path to the pickle file with the training data, a dictionary
        ``{event_id: {'parameter': array (n_parameters,), 'obs': array (2,
        n_obs) with the values and statistical errors}}``.
    parameter_file : str or path-like
        Path to the model parameter file.
    method : {"PCGP", "PCSK", "PCGPwImpute", "PCGPwM"}, default="PCGP"
        surmise emulation method. PCSK uses the statistical errors of the
        training data, PCGPwM and PCGPwImpute allow missing observables.
    log_trafo : bool, default=False
        If True, the emulator is trained on the log of the observables, which
        must be positive, and predict() returns the mean and covariance in log
        space. Experimental data used with the emulator must then be
        log-transformed as well.
    max_rel_uncertainty_data : float or None, default=None
        Training points with a larger relative statistical error of any
        observable are discarded. None disables this filter.
    exp_and_cov_diagonal : bool, default=False
        If True, predict() returns exp(mean) and a diagonal covariance in the
        original scale of the observables (see `EmulatorBase`). Requires
        ``log_trafo=True``.
    seed : int or None, default=None
        Seed of the random number generator that is set as the global RNG of
        surmise (>= 1.0.0) before each training, so that every training with
        the same seed and training points gives the same emulator.

    Raises
    ------
    ValueError
        For invalid training data or options (see `EmulatorBase`).
    """

    _METHODS = ("PCGP", "PCSK", "PCGPwImpute", "PCGPwM")

    def __init__(
        self,
        training_set_path,
        parameter_file,
        *,
        method="PCGP",
        log_trafo=False,
        max_rel_uncertainty_data=None,
        exp_and_cov_diagonal=False,
        seed=None,
    ):
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown method {method!r}, expected one of {', '.join(self._METHODS)}"
            )
        self.method = method
        self.seed = seed
        self._missing_observables = method in ("PCGPwM", "PCGPwImpute")
        super().__init__(
            training_set_path,
            parameter_file,
            log_trafo=log_trafo,
            max_rel_uncertainty_data=max_rel_uncertainty_data,
            exp_and_cov_diagonal=exp_and_cov_diagonal,
        )

    def train_emulator(self, event_mask):
        """
        Train the emulator on the training points selected by `event_mask`.

        Parameters
        ----------
        event_mask : ndarray of bool of shape (n_ev,)
            Mask of the training points to use.

        Raises
        ------
        ValueError
            If `event_mask` is not a boolean array of shape (n_ev,), or if an
            observable is missing in all selected training points.
        """
        event_mask = self._check_event_mask(event_mask)
        n_ev, n_obs = self.model_data[event_mask, :].shape
        n_values = np.sum(np.isfinite(self.model_data[event_mask, :]), axis=0)
        if np.any(n_values == 0):
            raise ValueError(
                f"The observables {np.flatnonzero(n_values == 0).tolist()} are "
                "missing in all selected training points"
            )
        logger.info(
            f"Training the surmise {self.method} emulator with {n_ev} training "
            "points ..."
        )
        # the observables are the "x" locations of surmise
        x = np.arange(n_obs).reshape(-1, 1)
        args = {"warnings": True}
        if self.method == "PCSK":
            # PCSK uses the statistical errors of the training data
            args["simsd"] = self.model_data_err[event_mask, :].T
        # the noise of the simulations, which PCSK models with the
        # statistical errors, but does not include in the predictive variance
        self._sim_noise_var = np.mean(self.model_data_err[event_mask, :] ** 2, axis=0)

        # surmise (>= 1.0.0) requires a global RNG to be set before the
        # training. A new generator is used for each training, so that the
        # same seed always gives the same emulator.
        surmise.set_RNG(np.random.default_rng(self.seed))
        # surmise resets all warning filters of the process
        # (warnings.resetwarnings), which are restored afterwards
        with warnings.catch_warnings():
            self.emu_ = emulator(
                x=x,
                theta=self.design_points[event_mask, :],
                f=self.model_data[event_mask, :].T,
                method=self.method,
                args=args,
                # keep points and observables with missing values, surmise
                # would otherwise remove those with >= 80% missing values
                options={"thetarmnan": False, "xrmnan": False},
            )
        logger.info("Emulator training finished")

    def _full_covariance(self, pred):
        """
        Covariance matrices of the surmise prediction `pred` with shape
        (ntheta, n_obs, n_obs). surmise's covx() only contains the variance of
        the emulated PCs, while var() also contains the variance of the
        discarded PCs, which is added to the diagonal here.
        """
        cov = np.array(pred.covx())
        missing_var = pred.var().T - np.diagonal(cov, axis1=1, axis2=2)
        idx = np.arange(cov.shape[1])
        cov[:, idx, idx] += np.clip(missing_var, 0.0, None)
        return cov

    def _noise_covariance(self):
        """
        Covariance of the noise (nugget) of the GPs of the PCs in the space of
        the observables, which is contained in the predictive covariance of
        surmise: sigma2hat * exp(hypnug) for PCGP, sig2 * nug for the other
        methods (PCSK, PCGPwM, PCGPwImpute).
        """
        info = self.emu_._info
        infos = info["emulist"]
        if self.method == "PCGP":
            noise = np.array([i["sigma2hat"] * np.exp(i["hypnug"]) for i in infos])
            pctscale = (info["pct"].T * info["scale"]).T
        else:
            noise = np.array([i["sig2"] * i["nug"] for i in infos])
            pctscale = (info["pcti"].T * info["standardpcinfo"]["scale"]).T
        return (pctscale * noise) @ pctscale.T

    def predict(self, X, return_cov=True, include_noise=False):
        """
        Predict model output at the parameter points `X`.

        By default, the covariance is the uncertainty of the emulated model
        function. With `include_noise`, the noise (nugget) of the GPs is
        included, i.e. the uncertainty of a new noisy simulation. PCSK models
        the noise with the statistical errors of the training data, so their
        mean variance is added to the diagonal instead.

        The variance of the discarded PCs is surmise's extravar, which is
        added to the diagonal only and is not corrected for the noise of the
        training data. It is zero for PCSK, so PCSK predicts no uncertainty in
        the directions of the PCs it dropped.

        With ``exp_and_cov_diagonal=True``, the mean is exp(mean) and the
        covariance is diagonal (see `EmulatorBase`).

        Parameters
        ----------
        X : array_like of shape (nsamples, n_parameters)
            Parameter points. A 1D array is treated as a single point.
        return_cov : bool, default=True
            If True, the covariance is returned as well.
        include_noise : bool, default=False
            If True, the noise of the simulations is included in the
            covariance (the nugget of the GPs, for PCSK the variance of the
            statistical errors averaged over the training points, which is
            the same for all points `X`).

        Returns
        -------
        mean : ndarray of shape (nsamples, n_obs)
            Predicted mean.
        cov : ndarray of shape (nsamples, n_obs, n_obs)
            Covariance between the observables. Only returned if `return_cov`
            is True.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        x = np.arange(self.n_obs).reshape(-1, 1)
        pred = self.emu_.predict(x=x, theta=X)

        mean = pred.mean().T
        if not return_cov:
            return np.exp(mean) if self.exp_and_cov_diagonal else mean

        cov = self._full_covariance(pred)
        if include_noise and self.method == "PCSK":
            idx = np.arange(self.n_obs)
            cov[:, idx, idx] += self._sim_noise_var
        elif not include_noise:
            cov = cov - self._noise_covariance()[None, :, :]
            # round-off can make variances slightly negative
            idx = np.arange(self.n_obs)
            cov[:, idx, idx] = np.maximum(cov[:, idx, idx], 0.0)

        if self.exp_and_cov_diagonal:
            # If the emulator is trained on the log of the data, we return the
            # predictions in the original scale with diagonal covariance matrix.
            mean = np.exp(mean)
            std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
            cov = np.zeros_like(cov)
            idx = np.arange(self.n_obs)
            cov[:, idx, idx] = (std * mean) ** 2

        return mean, cov
