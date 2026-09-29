"""
Training for Gaussian process emulators.

Uses the `Gaussian process regression
<https://surmise.readthedocs.io/en/latest/index.html>`_ implemented by the BAND
collaboration.
"""

import logging
import numpy as np
import surmise
from surmise.emulation import emulator

from .emulator_base import EmulatorBase

logger = logging.getLogger(__name__)


class EmulatorBAND(EmulatorBase):
    """
    Multidimensional Gaussian Process emulator wrapper for the GP emulators of
    the BAND collaboration. The number of principal components is chosen by
    surmise, so there is no npc argument.

    With `log_trafo` set to True, the emulator is trained on the log of the
    observables and predict() returns the mean and covariance in log space.
    Experimental data used with the emulator must then be log-transformed as
    well. With `exp_and_cov_diagonal` set to True, predict() returns exp(mean)
    and a diagonal covariance in the original scale of the observables.
    """

    _legacy_attributes = [("method_", "method"), ("rng_", "_rng"), ("emu", "emu_")]
    _legacy_defaults = {"seed": None, "_rng": lambda state: np.random.default_rng()}

    def __init__(
        self,
        training_set_path=".",
        parameter_file="ABCD.txt",
        method="PCGP",
        log_trafo=False,
        max_rel_uncertainty_data=None,
        exp_and_cov_diagonal=False,
        seed=None,
    ):
        self.method = method
        # surmise (>=1.0.0) requires a global RNG to be set before training
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        super().__init__(
            training_set_path,
            parameter_file,
            log_trafo,
            max_rel_uncertainty_data,
            exp_and_cov_diagonal,
        )

    def train_emulator(self, event_mask):
        logger.info("Performing emulator training ...")
        nev, nobs = self.model_data[event_mask, :].shape
        logger.info("Train GP emulators with {} training points ...".format(nev))
        X = np.arange(nobs).reshape(-1, 1)

        design_points = self.design_points[event_mask, :]

        surmise.set_RNG(self._rng)
        if self.method == "PCGP":
            self.emu_ = emulator(
                x=X,
                theta=design_points,
                f=self.model_data[event_mask, :].T,
                method="PCGP",
                args={"warnings": True},
            )
        elif self.method == "PCSK":
            sim_sdev = self.model_data_err[event_mask, :].T

            self.emu_ = emulator(
                x=X,
                theta=design_points,
                f=self.model_data[event_mask, :].T,
                method="PCSK",
                args={"warnings": True, "simsd": sim_sdev},
            )
        elif self.method == "PCGPwImpute":
            self.emu_ = emulator(
                x=X,
                theta=design_points,
                f=self.model_data[event_mask, :].T,
                method="PCGPwImpute",
                args={"warnings": True},
            )
        elif self.method == "PCGPwM":
            self.emu_ = emulator(
                x=X,
                theta=design_points,
                f=self.model_data[event_mask, :].T,
                method="PCGPwM",
                args={"warnings": True},
            )
        else:
            raise ValueError("Requested method not implemented!")

    def _full_covariance(self, gp):
        """
        Covariance matrices of the surmise prediction `gp` with shape
        (ntheta, nobs, nobs). surmise's covx() only contains the variance of
        the emulated PCs, while var() also contains the variance of the
        discarded PCs, which is added to the diagonal here.
        """
        cov = np.array(gp.covx())
        missing_var = gp.var().T - np.diagonal(cov, axis1=1, axis2=2)
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
        Predict model output. Here X is the parameter vector at the prediction
        point.

        By default, the covariance is the uncertainty of the emulated model
        function. With `include_noise`, the noise (nugget) of the GPs is
        included, i.e. the uncertainty of a new noisy simulation.

        The variance of the discarded PCs is surmise's extravar, which is
        zero for PCSK and not corrected for the noise of the training data.
        """
        x = np.arange(self.nobs).reshape(-1, 1)

        gp = self.emu_.predict(x=x, theta=X)

        if self.exp_and_cov_diagonal:
            # If the emulator is trained on the log of the data, we return the
            # predictions in the original scale with diagonal covariance matrix.
            fpredmean = np.exp(gp.mean().T)
        else:
            fpredmean = gp.mean().T

        fpredcov = self._full_covariance(gp)
        if not include_noise:
            fpredcov = fpredcov - self._noise_covariance()[None, :, :]

        if self.exp_and_cov_diagonal:
            fcov = np.zeros_like(fpredcov)
            # Extract the diagonal of the covariance matrix for each prediction
            for i in range(fpredcov.shape[0]):
                diagonal_cov = np.zeros((self.nobs, self.nobs))
                fstd = np.sqrt(np.diag(fpredcov[i]))
                np.fill_diagonal(diagonal_cov, (fstd * fpredmean[i]) ** 2)
                fcov[i] = diagonal_cov
            fpredcov = fcov

        if return_cov:
            return (fpredmean, fpredcov)
        else:
            return fpredmean
