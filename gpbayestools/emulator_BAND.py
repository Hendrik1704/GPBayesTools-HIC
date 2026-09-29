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

class EmulatorBAND(EmulatorBase):
    """
    Multidimensional Gaussian Process emulator wrapper for the GP emulators of 
    the BAND collaboration. The number of principal components is chosen by
    surmise, so there is no npc argument.

    With `logTrafo` set to True, the emulator is trained on the log of the
    observables and predict() returns the mean and covariance in log space.
    Experimental data used with the emulator must then be log-transformed as
    well. With `exp_and_cov_diagonal` set to True, predict() returns exp(mean)
    and a diagonal covariance in the original scale of the observables.
    """

    def __init__(self, training_set_path=".", parameter_file="ABCD.txt", 
                 method='PCGP',logTrafo=False,
                 max_rel_uncertainty_data=None, exp_and_cov_diagonal=False,
                 seed=None):
        self.method_ = method
        # surmise (>=1.0.0) requires a global RNG to be set before training
        self.rng_ = np.random.default_rng(seed)
        super().__init__(training_set_path, parameter_file, logTrafo,
                         max_rel_uncertainty_data, exp_and_cov_diagonal)


    def trainEmulator(self, event_mask):
        logging.info('Performing emulator training ...')
        nev, nobs = self.model_data[event_mask, :].shape
        logging.info(
            'Train GP emulators with {} training points ...'.format(nev))
        X = np.arange(nobs).reshape(-1, 1)

        design_points = self.design_points[event_mask, :]

        surmise.set_RNG(self.rng_)
        if self.method_ == 'PCGP':
            self.emu = emulator(x=X,theta=design_points,
                            f=self.model_data[event_mask, :].T,
                            method='PCGP',
                            args={'warnings': True}
                            )
        elif self.method_ == 'PCSK':
            sim_sdev = self.model_data_err[event_mask, :].T

            self.emu = emulator(x=X,theta=design_points,
                                f=self.model_data[event_mask, :].T,
                                method='PCSK',
                                args={'warnings': True, 'simsd': sim_sdev}
                                )
        elif self.method_ == 'PCGPwImpute':
            self.emu = emulator(x=X,theta=design_points,
                                f=self.model_data[event_mask, :].T,
                                method='PCGPwImpute',
                                args={'warnings': True})
        elif self.method_ == 'PCGPwM':
            self.emu = emulator(x=X,theta=design_points,
                                f=self.model_data[event_mask, :].T,
                                method='PCGPwM',
                                args={'warnings': True})
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


    def predict(self,X,return_cov=True):
        """
        Predict model output. Here X is the parameter vector at the prediction
        point.
        """
        x = np.arange(self.nobs).reshape(-1, 1)

        gp = self.emu.predict(x=x,theta=X)

        if self.exp_and_cov_diagonal_:
            # If the emulator is trained on the log of the data, we return the
            # predictions in the original scale with diagonal covariance matrix.
            fpredmean = np.exp(gp.mean().T)
        else:
            fpredmean = gp.mean().T

        fpredcov = self._full_covariance(gp)

        if self.exp_and_cov_diagonal_:
            fcov = np.zeros_like(fpredcov)
            # Extract the diagonal of the covariance matrix for each prediction
            for i in range(fpredcov.shape[0]):
                diagonal_cov = np.zeros((self.nobs, self.nobs))
                fstd = np.sqrt(np.diag(fpredcov[i]))
                np.fill_diagonal(diagonal_cov, (fstd * fpredmean[i])**2)
                fcov[i] = diagonal_cov
            fpredcov = fcov

        if return_cov:
            return (fpredmean, fpredcov)
        else:
            return fpredmean
