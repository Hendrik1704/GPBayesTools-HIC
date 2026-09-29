"""
Sparse variational Gaussian process emulator.

EmulatorSparseGP provides the interface of the other emulators of the package
for the PCA-reduced sparse variational GP emulators of svgp.py.
PCASparseGPEmulator and PCASparseGPEnsemble can also be imported from this
module.
"""

import logging

import jax
import numpy as np

from .emulator_base import EmulatorBase, check_npc
from .svgp import PCASparseGPEmulator, PCASparseGPEnsemble

__all__ = ["EmulatorSparseGP", "PCASparseGPEmulator", "PCASparseGPEnsemble"]

logger = logging.getLogger(__name__)


# =============================================================================
# EmulatorSparseGP — high-level wrapper (same interface as EmulatorBAND)
# =============================================================================


class EmulatorSparseGP(EmulatorBase):
    """
    Sparse GP emulator with the interface of the other emulators.

    High-level wrapper around PCASparseGPEmulator / PCASparseGPEnsemble that
    follows the same interface as EmulatorBAND. It can operate in two modes
    controlled by ``n_ensemble``:

    * ``n_ensemble=1`` — single PCASparseGPEmulator.
    * ``n_ensemble>1`` — PCASparseGPEnsemble of that many members.

    With ``log_trafo=True``, the emulator is trained on the log of the
    observables and predict() returns the mean and covariance in log space.
    Experimental data used with the emulator must then be log-transformed as
    well. With ``exp_and_cov_diagonal=True``, predict() returns the predictions
    in the original scale of the observables (see __init__).
    """

    def __init__(
        self,
        training_set_path=".",
        parameter_file="ABCD.txt",
        npc=0.999,
        n_inducing=200,
        n_ensemble=1,
        init_strategy="maxmin",
        bootstrap=False,
        log_trafo=False,
        max_rel_uncertainty_data=None,
        exp_and_cov_diagonal=False,
        seed=None,
    ):
        """
        Initialize the emulator and load the training data.

        Parameters
        ----------
        training_set_path : str
            Path to the pickle file with the training data (default '.').
        parameter_file : str
            Path to the model parameter file (default 'ABCD.txt').
        npc : float or int
            Number of principal components: int for a fixed number, float in
            (0, 1) for the fraction of the explained variance (default 0.999).
            Passed to the inner emulator as n_pc.
        n_inducing : int
            Number of inducing points (default 200). Passed to the inner
            emulator as M.
        n_ensemble : int
            Number of ensemble members (default 1). Use 1 for a single
            emulator.
        init_strategy : str
            Inducing-point initialization strategy, see PCASparseGPEmulator
            (default 'maxmin').
        bootstrap : bool
            Bootstrap resampling for the ensemble members (default False).
            Only used with n_ensemble > 1.
        log_trafo : bool
            If True, the emulator is trained on the log of the outputs and
            predict() returns the mean and covariance in log space, like the
            other emulators (default False). The experimental data used with
            the emulator must then also be log-transformed.
        max_rel_uncertainty_data : float or None
            Maximum relative statistical uncertainty; training points with
            larger values are discarded. None (default) disables this
            filter.
        exp_and_cov_diagonal : bool
            Only with log_trafo=True: predict() returns the predictions
            transformed back to the original scale, exp(mean) and the
            covariance cov_ij * exp(mean_i) * exp(mean_j) (delta method).
            Unlike the other emulators, the correlations between the
            observables are kept (default False).
        seed : int or None
            Seed for the random numbers of the training (inducing points,
            mini-batches, ensemble members). None (default) uses fixed
            default keys.

        Raises
        ------
        ValueError
            If npc is out of range, or if exp_and_cov_diagonal is True
            without log_trafo.
        TypeError
            If npc is neither an int nor a float.
        """
        check_npc(npc)
        if n_ensemble < 1:
            raise ValueError(f"n_ensemble must be >= 1, got {n_ensemble}")
        self.npc = npc
        self.seed = seed
        self.n_inducing = n_inducing
        self.n_ensemble = n_ensemble
        self.init_strategy = init_strategy
        self.bootstrap = bootstrap
        super().__init__(
            training_set_path,
            parameter_file,
            log_trafo,
            max_rel_uncertainty_data,
            exp_and_cov_diagonal,
        )

    # -------------------------
    # Training
    # -------------------------
    def _key(self):
        """JAX random key from the seed, or None for the default keys."""
        seed = self.seed
        return None if seed is None else jax.random.PRNGKey(seed)

    def train_emulator(self, event_mask, **fit_kwargs):
        """
        Train the (ensemble) emulator on the masked subset of training data.

        The statistical errors of the training data are passed as Y_err.

        Parameters
        ----------
        event_mask : array of bool (nev,)
            True entries are included in the training.
        **fit_kwargs
            Forwarded to PCASparseGPEmulator.fit() or
            PCASparseGPEnsemble.fit(). verbose_members is dropped for a single
            emulator.
        """
        X = self.design_points[event_mask, :]
        Y = self.model_data[event_mask, :]
        Y_err = self.model_data_err[event_mask, :]
        npc = self.npc

        if self.n_ensemble <= 1:
            # verbose_members only exists for the ensemble
            fit_kwargs = {k: v for k, v in fit_kwargs.items() if k != "verbose_members"}
            self.emu_ = PCASparseGPEmulator(
                n_pc=npc,
                M=self.n_inducing,
                key=self._key(),
                init_strategy=self.init_strategy,
            )
            self.emu_.fit(X, Y, Y_err=Y_err, **fit_kwargs)
            self.npc_ = int(self.emu_.n_pc)
        else:
            self.emu_ = PCASparseGPEnsemble(
                n_ensemble=self.n_ensemble,
                n_pc=npc,
                M=self.n_inducing,
                base_key=self._key(),
                init_strategy=self.init_strategy,
                bootstrap=self.bootstrap,
            )
            self.emu_.fit(X, Y, Y_err=Y_err, **fit_kwargs)
            self.npc_ = int(self.emu_.members[0].n_pc)

    # -------------------------
    # Prediction
    # -------------------------
    def predict(
        self,
        X,
        return_cov=True,
        include_noise=False,
        include_truncation=True,
        include_pca_sampling=False,
        include_obs_noise=None,
    ):
        """
        Predict the model output at the parameter points ``X``.

        Parameters
        ----------
        X : array (N_test, nparameters) or (nparameters,)
            Parameter points in the original (non-normalized) space.
        return_cov : bool
            If True, also return the covariance matrices (default True).
        include_noise : bool
            Include the learned nugget in the predictive variance (default
            False). Together with the default of `include_obs_noise`, the
            covariance is then the uncertainty of a new noisy simulation, as
            in the other emulators.
        include_truncation : bool
            Include the PCA truncation covariance (default True).
        include_pca_sampling : bool
            Include the finite-data PCA sampling uncertainty (default False).
        include_obs_noise : bool or None
            Include the statistical noise propagated from Y_err in the
            predictions. None (default) uses the value of `include_noise`.
            For MCMC calibration against experimental means, keep this False.

        Returns
        -------
        mean : array (N_test, nobs)
            Predictive mean; exp(mean) with exp_and_cov_diagonal=True.
        cov : array (N_test, nobs, nobs)
            Predictive covariance, only returned if return_cov=True. With
            exp_and_cov_diagonal=True, transformed with the delta method.

        Raises
        ------
        RuntimeError
            If train_emulator() has not been called.
        """
        if not hasattr(self, "emu_"):
            raise RuntimeError("Call train_emulator() before predict().")

        if include_obs_noise is None:
            include_obs_noise = include_noise
        X = np.atleast_2d(X)
        mean, cov = self.emu_.predict(
            X,
            include_noise=include_noise,
            include_truncation=include_truncation,
            include_pca_sampling=include_pca_sampling,
            include_obs_noise=include_obs_noise,
        )
        mean = np.array(mean)
        cov = np.array(cov)

        if self.exp_and_cov_diagonal:
            # delta method: Cov_y[i,j] = exp(mu_i) * Cov_log[i,j] * exp(mu_j),
            # which keeps the correlations between the observables
            mean = np.exp(mean)
            cov = cov * mean[:, :, None] * mean[:, None, :]

        if return_cov:
            return mean, cov
        return mean
