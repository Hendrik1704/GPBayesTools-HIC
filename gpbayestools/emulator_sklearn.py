"""
Emulator of the model outputs with scikit-learn GPs of the PCs of the
outputs (training, prediction and validation).

Uses the `scikit-learn <http://scikit-learn.org>`_ implementations of
`principal component analysis (PCA)
<http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
and `Gaussian process regression
<http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_.
"""

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.preprocessing import StandardScaler

from .emulator_base import EmulatorBase, check_npc, number_of_pcs, truncation_signal

logger = logging.getLogger(__name__)


class EmulatorSklearn(EmulatorBase):
    """
    Multidimensional Gaussian process emulator using principal component
    analysis.

    The model training data are standardized (subtract mean and scale to unit
    variance), then transformed through PCA. The first `npc` principal
    components (PCs) are emulated by independent Gaussian processes (GPs).
    The remaining components are not emulated; their variance is added to the
    predicted covariance as truncation covariance (without the noise of the
    training data by default, see `predict`). There is the option to
    switch off the PCA transformation and use the raw data for the Gaussian
    process emulation.

    Parameters
    ----------
    training_set_path : str or path-like
        Path to the pickle file with the training data, a dictionary
        ``{event_id: {'parameter': array (n_parameters,), 'obs': array (2,
        n_obs) with the values and statistical errors}}``.
    parameter_file : str or path-like
        Path to the model parameter file.
    npc : int or float, default=10
        Number of PCs (int >= 1) or fraction of the explained variance (float
        in (0, 1)).
    n_restarts : int, default=0
        Number of restarts of the GP hyperparameter optimizer.
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
    perform_no_pca : bool, default=False
        If True, the PCA transformation is switched off and the raw
        (standardized) data are used for the Gaussian process emulation.
    seed : int or None, default=None
        Random state of the restarts of the GP hyperparameter optimization
        (with n_restarts > 0), for reproducible training.
    alpha : float, default=1e-8
        Value added to the diagonal of the GP kernel matrices in the training.
        It is only meant for numerical stability, the noise of the training
        data is fitted by the WhiteKernel of the GPs. Versions < 3.0.0 used
        alpha = 0.1, which treats a fixed 10% of the variance of each
        (whitened) PC as noise and overestimates the emulator uncertainty.

    Raises
    ------
    ValueError
        If `npc` is out of range, or for invalid training data or options
        (see `EmulatorBase`).
    TypeError
        If `npc` is neither an int nor a float.
    """

    _legacy_attributes = [
        ("npc", "npc_"),
        ("npc_requested_", "npc"),
        ("nrestarts", "n_restarts"),
        ("perform_no_PCA_", "perform_no_pca"),
        ("seed_", "seed"),
        ("alpha_", "alpha"),
        ("scaler", "scaler_"),
        ("pca", "pca_"),
        ("gps", "gps_"),
    ]
    _legacy_defaults = {
        "npc": lambda state: state.get("npc_"),
        "perform_no_pca": False,
        "seed": None,
        "alpha": 0.1,
        "_cov_trunc_signal": lambda state: state.get("_cov_trunc"),
    }

    def __init__(
        self,
        training_set_path,
        parameter_file,
        *,
        npc=10,
        n_restarts=0,
        log_trafo=False,
        max_rel_uncertainty_data=None,
        exp_and_cov_diagonal=False,
        perform_no_pca=False,
        seed=None,
        alpha=1e-8,
    ):
        super().__init__(
            training_set_path,
            parameter_file,
            log_trafo=log_trafo,
            max_rel_uncertainty_data=max_rel_uncertainty_data,
            exp_and_cov_diagonal=exp_and_cov_diagonal,
        )
        self.perform_no_pca = perform_no_pca

        check_npc(npc)
        self.npc = npc
        self.n_restarts = n_restarts
        # random state of the restarts of the GP hyperparameter optimizer
        self.seed = seed
        # value added to the diagonal of the GP kernel matrices in the
        # training, for numerical stability
        self.alpha = alpha

    def _pca_of_all_data(self):
        """
        First npc PCs of all training data. Separate scaler and PCA objects
        are used, so that the trained emulator is not modified.
        """
        scaler = StandardScaler()
        pca = PCA(whiten=True, svd_solver="full")
        Z = pca.fit_transform(scaler.fit_transform(self.model_data))
        npc = number_of_pcs(self.npc, pca.explained_variance_ratio_)
        return Z[:, :npc]

    def output_pca_vs_param(self):
        """
        Return the principal components of all training data.

        The PCA is fitted to all training data, independently of the trained
        emulator, which is not modified.

        Returns
        -------
        design_points : ndarray of shape (n_ev, n_parameters)
            Parameter points of the training data.
        Z : ndarray of shape (n, n_ev)
            The first n PCs at the training points, where n is the number of
            PCs chosen by `npc` for all training data.
        """
        Z = self._pca_of_all_data()
        return (self.design_points, Z.T)

    def train_emulator(self, event_mask, kernel_type="RBF"):
        """
        Train the emulator on the training points selected by `event_mask`.

        The scaler and the PCA are fitted to the selected training data, and
        one GP is fitted to each PC (or each standardized observable with
        ``perform_no_pca=True``). The kernel is the correlation kernel plus a
        homoscedastic noise term (WhiteKernel).

        Parameters
        ----------
        event_mask : ndarray of bool of shape (n_ev,)
            Mask of the training points to use.
        kernel_type : {"RBF", "Matern"}, default="RBF"
            Correlation kernel of the GPs: Gaussian (RBF) or Matern with
            nu = 1.5. The length scales are bounded by 1e-3 and 1e5 times the
            parameter ranges.

        Raises
        ------
        ValueError
            If `kernel_type` is unknown or if `event_mask` is not a boolean
            array of shape (n_ev,).
        """
        # check before the trained emulator is modified
        if kernel_type not in ("RBF", "Matern"):
            raise ValueError(
                f"Unknown kernel type {kernel_type!r}, expected 'RBF' or 'Matern'"
            )
        event_mask = self._check_event_mask(event_mask)
        data_to_use = self.model_data[event_mask, :]
        # Standardize the input data. New scaler and PCA objects are used,
        # so that the previously trained ones are not modified.
        self.scaler_ = StandardScaler()
        standardized_data = self.scaler_.fit_transform(data_to_use)

        if self.perform_no_pca:
            Z = standardized_data
            # one GP per standardized observable
            self.npc_ = Z.shape[1]
        else:
            self.pca_ = PCA(whiten=True, svd_solver="full")
            # Transform data with PCA. Use the first
            # `npc` components but save the full PC transformation for later.
            Z = self.pca_.fit_transform(standardized_data)
            # the PCA has at most min(n_training_points, n_obs) components
            self.npc_ = number_of_pcs(
                self.npc,
                self.pca_.explained_variance_ratio_,
            )
            Z = Z[:, : self.npc_]

            logger.info(
                f"Using {self.npc_} PCs, which explain "
                f"{self.pca_.explained_variance_ratio_[: self.npc_].sum():.5f} "
                "of the variance"
            )

        n_ev = Z.shape[0]
        targets = "standardized observables" if self.perform_no_pca else "PCs"
        logger.info(
            f"Training {Z.shape[1]} GPs ({kernel_type} kernel) for the {targets} "
            f"with {n_ev} training points ..."
        )

        design_points = self.design_points[event_mask, :]

        # Define kernel (covariance function):
        # correlation kernel (RBF or Matern) plus a noise term.
        # the length scales start at the parameter ranges and are bounded by
        # 1e-3 and 1e5 times the ranges
        ptp = self.design_max - self.design_min
        length_scale_bounds = np.outer(ptp, (1e-3, 1e5))
        if kernel_type == "RBF":
            corr_kern = 1.0 * kernels.RBF(
                length_scale=ptp, length_scale_bounds=length_scale_bounds
            )
        elif kernel_type == "Matern":
            corr_kern = 1.0 * kernels.Matern(
                length_scale=ptp, length_scale_bounds=length_scale_bounds, nu=1.5
            )
        else:
            raise AssertionError(kernel_type)

        # homoscedastic noise kernel
        hom_white_kern = kernels.WhiteKernel(
            noise_level=0.05, noise_level_bounds=(1e-6, 1e2)
        )
        kernel = corr_kern + hom_white_kern

        # Fit a GP (optimize the kernel hyperparameters) to each PC.
        self.gps_ = [
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts,
                copy_X_train=False,
                random_state=self.seed,
            ).fit(design_points, z)
            for z in Z.T
        ]
        for n, (gp, z) in enumerate(zip(self.gps_, Z.T, strict=True)):
            evr = (
                ""
                if self.perform_no_pca
                else f"{self.pca_.explained_variance_ratio_[n]:.5f} of the variance, "
            )
            logger.info(
                f"GP {n + 1}: {evr}"
                f"LML = {gp.log_marginal_likelihood_value_:.5g}, "
                f"R^2 = {gp.score(design_points, z):.4f}, kernel: {gp.kernel_}"
            )

        if not self.perform_no_pca:
            # Construct the full linear transformation matrix, which is just the
            # PC matrix with the first axis multiplied by the explained standard
            # deviation of each PC and the second axis multiplied by the
            # standardization scale factor of each observable.
            self._trans_matrix = (
                self.pca_.components_
                * np.sqrt(self.pca_.explained_variance_[:, np.newaxis])
                * self.scaler_.scale_
            )

            # Pre-calculate some arrays for inverse transforming the predictive
            # variance (from PC space to physical space).

            # Assuming the PCs are uncorrelated, the transformation is
            #
            #   cov_ij = sum_k A_ki var_k A_kj
            #
            # where A is the trans matrix and var_k is the variance of the kth
            # PC.
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty

            # Compute the partial transformation for the first `npc` components
            # that are actually emulated.
            A = self._trans_matrix[: self.npc_]
            self._var_trans = np.einsum("ki,kj->kij", A, A, optimize=False).reshape(
                self.npc_, self.n_obs**2
            )

            # Compute the covariance matrix of the PCs that are not emulated
            # (truncation error). The whitened components have variance 1.
            B = self._trans_matrix[self.npc_ :]
            self._cov_trunc = np.dot(B.T, B)

            # The truncation covariance also contains the statistical noise of
            # the training data in the discarded PC directions. Its signal
            # part is used for predictions of the model function
            # (include_noise=False).
            scale = self.scaler_.scale_
            # covariances in standardized units
            err_scaled = self.model_data_err[event_mask, :] / scale
            noise_cov_scaled = np.diag(np.mean(err_scaled**2, axis=0))
            trunc_cov_scaled = self._cov_trunc / np.outer(scale, scale)
            self._cov_trunc_signal = truncation_signal(
                trunc_cov_scaled, noise_cov_scaled
            ) * np.outer(scale, scale)

            # Add small term to diagonal for numerical stability.
            self._cov_trunc.flat[:: self.n_obs + 1] += 1e-4 * self.scaler_.var_
            self._cov_trunc_signal.flat[:: self.n_obs + 1] += 1e-4 * self.scaler_.var_

        logger.info("Emulator training finished")

    def _inverse_transform(self, Z):
        """
        Inverse transform principal components to observables.

        `Z` has shape (..., npc), the result `Y` has shape (..., n_obs).
        """
        Y = np.dot(Z, self._trans_matrix[: Z.shape[-1]])
        Y += self.scaler_.mean_
        return Y

    @staticmethod
    def _gp_noise(kernel):
        """Noise variance of the WhiteKernel terms of a fitted kernel."""
        if isinstance(kernel, kernels.WhiteKernel):
            return kernel.noise_level
        if isinstance(kernel, kernels.Sum):
            return EmulatorSklearn._gp_noise(kernel.k1) + EmulatorSklearn._gp_noise(
                kernel.k2
            )
        return 0.0

    def predict(self, X, return_cov=True, include_noise=False):
        """
        Predict model output at the parameter points `X`.

        By default, the covariance is the uncertainty of the emulated model
        function. With `include_noise`, the noise fitted by the GPs
        (WhiteKernel) is included, i.e. the uncertainty of a new noisy
        simulation.

        The covariance also contains the covariance of the discarded PCs
        (truncation). By default, the statistical noise of the training data
        in these directions is removed from it, with `include_noise` the full
        truncation covariance is used.

        With ``exp_and_cov_diagonal=True``, the mean is exp(mean) and the
        covariance is diagonal (see `EmulatorBase`).

        Parameters
        ----------
        X : array_like of shape (nsamples, n_parameters)
            Parameter points. A 1D array is treated as a single point.
        return_cov : bool, default=True
            If True, the covariance is returned as well.
        include_noise : bool, default=False
            If True, the noise fitted by the GPs is included in the
            covariance.

        Returns
        -------
        mean : ndarray of shape (nsamples, n_obs)
            Predicted mean.
        cov : ndarray of shape (nsamples, n_obs, n_obs)
            Covariance between the observables. Only returned if `return_cov`
            is True.
        """
        X = np.atleast_2d(X)
        # only the variances of the GPs at the points are needed, not their
        # covariances between the points
        gp_pred = [gp.predict(X, return_std=return_cov) for gp in self.gps_]
        if return_cov:
            gp_mean, gp_std = zip(*gp_pred, strict=True)
        else:
            gp_mean = gp_pred
        # shape (nsamples, npc)
        gp_mean = np.column_stack(gp_mean)

        if not self.perform_no_pca:
            mean = self._inverse_transform(gp_mean)
        else:
            mean = self.scaler_.inverse_transform(gp_mean)

        if self.exp_and_cov_diagonal:
            mean = np.exp(mean)

        if return_cov:
            # predictive variances of the GPs, shape (nsamples, npc)
            gp_var = np.column_stack(gp_std) ** 2
            if not include_noise:
                # the predictive variance of sklearn includes the WhiteKernel
                noise = np.array([self._gp_noise(gp.kernel_) for gp in self.gps_])
                gp_var = np.maximum(gp_var - noise, 0.0)

            if not self.perform_no_pca:
                # Compute the covariance at each sample point using the
                # pre-calculated arrays (see train_emulator).
                cov = np.dot(gp_var, self._var_trans).reshape(
                    X.shape[0], self.n_obs, self.n_obs
                )
                if include_noise:
                    cov += self._cov_trunc
                else:
                    cov += self._cov_trunc_signal
            else:
                # Create a covariance matrix for each sample point from gp_var,
                # transformed from standardized units back to observable units
                cov = np.zeros((X.shape[0], self.n_obs, self.n_obs))
                for i in range(X.shape[0]):
                    cov[i] = np.diag(gp_var[i] * self.scaler_.var_)

            if self.exp_and_cov_diagonal:
                # If the emulator is trained on the log of the data, we return
                # the predictions in the original scale with diagonal
                # covariance matrix.
                std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
                cov = np.zeros_like(cov)
                idx = np.arange(self.n_obs)
                cov[:, idx, idx] = (std * mean) ** 2

            return mean, cov
        else:
            return mean
