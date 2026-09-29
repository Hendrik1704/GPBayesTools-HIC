"""
Training for Gaussian process emulators.

Uses the `Gaussian process regression with heteroskedastic emulator
<https://hetgpy.readthedocs.io/en/v1.0.4/>`_ implemented in the `hetgpy` package.
"""

import logging
import pickle

import numpy as np
from hetgpy import hetGP, homGP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .emulator_base import EmulatorBase, check_npc, truncation_signal

logger = logging.getLogger(__name__)


class EmulatorHetGP(EmulatorBase):
    """
    Emulator with heteroskedastic GPs of the hetgpy package.

    The GPs emulate the principal components of the (standardized)
    observables.

    Parameters
    ----------
    training_set_path : str, default="."
        Path to the pickle file with the training data, a dictionary
        ``{event_id: {'parameter': array (nparameters,), 'obs': array (2,
        nobs) with the values and statistical errors}}``.
    parameter_file : str, default="ABCD.txt"
        Path to the model parameter file.
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
        original scale of the observables. Requires ``log_trafo=True``.
    npc : int or float, default=0.99
        Number of PCs (int >= 1) or fraction of the explained variance (float
        in (0, 1)).

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
        ("targetVariance", "npc"),
        ("outputScaler", "output_scaler_"),
        ("outputPCA", "output_pca_"),
        ("model_data_pca", "model_data_pca_"),
    ]

    def __init__(
        self,
        training_set_path=".",
        parameter_file="ABCD.txt",
        log_trafo=False,
        max_rel_uncertainty_data=None,
        exp_and_cov_diagonal=False,
        npc=0.99,
    ):
        super().__init__(
            training_set_path,
            parameter_file,
            log_trafo,
            max_rel_uncertainty_data,
            exp_and_cov_diagonal,
        )

        # The outputs are standardized and transformed with a PCA in
        # train_emulator(), keeping npc PCs (int) or the PCs explaining the
        # fraction npc (float) of the variance. The GP emulators are then
        # trained on the resulting principal components.
        check_npc(npc)
        self.npc = npc

    def _fit_output_pca(self, data, data_err=None):
        """
        Fit the output standardization and PCA to the training data.

        Also compute the truncation covariance. `data` are the training data
        and `data_err` their statistical errors.
        """
        logger.info("Performing output PCA for hetGP emulator ...")
        self.output_scaler_ = StandardScaler()
        standardized_outputs = self.output_scaler_.fit_transform(data)
        npc = self.npc
        if isinstance(npc, (int, np.integer)) and npc > min(data.shape):
            logger.warning(
                f"Only {min(data.shape)} PCs available, using npc = {min(data.shape)}"
            )
            npc = min(data.shape)
        self.output_pca_ = PCA(n_components=npc)
        self.model_data_pca_ = self.output_pca_.fit_transform(standardized_outputs)
        self.npc_ = self.output_pca_.n_components_
        self._compute_truncation_cov(data, self.model_data_pca_, data_err)
        logger.info(
            f"Output PCA uses {self.npc_} PCs to explain "
            f"{100.0 * self.output_pca_.explained_variance_ratio_.sum():.1f}% "
            "of the variance ..."
        )

    def _compute_truncation_cov(self, data, data_pca, data_err=None):
        """
        Compute the covariance of the PCs discarded by the output PCA.

        The covariance is in observable units. It is added to the predicted
        covariance, since the emulator cannot resolve this part of the
        variance. `data` are the training data, `data_pca` their principal
        components and `data_err` their statistical errors.

        The truncation covariance also contains the statistical noise of the
        training data in the discarded PC directions. Its signal part without
        this noise, _cov_trunc_signal, is used for predictions of the model
        function (include_noise=False).
        """
        standardized_outputs = self.output_scaler_.transform(data)
        residuals = standardized_outputs - self.output_pca_.inverse_transform(data_pca)
        scales = self.output_scaler_.scale_
        self._cov_trunc = np.cov(residuals, rowvar=False) * np.outer(scales, scales)
        self._cov_trunc_signal = self._cov_trunc
        if data_err is not None:
            noise_std = np.diag(np.mean((data_err / scales) ** 2, axis=0))
            trunc_std = np.cov(residuals, rowvar=False)
            self._cov_trunc_signal = truncation_signal(trunc_std, noise_std) * np.outer(
                scales, scales
            )

    def __getstate__(self):
        """
        Prepare a pickleable state.

        The fitted hetgpy models can be serialized with the standard
        ``pickle`` module, but not with ``dill``, which is used to save the
        emulators. We therefore store the models as a ``pickle`` byte string,
        which ``dill`` can serialize, so that the unpickled emulator contains
        exactly the trained models.
        """
        state = self.__dict__.copy()
        emu = state.pop("emu_", None)
        if emu is not None:
            state["_emu_list_pickle"] = pickle.dumps(emu)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        state = self._migrate_legacy_state(state)
        emu_list_pickle = state.pop("_emu_list_pickle", None)
        hyperparams = state.pop("_gp_hyperparams", None)
        state.pop("emu_list", None)
        self.__dict__.update(state)
        if "_cov_trunc" not in self.__dict__ and "model_data_pca_" in self.__dict__:
            # older versions fitted the output PCA to all training data
            self._compute_truncation_cov(self.model_data, self.model_data_pca_)
        if "_cov_trunc" in self.__dict__ and "_cov_trunc_signal" not in self.__dict__:
            self._cov_trunc_signal = self._cov_trunc
        if emu_list_pickle is not None:
            self.emu_ = pickle.loads(emu_list_pickle)
        elif hyperparams is not None:
            # emulators saved with older versions only contain the
            # hyperparameters of the GP models
            logger.warning(
                "Emulator saved with an older version: rebuilding "
                "GP models from saved hyperparameters. The "
                "predictions can differ from the trained models. "
                "Save the emulator again to avoid this."
            )
            self._rebuild_from_hyperparams(hyperparams)
        # otherwise the emulator was not trained and stays untrained

    def _rebuild_from_hyperparams(self, hyperparams, maxit=0):
        """
        Rebuild GP models of emulators saved with older versions.

        These emulators only contain the hyperparameters of the GP models,
        which are used as initial values of a new fit to all training points.
        With the default ``maxit=0`` the saved hyperparameters are used
        without optimization, the predictions can still differ from the
        originally trained models.

        Parameters
        ----------
        hyperparams : list of dict
            One dict per principal component with 'model_type' ('hetGP' or
            'homGP'), 'theta', and 'g' for homGP models or 'Delta' for hetGP
            models.
        maxit : int, default=0
            Maximum number of optimizer iterations.
        """
        design_points_masked = self.design_points
        data_pca_masked = self.model_data_pca_

        self.emu_ = []
        for j in range(self.npc_):
            Z_train = data_pca_masked[:, j]
            hp = hyperparams[j]

            if hp.get("model_type", "hetGP") == "homGP":
                # Rebuild as a homoskedastic GP
                model = homGP()
                model.mleHomGP(
                    X=design_points_masked,
                    Z=Z_train,
                    init={"theta": hp["theta"], "g": hp["g"]},
                    settings={"return_Ki": True},
                    covtype="Matern3_2",
                    maxit=maxit,
                )
            else:
                # Rebuild as a heteroskedastic GP
                init_dict = {"theta": hp["theta"], "Delta": hp["Delta"]}
                model = hetGP()
                model.mleHetGP(
                    X=design_points_masked,
                    Z=Z_train,
                    init=init_dict,
                    noiseControl={
                        "g_min": 1e-8,
                        "g_max": 100,
                        "k_theta_g_bounds": (1, 100),
                        "g_bounds": (1e-6, 1),
                    },
                    settings={"return_matrices": True},
                    covtype="Matern3_2",
                    maxit=maxit,
                )
            self.emu_.append(model)

        logger.info(f"Rebuilt {self.npc_} GP models via warm-start (maxit={maxit}).")

    def train_emulator(self, event_mask):
        """
        Train the emulator on the training points selected by `event_mask`.

        The output standardization and PCA are fitted to the selected training
        points only, then one hetGP model is trained per PC.

        Parameters
        ----------
        event_mask : array_like of bool of shape (nev,)
            Mask of the training points to use.
        """
        logger.info("Performing emulator training ...")
        # Subselect training data
        event_mask = np.asarray(event_mask, dtype=bool)
        design_points_masked = self.design_points[event_mask, :]
        # fit the output PCA only to the training points
        self._fit_output_pca(
            self.model_data[event_mask, :], self.model_data_err[event_mask, :]
        )
        data_pca_masked = self.model_data_pca_

        nev_train = design_points_masked.shape[0]
        logger.info(
            f"Train hetGP emulators for {nev_train} training points and "
            f"{self.npc_} PCs ..."
        )

        # Train one hetGP model per principal component of the outputs.
        self.emu_ = []
        for j in range(self.npc_):
            Z_train = data_pca_masked[:, j]
            model = hetGP()
            model.mleHetGP(
                X=design_points_masked,
                Z=Z_train,
                settings={"return_matrices": True},
                covtype="Matern3_2",
                maxit=100,
            )
            self.emu_.append(model)

    def predict(self, X, return_cov=True, include_noise=False):
        """
        Predict model output at the parameter points `X`.

        By default, the covariance is the uncertainty of the emulated model
        function. With `include_noise`, the noise variance estimated by the
        hetGP models (nugs) is included, i.e. the uncertainty of a new noisy
        simulation.

        Parameters
        ----------
        X : array_like of shape (nsamples, nparameters)
            Parameter points. A 1D array is treated as a single point.
        return_cov : bool, default=True
            If True, the covariance is returned as well.
        include_noise : bool, default=False
            If True, the noise variance estimated by the hetGP models is
            included in the covariance.

        Returns
        -------
        mean : ndarray of shape (nsamples, nobs)
            Predicted mean.
        cov : ndarray of shape (nsamples, nobs, nobs)
            Covariance between the observables. Only returned if `return_cov`
            is True.
        """
        X = np.atleast_2d(X)
        n_theta = X.shape[0]

        # Predict principal components at given parameter points
        pc_means = np.zeros((self.npc_, n_theta))
        pc_vars = np.zeros((self.npc_, n_theta))

        for j, model in enumerate(self.emu_):
            pred = model.predict(x=X)
            mean_j = np.asarray(pred["mean"]).reshape(-1)
            var_j = np.asarray(pred["sd2"]).reshape(-1)
            if include_noise and "nugs" in pred:
                var_j = var_j + np.asarray(pred["nugs"]).reshape(-1)
            pc_means[j, :] = mean_j
            pc_vars[j, :] = var_j

        # Reconstruct observables from PCs
        Z_pred = pc_means.T  # (n_theta, npc)
        standardized_pred = self.output_pca_.inverse_transform(Z_pred)
        Y_pred = self.output_scaler_.inverse_transform(
            standardized_pred
        )  # (n_theta, nobs)

        # Build covariance matrices in observable space
        components = self.output_pca_.components_  # (npc, nobs)
        W = components.T  # (nobs, npc)
        scales = self.output_scaler_.scale_  # (nobs,)
        D = np.diag(scales)

        covs = np.zeros((n_theta, self.nobs, self.nobs))
        for k in range(n_theta):
            var_z = np.maximum(pc_vars[:, k], 0.0)
            Sigma_S = W @ np.diag(var_z) @ W.T
            Sigma_Y = D @ Sigma_S @ D
            if include_noise:
                covs[k] = Sigma_Y + self._cov_trunc
            else:
                covs[k] = Sigma_Y + self._cov_trunc_signal

        fpredmean = Y_pred
        fpredcov = covs

        if self.exp_and_cov_diagonal:
            # If the emulator is trained on the log of the data, we return the
            # predictions in the original scale with diagonal covariance matrix.
            fpredmean = np.exp(fpredmean)

            fcov = np.zeros((n_theta, self.nobs, self.nobs))
            for i in range(n_theta):
                diagonal_cov = np.zeros((self.nobs, self.nobs))
                fstd = np.sqrt(np.diag(fpredcov[i]))
                np.fill_diagonal(diagonal_cov, (fstd * fpredmean[i]) ** 2)
                fcov[i] = diagonal_cov
            fpredcov = fcov

        if return_cov:
            return (fpredmean, fpredcov)
        else:
            return fpredmean
