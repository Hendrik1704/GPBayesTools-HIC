"""
Training for Gaussian process emulators.

Uses the `Gaussian process regression with heteroskedastic emulator
<https://hetgpy.readthedocs.io/en/v1.0.4/>`_ implemented in the `hetgpy` package.
"""

import logging
import numpy as np
import pickle
from hetgpy import hetGP, homGP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .emulator_base import EmulatorBase, check_npc

class EmulatorHETGPy(EmulatorBase):
    """
    Emulator with heteroskedastic GPs of the hetgpy package for the principal
    components of the (standardized) observables. `npc` is the number of PCs
    (int) or the fraction of the explained variance (float in (0, 1),
    default 0.99).

    With `logTrafo` set to True, the emulator is trained on the log of the
    observables and predict() returns the mean and covariance in log space.
    Experimental data used with the emulator must then be log-transformed as
    well. With `exp_and_cov_diagonal` set to True, predict() returns exp(mean)
    and a diagonal covariance in the original scale of the observables.
    """
    def __init__(self, training_set_path=".", parameter_file="ABCD.txt",
                 logTrafo=False, max_rel_uncertainty_data=None, 
                 exp_and_cov_diagonal=False, npc=0.99):
        super().__init__(training_set_path, parameter_file, logTrafo,
                         max_rel_uncertainty_data, exp_and_cov_diagonal)

        # The outputs are standardized and transformed with a PCA in
        # trainEmulator(), keeping npc PCs (int) or the PCs explaining the
        # fraction npc (float) of the variance. The GP emulators are then
        # trained on the resulting principal components.
        check_npc(npc)
        self.npc_requested_ = npc

    def _fit_output_pca(self, data):
        """Fit the output standardization and PCA to the training data
        `data`, and compute the truncation covariance."""
        logging.info("Performing output PCA for hetGP emulator ...")
        self.outputScaler = StandardScaler()
        standardized_outputs = self.outputScaler.fit_transform(data)
        # emulators saved with older versions had a fixed targetVariance
        npc = getattr(self, 'npc_requested_', getattr(self, 'targetVariance', 0.99))
        if isinstance(npc, (int, np.integer)) and npc > min(data.shape):
            logging.warning('Only {} PCs available, using npc = {}'.format(
                min(data.shape), min(data.shape)))
            npc = min(data.shape)
        self.outputPCA = PCA(n_components=npc)
        self.model_data_pca = self.outputPCA.fit_transform(standardized_outputs)
        self.npc = self.outputPCA.n_components_
        self._compute_truncation_cov(data, self.model_data_pca)
        logging.info(
            "Output PCA uses {} PCs to explain {:.1f}% of the variance ...".format(
                self.npc, 100.0 * self.outputPCA.explained_variance_ratio_.sum()
            )
        )

    def _compute_truncation_cov(self, data, data_pca):
        """Covariance of the PCs discarded by the output PCA in observable
        units. It is added to the predicted covariance, since the emulator
        cannot resolve this part of the variance. `data` are the training
        data and `data_pca` their principal components."""
        standardized_outputs = self.outputScaler.transform(data)
        residuals = standardized_outputs - self.outputPCA.inverse_transform(
            data_pca)
        scales = self.outputScaler.scale_
        self._cov_trunc = (np.cov(residuals, rowvar=False)
                           * np.outer(scales, scales))

    def __getstate__(self):
        """Prepare a pickleable state.

        The fitted hetgpy models can be serialized with the standard
        ``pickle`` module, but not with ``dill``, which is used to save the
        emulators. We therefore store the models as a ``pickle`` byte string,
        which ``dill`` can serialize, so that the unpickled emulator contains
        exactly the trained models.
        """
        state = self.__dict__.copy()
        emu_list = state.pop("emu_list", None)
        if emu_list is not None:
            state["_emu_list_pickle"] = pickle.dumps(emu_list)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        emu_list_pickle = state.pop("_emu_list_pickle", None)
        hyperparams = state.pop("_gp_hyperparams", None)
        self.__dict__.update(state)
        if ("_cov_trunc" not in self.__dict__
                and "model_data_pca" in self.__dict__):
            # older versions fitted the output PCA to all training data
            self._compute_truncation_cov(self.model_data, self.model_data_pca)
        if emu_list_pickle is not None:
            self.emu_list = pickle.loads(emu_list_pickle)
        elif hyperparams is not None:
            # emulators saved with older versions only contain the
            # hyperparameters of the GP models
            logging.warning("Emulator saved with an older version: rebuilding "
                            "GP models from saved hyperparameters. The "
                            "predictions can differ from the trained models. "
                            "Save the emulator again to avoid this.")
            self._rebuild_from_hyperparams(hyperparams)
        # otherwise the emulator was not trained and stays untrained

    def _rebuild_from_hyperparams(self, hyperparams, maxit=0):
        """Rebuild GP models using saved hyperparameters as initial values.

        This is much faster than a full re-training because the optimizer
        starts at the already-converged solution and finishes in very few
        iterations.

        Parameters
        ----------
        hyperparams : list of dict
            One dict per principal component.  Each dict contains
            'model_type' ('hetGP' or 'homGP'), 'theta', 'g', and for
            hetGP models also 'Delta' and 'k_theta_g'.
        maxit : int
            Maximum optimizer iterations for the warm-start (default 0).
        """
        event_mask = np.ones(self.nev, dtype=bool)
        design_points_masked = self.design_points[event_mask, :]
        data_pca_masked = self.model_data_pca[event_mask, :]

        self.emu_list = []
        for j in range(self.npc):
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
            self.emu_list.append(model)

        logging.info(
            "Rebuilt {} GP models via warm-start (maxit={}).".format(
                self.npc, maxit
            )
        )

    def trainEmulator(self, event_mask):
        logging.info('Performing emulator training ...')
        # Subselect training data
        event_mask = np.asarray(event_mask, dtype=bool)
        design_points_masked = self.design_points[event_mask, :]
        # fit the output PCA only to the training points
        self._fit_output_pca(self.model_data[event_mask, :])
        data_pca_masked = self.model_data_pca

        nev_train = design_points_masked.shape[0]
        logging.info(
            'Train hetGP emulators for {} training points and {} PCs ...'.format(
                nev_train, self.npc
            )
        )

        # Train one hetGP model per principal component of the outputs.
        self.emu_list = []
        for j in range(self.npc):
            Z_train = data_pca_masked[:, j]
            model = hetGP()
            model.mleHetGP(
                X=design_points_masked,
                Z=Z_train,
                settings={'return_matrices': True},
                covtype="Matern3_2",
                maxit=100,
            )
            self.emu_list.append(model)

    def predict(self,X,return_cov=True):
        """
        Predict model output. Here X is the parameter vector at the prediction
        point.
        """
        X = np.atleast_2d(X)
        n_theta = X.shape[0]

        # Predict principal components at given parameter points
        pc_means = np.zeros((self.npc, n_theta))
        pc_vars = np.zeros((self.npc, n_theta))

        for j, model in enumerate(self.emu_list):
            pred = model.predict(x=X)
            mean_j = np.asarray(pred["mean"]).reshape(-1)
            var_j = np.asarray(pred["sd2"]).reshape(-1)
            if "nugs" in pred:
                var_j = var_j + np.asarray(pred["nugs"]).reshape(-1)
            pc_means[j, :] = mean_j
            pc_vars[j, :] = var_j

        # Reconstruct observables from PCs
        Z_pred = pc_means.T  # (n_theta, npc)
        standardized_pred = self.outputPCA.inverse_transform(Z_pred)
        Y_pred = self.outputScaler.inverse_transform(standardized_pred)  # (n_theta, nobs)

        # Build covariance matrices in observable space
        components = self.outputPCA.components_  # (npc, nobs)
        W = components.T  # (nobs, npc)
        scales = self.outputScaler.scale_  # (nobs,)
        D = np.diag(scales)

        covs = np.zeros((n_theta, self.nobs, self.nobs))
        for k in range(n_theta):
            var_z = np.maximum(pc_vars[:, k], 0.0)
            Sigma_S = W @ np.diag(var_z) @ W.T
            Sigma_Y = D @ Sigma_S @ D
            covs[k] = Sigma_Y + self._cov_trunc

        fpredmean = Y_pred
        fpredcov = covs

        if self.exp_and_cov_diagonal_:
            # If the emulator is trained on the log of the data, we return the
            # predictions in the original scale with diagonal covariance matrix.
            fpredmean = np.exp(fpredmean)

            fcov = np.zeros((n_theta, self.nobs, self.nobs))
            for i in range(n_theta):
                diagonal_cov = np.zeros((self.nobs, self.nobs))
                fstd = np.sqrt(np.diag(fpredcov[i]))
                np.fill_diagonal(diagonal_cov, (fstd * fpredmean[i])**2)
                fcov[i] = diagonal_cov
            fpredcov = fcov

        if return_cov:
            return (fpredmean, fpredcov)
        else:
            return fpredmean
        
