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
from . import cachedir, parse_model_parameter_file

class EmulatorHETGPy:
    def __init__(self, training_set_path=".", parameter_file="ABCD.txt",
                 logTrafo=False, max_rel_uncertainty_data=0.1, 
                 exp_and_cov_diagonal=False):
        self.logTrafo_ = logTrafo
        self.max_rel_uncertainty_data_ = max_rel_uncertainty_data
        self.exp_and_cov_diagonal_ = exp_and_cov_diagonal
        self._load_training_data_pickle(training_set_path)

        if not self.logTrafo_ and self.exp_and_cov_diagonal_:
            raise ValueError("exp_and_cov_diagonal can only be set to True if logTrafo is True.")

        self.pardict = parse_model_parameter_file(parameter_file)
        self.design_min = []
        self.design_max = []
        for par, val in self.pardict.items():
            self.design_min.append(val[1])
            self.design_max.append(val[2])
        self.design_min = np.array(self.design_min)
        self.design_max = np.array(self.design_max)

        self.nev, self.nobs = self.model_data.shape
        self.nparameters = self.design_points.shape[1]

        # Perform PCA on the outputs to reduce dimensionality while
        # retaining 99% of the variance. The GP emulators are then
        # trained on the resulting principal components.
        self.targetVariance = 0.99
        logging.info("Performing output PCA for hetGP emulator ...")
        self.outputScaler = StandardScaler()
        standardized_outputs = self.outputScaler.fit_transform(self.model_data)
        self.outputPCA = PCA(n_components=self.targetVariance)
        self.model_data_pca = self.outputPCA.fit_transform(standardized_outputs)
        self.npc = self.outputPCA.n_components_
        logging.info(
            "Output PCA uses {} PCs to explain {:.1f}% of the variance ...".format(
                self.npc, self.targetVariance * 100.0
            )
        )

    def __getstate__(self):
        """Prepare a pickleable state.

        hetGP models may contain non-pickleable Fortran objects. To avoid
        serialization issues, we do not attempt to serialize them. Instead
        we extract the fitted hyperparameters (theta, Delta, k_theta_g,
        theta_g, g) so that the GP models can be rapidly rebuilt via a
        warm-start after unpickling.
        """
        state = self.__dict__.copy()
        # Extract hyperparameters before discarding the models
        if "emu_list" in state and state["emu_list"] is not None:
            hyperparams = []
            for model in state["emu_list"]:
                is_hom = isinstance(model, homGP) and not isinstance(model, hetGP)
                hp = {
                    "model_type": "homGP" if is_hom else "hetGP",
                    "theta": np.array(model.theta),
                    "g": float(model.g),
                }
                if not is_hom:
                    hp["Delta"] = np.array(model.Delta)
                    hp["k_theta_g"] = float(model.k_theta_g)
                    if model.theta_g is not None:
                        hp["theta_g"] = np.array(model.theta_g)
                hyperparams.append(hp)
            state["_gp_hyperparams"] = hyperparams
        else:
            state["_gp_hyperparams"] = None
        state["emu_list"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling.

        GP models are rebuilt using the saved hyperparameters as starting
        values (warm-start), so re-training converges almost immediately.
        """
        self.__dict__.update(state)
        hyperparams = self.__dict__.pop("_gp_hyperparams", None)
        if hyperparams is not None:
            logging.info("Rebuilding GP models from saved hyperparameters "
                         "(warm-start) ...")
            self._rebuild_from_hyperparams(hyperparams)
        else:
            logging.info("No saved hyperparameters found, performing full "
                         "re-training ...")
            self.trainEmulatorAutoMask()

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
            Maximum optimizer iterations for the warm-start (default 2).
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

    def _load_training_data_pickle(self, dataFile):
        """This function reads in training data sets at every sample point"""
        logging.info("loading training data from {} ...".format(dataFile))
        self.model_data = []
        self.model_data_err = []
        self.design_points = []
        with open(dataFile, "rb") as fp:
            dataDict = pickle.load(fp)

        # Sort keys in ascending order
        sorted_event_ids = sorted(dataDict.keys(), key=lambda x: int(x))

        discarded_points = 0
        for event_id in sorted_event_ids:
            temp_data = dataDict[event_id]["obs"].transpose()
            statErrMax = np.abs((temp_data[:, 1]/(temp_data[:, 0]+1e-16))).max()
            if statErrMax > self.max_rel_uncertainty_data_:
                logging.info("Discard Parameter {}, stat err = {:.2f}".format(
                                                    event_id, statErrMax))
                discarded_points += 1
                continue
            self.design_points.append(dataDict[event_id]["parameter"])
            if self.logTrafo_ == False:
                self.model_data.append(temp_data[:, 0])
                self.model_data_err.append(temp_data[:, 1])
            else:
                self.model_data.append(np.log(np.abs(temp_data[:, 0]) + 1e-30))
                self.model_data_err.append(
                    np.abs(temp_data[:, 1]/(temp_data[:, 0] + 1e-30))
                )
        self.design_points = np.array(self.design_points)
        self.model_data = np.array(self.model_data)
        self.model_data_err = np.nan_to_num(np.abs(np.array(self.model_data_err)))
        logging.info("All training data are loaded.")
        logging.info("Training dataset size: {}, discarded points: {}".format(
            len(self.model_data),discarded_points))
        
    def trainEmulatorAutoMask(self):
        trainEventMask = [True]*self.nev
        self.trainEmulator(trainEventMask)

    def trainEmulator(self, event_mask):
        logging.info('Performing emulator training ...')
        # Subselect training data
        event_mask = np.asarray(event_mask, dtype=bool)
        design_points_masked = self.design_points[event_mask, :]
        data_pca_masked = self.model_data_pca[event_mask, :]

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

    def predict_test_emu_errors(self,X,theta):
        """
        Predict model output.
        """
        # theta: (n_theta, nparameters)
        theta = np.atleast_2d(theta)
        n_theta = theta.shape[0]

        # Predict principal components at given parameter points
        pc_means = np.zeros((self.npc, n_theta))
        pc_vars = np.zeros((self.npc, n_theta))

        for j, model in enumerate(self.emu_list):
            pred = model.predict(x=theta)
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
            covs[k] = Sigma_Y

        # By convention of this wrapper, fpredmean has shape (nobs, n_theta)
        fpredmean = Y_pred.T
        fpredcov = covs

        if self.exp_and_cov_diagonal_:
            # If the emulator is trained on the log of the data, we return the
            # predictions in the original scale with diagonal covariance matrix.
            fpredmean = np.exp(fpredmean)

            fcov = np.zeros((n_theta, self.nobs, self.nobs))
            for i in range(n_theta):
                diagonal_cov = np.zeros((self.nobs, self.nobs))
                # covariances are in log-space; extract diagonal std
                fstd = np.sqrt(np.diag(fpredcov[i]))
                np.fill_diagonal(diagonal_cov, (fstd * fpredmean.T[i])**2)
                fcov[i] = diagonal_cov
            fpredcov = fcov

        return (fpredmean, fpredcov)
    
    def predict(self,X,return_cov=True, extra_std=0.0):
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
            covs[k] = Sigma_Y

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
        
    def testEmulatorErrors(self, number_test_points=1):
        """
        This function uses (nev - number_test_points) points to train the 
        emulator and use number_test_points points to test the emulator in each 
        iteration.
        It returns the emulator predictions, their errors,
        the actual values of observables and their errors as four arrays.
        """
        emulator_predictions = []
        emulator_predictions_err = []
        validation_data = []
        validation_data_err = []

        logging.info("Validation GP emulator ...")
        event_idx_list = range(self.nev - number_test_points, self.nev)
        train_event_mask = [True]*self.nev
        for event_i in event_idx_list:
            train_event_mask[event_i] = False
        self.trainEmulator(train_event_mask)
        validate_event_mask = [not i for i in train_event_mask]

        x = np.arange(self.nobs).reshape(-1, 1)
        pred_mean, pred_cov = self.predict_test_emu_errors(x,
            self.design_points[validate_event_mask, :])
        pred_mean = pred_mean.T
        pred_var = np.sqrt(np.array([pred_cov[i].diagonal() for i in range(pred_cov.shape[0])]))

        # if logTrafo is True, then the predictions are in log space
        # and we need to transform them back to the original space
        # if exp_and_cov_diag_ is True, then the predictions are not in log space
        if self.logTrafo_ and not self.exp_and_cov_diagonal_:
            emulator_predictions = np.exp(pred_mean)
            emulator_predictions_err = pred_var*np.exp(pred_mean)
        else:
            emulator_predictions = pred_mean
            emulator_predictions_err = pred_var

        if self.logTrafo_:
            validation_data = np.exp(self.model_data[validate_event_mask, :])
            validation_data_err = self.model_data_err[validate_event_mask, :]*np.exp(self.model_data[validate_event_mask, :])
        else:
            validation_data = self.model_data[validate_event_mask, :]
            validation_data_err = self.model_data_err[validate_event_mask, :]

        emulator_predictions = np.array(emulator_predictions).reshape(-1, self.nobs)
        emulator_predictions_err = np.array(emulator_predictions_err).reshape(-1, self.nobs)
        validation_data = np.array(validation_data).reshape(-1, self.nobs)
        validation_data_err = np.array(validation_data_err).reshape(-1, self.nobs)

        return (emulator_predictions, emulator_predictions_err, 
                    validation_data, validation_data_err)
    
    def testEmulatorErrorsWithTrainingPoints(self, number_test_points=1):
        """
        This function uses number_test_points points to train the 
        emulator and the same points to test the emulator in each 
        iteration. The resulting errors should be very small.
        It returns the emulator predictions, their errors,
        the actual values of observables and their errors as four arrays.
        """
        emulator_predictions = []
        emulator_predictions_err = []
        validation_data = []
        validation_data_err = []

        logging.info("Validation GP emulator ...")
        event_idx_list = range(self.nev - number_test_points, self.nev)
        train_event_mask = [True]*self.nev
        for event_i in event_idx_list:
            train_event_mask[event_i] = False
        self.trainEmulator(train_event_mask)
        validate_event_mask = [i for i in train_event_mask] # here is the difference to the previous function

        x = np.arange(self.nobs).reshape(-1, 1)
        pred_mean, pred_cov = self.predict_test_emu_errors(x,
            self.design_points[validate_event_mask, :])
        pred_mean = pred_mean.T
        pred_var = np.sqrt(np.array([pred_cov[i].diagonal() for i in range(pred_cov.shape[0])]))

        if self.logTrafo_ and not self.exp_and_cov_diagonal_:
            emulator_predictions = np.exp(pred_mean)
            emulator_predictions_err = pred_var*np.exp(pred_mean)
        else:
            emulator_predictions = pred_mean
            emulator_predictions_err = pred_var

        if self.logTrafo_:
            validation_data = np.exp(self.model_data[validate_event_mask, :])
            validation_data_err = self.model_data_err[validate_event_mask, :]*np.exp(self.model_data[validate_event_mask, :])
        else:
            validation_data = self.model_data[validate_event_mask, :]
            validation_data_err = self.model_data_err[validate_event_mask, :]

        emulator_predictions = np.array(emulator_predictions).reshape(-1, self.nobs)
        emulator_predictions_err = np.array(emulator_predictions_err).reshape(-1, self.nobs)
        validation_data = np.array(validation_data).reshape(-1, self.nobs)
        validation_data_err = np.array(validation_data_err).reshape(-1, self.nobs)

        return (emulator_predictions, emulator_predictions_err, 
                    validation_data, validation_data_err)