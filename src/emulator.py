"""
Trains Gaussian process emulators.

When run as a script, allows retraining emulators, specifying the number of
principal components, and other options (however it is not necessary to do this
explicitly --- the emulators will be trained automatically when needed).  Run
``python -m src.emulator --help`` for usage information.

Uses the `scikit-learn <http://scikit-learn.org>`_ implementations of
`principal component analysis (PCA)
<http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
and `Gaussian process regression
<http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_.
"""

import logging
import numpy as np
import pickle
from os import path
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.model_selection import learning_curve
#from gp_extras.kernels import HeteroscedasticKernel
from sklearn.cluster import KMeans

from . import cachedir, parse_model_parameter_file


class Emulator:
    """
    Multidimensional Gaussian process emulator using principal component
    analysis.

    The model training data are standardized (subtract mean and scale to unit
    variance), then transformed through PCA.  The first `npc` principal
    components (PCs) are emulated by independent Gaussian processes (GPs).  The
    remaining components are neglected, which is equivalent to assuming they
    are standard zero-mean unit-variance GPs.

    This class has become a bit messy but it still does the job.  It would
    probably be better to refactor some of the data transformations /
    preprocessing into modular classes, to be used with an sklearn pipeline.
    The classes would also need to handle transforming uncertainties, which
    could be tricky.
    """
    def __init__(self, training_set_path=".", parameter_file="ABCD.txt",
                 npc=10, nrestarts=0, logTrafo=False, parameterTrafoPCA=False,
                 max_rel_uncertainty_data=0.1):
        self.logTrafo_ = logTrafo
        self.parameterTrafoPCA_ = parameterTrafoPCA
        self.max_rel_uncertainty_data_ = max_rel_uncertainty_data
        self._load_training_data_pickle(training_set_path)

        self.pardict = parse_model_parameter_file(parameter_file)
        self.design_min = []
        self.design_max = []
        for par, val in self.pardict.items():
            self.design_min.append(val[1])
            self.design_max.append(val[2])
        self.design_min = np.array(self.design_min)
        self.design_max = np.array(self.design_max)

        self.npc = npc
        self.nrestarts = nrestarts
        self.nev, self.nobs = self.model_data.shape

        self.scaler = StandardScaler(copy=False)
        self.pca = PCA(copy=False, whiten=True, svd_solver='full')

        if self.parameterTrafoPCA_:
            self.targetVariance = 0.99
            # the order of the PCA trafos is important here, since the second and
            # third trafo will update the PCA_new_design_points
            logging.info("Prepare bulk viscosity parameter PCA ...")
            self.paramTrafoScaler_bulk = StandardScaler()
            self.paramTrafoPCA_bulk = PCA(n_components=self.targetVariance)# 0.99 is the minimum of explained variance
            self.indices_zeta_s_parameters = [15,16,17,18] # zeta_max,T_zeta0,sigma_plus,sigma_minus
            self.perform_bulk_viscosity_PCA()

            logging.info("Prepare shear viscosity parameter PCA ...")
            self.paramTrafoScaler_shear = StandardScaler()
            self.paramTrafoPCA_shear = PCA(n_components=self.targetVariance)# 0.99 is the minimum of explained variance
            self.indices_eta_s_parameters = [12,13,14]
            self.perform_shear_viscosity_PCA()

            logging.info("Prepare yloss parameter PCA ...")
            self.paramTrafoScaler_yloss = StandardScaler()
            self.paramTrafoPCA_yloss = PCA(n_components=self.targetVariance)# 0.99 is the minimum of explained variance
            self.indices_yloss_parameters = [2,3,4]
            self.perform_yloss_PCA()


    def parametrization_zeta_over_s_vs_T(self,zeta_max,T_zeta0,
                                         sigma_plus,sigma_minus,T,mu_B):
        T_zeta_muB = T_zeta0 - 0.15*mu_B**2.
        if T < T_zeta0:
            return zeta_max * np.exp(-(T-T_zeta_muB)**2./(2.*sigma_minus**2.))
        else:
            return zeta_max * np.exp(-(T-T_zeta_muB)**2./(2.*sigma_plus**2.))


    def parametrization_eta_over_s_vs_mu_B(self,eta_0,eta_2,eta_4,mu_B):
        if 0. < mu_B and mu_B <= 0.2:
            return eta_0 + (eta_2 - eta_0) * (mu_B / 0.2)
        elif 0.2 < mu_B and mu_B < 0.4:
            return eta_2 + (eta_4 - eta_2) * ((mu_B - 0.2) / 0.2)
        else:
            return eta_4


    def parametrization_y_loss_vs_y_init(self,yloss_2,yloss_4,yloss_6,y_init):
        if 0. < y_init and y_init <= 2.:
            return yloss_2 * (y_init / 2.)
        elif 2. < y_init and y_init < 4.:
            return yloss_2 + (yloss_4 - yloss_2) * ((y_init - 2.) / 2.)
        else:
            return yloss_4 + (yloss_6 - yloss_4) * ((y_init - 4.) / 2.)


    def perform_bulk_viscosity_PCA(self):
        # get the corresponding parameters for the training points
        bulk_viscosity_parameters = self.design_points[:,self.indices_zeta_s_parameters]
        T_range = np.linspace(0.0, 0.5, 100)
        data_functions = []
        # Iterate over each parameter set
        for p in range(self.nev):
            # Evaluate the function for each temperature value in T_range
            parameter_function = [self.parametrization_zeta_over_s_vs_T(
                bulk_viscosity_parameters[p, 0], bulk_viscosity_parameters[p, 1],
                bulk_viscosity_parameters[p, 2], bulk_viscosity_parameters[p, 3],
                T, 0.0) for T in T_range]
            data_functions.append(parameter_function)

        data_functions = np.array(data_functions)
        scaled_data_functions = self.paramTrafoScaler_bulk.fit_transform(data_functions)
        self.paramTrafoPCA_bulk.fit(scaled_data_functions)

        # Get the number of components needed to achieve the target variance
        n_components = self.paramTrafoPCA_bulk.n_components_
        logging.info(f"Bulk viscosity parameter PCA uses {n_components} PCs to explain {self.targetVariance*100}% of the variance ...")

        # Get the principal components
        # principal_components will have shape (1000, n_components)
        principal_components = self.paramTrafoPCA_bulk.transform(scaled_data_functions)

        # Modify the design points
        self.PCA_new_design_points = np.delete(self.design_points, self.indices_zeta_s_parameters, axis=1)
        self.PCA_new_design_points = np.concatenate((self.PCA_new_design_points, principal_components), axis=1)

        # delete the parameters from the pardict and add the new ones
        self.design_min = np.delete(self.design_min, self.indices_zeta_s_parameters)
        self.design_max = np.delete(self.design_max, self.indices_zeta_s_parameters)
        min_values_PC = np.min(principal_components, axis=0)
        max_values_PC = np.max(principal_components, axis=0)
        self.design_min = np.concatenate((self.design_min,min_values_PC))
        self.design_max = np.concatenate((self.design_max,max_values_PC))


    def perform_shear_viscosity_PCA(self):
        # get the corresponding parameters for the training points
        shear_viscosity_parameters = self.design_points[:,self.indices_eta_s_parameters]
        mu_B_range = np.linspace(0.0, 0.6, 100)
        data_functions = []
        # Iterate over each parameter set
        for p in range(self.nev):
            # Evaluate the function for each mu_B value in mu_B_range
            parameter_function = [self.parametrization_eta_over_s_vs_mu_B(
                shear_viscosity_parameters[p, 0], shear_viscosity_parameters[p, 1],
                shear_viscosity_parameters[p, 2], mu_B) for mu_B in mu_B_range]
            data_functions.append(parameter_function)

        data_functions = np.array(data_functions)
        scaled_data_functions = self.paramTrafoScaler_shear.fit_transform(data_functions)
        self.paramTrafoPCA_shear.fit(scaled_data_functions)

        # Get the number of components needed to achieve the target variance
        n_components = self.paramTrafoPCA_shear.n_components_
        logging.info(f"Shear viscosity parameter PCA uses {n_components} PCs to explain {self.targetVariance*100}% of the variance ...")

        # Get the principal components
        # principal_components will have shape (1000, n_components)
        principal_components = self.paramTrafoPCA_shear.transform(scaled_data_functions)

        # Modify the design points
        self.PCA_new_design_points = np.delete(self.PCA_new_design_points, self.indices_eta_s_parameters, axis=1)
        self.PCA_new_design_points = np.concatenate((self.PCA_new_design_points, principal_components), axis=1)

        # delete the parameters from the pardict and add the new ones
        self.design_min = np.delete(self.design_min, self.indices_eta_s_parameters)
        self.design_max = np.delete(self.design_max, self.indices_eta_s_parameters)
        min_values_PC = np.min(principal_components, axis=0)
        max_values_PC = np.max(principal_components, axis=0)
        self.design_min = np.concatenate((self.design_min,min_values_PC))
        self.design_max = np.concatenate((self.design_max,max_values_PC))


    def perform_yloss_PCA(self):
        # get the corresponding parameters for the training points
        yloss_parameters = self.design_points[:,self.indices_yloss_parameters]
        yinit_range = np.linspace(0.0, 6.2, 100)
        data_functions = []
        # Iterate over each parameter set
        for p in range(self.nev):
            # Evaluate the function for each value in yinit_range
            parameter_function = [self.parametrization_y_loss_vs_y_init(
                yloss_parameters[p, 0], yloss_parameters[p, 1],
                yloss_parameters[p, 2], yinit) for yinit in yinit_range]
            data_functions.append(parameter_function)

        data_functions = np.array(data_functions)
        scaled_data_functions = self.paramTrafoScaler_yloss.fit_transform(data_functions)
        self.paramTrafoPCA_yloss.fit(scaled_data_functions)

        # Get the number of components needed to achieve the target variance
        n_components = self.paramTrafoPCA_yloss.n_components_
        logging.info(f"yloss parameter PCA uses {n_components} PCs to explain {self.targetVariance*100}% of the variance ...")

        # Get the principal components
        # principal_components will have shape (1000, n_components)
        principal_components = self.paramTrafoPCA_yloss.transform(scaled_data_functions)

        # Modify the design points
        self.PCA_new_design_points = np.delete(self.PCA_new_design_points, self.indices_yloss_parameters, axis=1)
        self.PCA_new_design_points = np.concatenate((self.PCA_new_design_points, principal_components), axis=1)

        # delete the parameters from the pardict and add the new ones
        self.design_min = np.delete(self.design_min, self.indices_yloss_parameters)
        self.design_max = np.delete(self.design_max, self.indices_yloss_parameters)
        min_values_PC = np.min(principal_components, axis=0)
        max_values_PC = np.max(principal_components, axis=0)
        self.design_min = np.concatenate((self.design_min,min_values_PC))
        self.design_max = np.concatenate((self.design_max,max_values_PC))


    def outputPCAvsParam(self):
        logging.info('Performing PCA ...')
        Z = self.pca.fit_transform(
                self.scaler.fit_transform(self.model_data)
        )[:, :self.npc]
        return(self.design_points, Z.T)


    def trainEmulatorAutoMask(self):
        trainEventMask = [True]*self.nev
        self.trainEmulator(trainEventMask)


    def trainEmulator(self, eventMask):
        # Standardize observables and transform through PCA.  Use the first
        # `npc` components but save the full PC transformation for later.
        logging.info('Performing PCA ...')
        Z = self.pca.fit_transform(
                self.scaler.fit_transform(self.model_data[eventMask, :])
        )[:, :self.npc]

        logging.info('{} PCs explain {:.5f} of variance'.format(
            self.npc, self.pca.explained_variance_ratio_[:self.npc].sum()
        ))

        nev, nobs = self.model_data[eventMask, :].shape
        logging.info(
            'Train GP emulators with {} training points ...'.format(nev))

        design_points = self.design_points[eventMask, :]
        if self.parameterTrafoPCA_:
            design_points = self.PCA_new_design_points[eventMask, :]

        # Define kernel (covariance function):
        # Gaussian correlation (RBF) plus a noise term.
        ptp = self.design_max - self.design_min

        rbf_kern = 1. * kernels.RBF(
                      length_scale=ptp,
                      length_scale_bounds=np.outer(ptp, (1e-1, 1e2)),
                   )
        const_kern = kernels.ConstantKernel()

        #homoscedastic noise kernel
        hom_white_kern = kernels.WhiteKernel(
                                 noise_level=.05,
                                 noise_level_bounds=(1e-2, 1e2)
                                 )

        #heteroscedastic noise kernel
        use_hom_sced_noise = True

        if use_hom_sced_noise:
            kernel = (rbf_kern + hom_white_kern)
        else:
            n_clusters = 10
            prototypes = KMeans(n_clusters=n_clusters).fit(
                        design_points).cluster_centers_
            het_noise_kern = HeteroscedasticKernel.construct(
                prototypes, 1., (1e-1, 1e1), gamma=1e-5, gamma_bounds="fixed")
            kernel = (rbf_kern + het_noise_kern)

        # Fit a GP (optimize the kernel hyperparameters) to each PC.
        self.gps = [
            GPR(kernel=kernel, alpha=0.1,
                n_restarts_optimizer=self.nrestarts,
                copy_X_train=False
            ).fit(design_points, z)
            for z in Z.T
        ]
        gpScores = []
        for i, gp in enumerate(self.gps):
            gpScores.append(gp.score(design_points, Z.T[i]))

        for n, (evr, gp) in enumerate(zip(
                self.pca.explained_variance_ratio_, self.gps
        )):
            logging.info(
                'GP {}: {:.5f} of variance, LML = {:.5g}, Score = {:.2f}, kernel: {}'
                .format(n, evr, gp.log_marginal_likelihood_value_, gpScores[n],
                        gp.kernel_)
            )

        # Construct the full linear transformation matrix, which is just the PC
        # matrix with the first axis multiplied by the explained standard
        # deviation of each PC and the second axis multiplied by the
        # standardization scale factor of each observable.
        self._trans_matrix = (
            self.pca.components_
            * np.sqrt(self.pca.explained_variance_[:, np.newaxis])
            * self.scaler.scale_
        )

        # Pre-calculate some arrays for inverse transforming the predictive
        # variance (from PC space to physical space).

        # Assuming the PCs are uncorrelated, the transformation is
        #
        #   cov_ij = sum_k A_ki var_k A_kj
        #
        # where A is the trans matrix and var_k is the variance of the kth PC.
        # https://en.wikipedia.org/wiki/Propagation_of_uncertainty

        # Compute the partial transformation for the first `npc` components
        # that are actually emulated.
        A = self._trans_matrix[:self.npc]
        self._var_trans = np.einsum(
            'ki,kj->kij', A, A, optimize=False).reshape(self.npc, self.nobs**2)

        # Compute the covariance matrix for the remaining neglected PCs
        # (truncation error).  These components always have variance == 1.
        B = self._trans_matrix[self.npc:]
        self._cov_trunc = np.dot(B.T, B)

        # Add small term to diagonal for numerical stability.
        self._cov_trunc.flat[::self.nobs + 1] += 1e-4 * self.scaler.var_


    def _inverse_transform(self, Z):
        """
        Inverse transform principal components to observables.
        # Z shape (..., npc)
        # Y shape (..., nobs)

        """
        Y = np.dot(Z, self._trans_matrix[:Z.shape[-1]])
        Y += self.scaler.mean_
        return Y


    def _load_training_data_pickle(self, dataFile):
        """This function read in training data set at every sample point"""
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
        self.design_points_org_ = np.copy(self.design_points)
        self.model_data = np.array(self.model_data)
        self.model_data_err = np.nan_to_num(
                np.abs(np.array(self.model_data_err)))
        logging.info("All training data are loaded.")
        logging.info("Training dataset size: {}, discarded points: {}".format(
            len(self.model_data),discarded_points))


    def getAvgTrainingDataRelError(self,):
        relErr = np.mean(np.nan_to_num(self.model_data_err/self.model_data),
                         axis=0)
        return(relErr)


    def print_learning_curve(self):
        Z = self.pca.fit_transform(
                self.scaler.fit_transform(self.model_data))[:, :self.npc]
        # Define kernel (covariance function):
        # Gaussian correlation (RBF) plus a noise term.
        ptp = self.design_max - self.design_min
        kernel = (
            1. * kernels.RBF(
                length_scale=ptp,
                length_scale_bounds=np.outer(ptp, (.01, 100))
            ) +
            kernels.WhiteKernel(
                noise_level=.01**2,
                noise_level_bounds=(.001**2, 1)
            )
        )

        design_points = self.design_points
        if self.parameterTrafoPCA_:
            design_points = self.PCA_new_design_points
        
        trainStatus = []
        for i, z in enumerate(Z.T):
            train_size_abs, train_scores, test_scores = learning_curve(
                GPR(kernel=kernel, alpha=0.,
                    copy_X_train=False),
                design_points, z, train_sizes=[0.2, 0.4, 0.6, 0.8, 0.9]
            )
            output = np.array([train_size_abs, np.mean(train_scores, axis=1),
                               np.mean(test_scores, axis=1)])
            trainStatus.append(output.transpose())
            logging.info("GP {}:".format(i))
            for train_size, cv_train_scores, cv_test_scores in zip(
                    train_size_abs, train_scores, test_scores
            ):
                logging.info(f"{train_size} samples were used to train the model")
                logging.info(f"The average train accuracy is {cv_train_scores.mean():.2f}")
                logging.info(f"The average test accuracy is {cv_test_scores.mean():.2f}")
        return(trainStatus)


    def predict(self, X, return_cov=True, extra_std=0):
        """
        Predict model output at `X`.

        X must be a 2D array-like with shape ``(nsamples, ndim)``. It is passed
        directly to sklearn :meth:`GaussianProcessRegressor.predict`.

        If `return_cov` is true, return a tuple ``(mean, cov)``, otherwise only
        return the mean.

        The mean is returned as a nested dict of observable arrays, each with
        shape ``(nsamples, n_cent_bins)``.

        The covariance is returned as a proxy object which extracts observable
        sub-blocks using a dict-like interface:

        The shape of the extracted covariance blocks are
        ``(nsamples, n_cent_bins_1, n_cent_bins_2)``.

        NB: the covariance is only computed between observables 
            not between sample points.

        `extra_std` is additional uncertainty which is added to each GP's
        predictive uncertainty, e.g. to account for model systematic error.
        It may either be a scalar or an array-like of length nsamples.

        """
        if self.parameterTrafoPCA_:
            if np.ndim(X) == 1:
                bulk_viscosity_parameters = X[self.indices_zeta_s_parameters]
            else:
                bulk_viscosity_parameters = X[:,self.indices_zeta_s_parameters]
            T_range = np.linspace(0.0, 0.5, 100)
            data_functions = []
            for p in range(X.shape[0]):
                parameter_function = [self.parametrization_zeta_over_s_vs_T(
                    bulk_viscosity_parameters[p, 0], bulk_viscosity_parameters[p, 1],
                    bulk_viscosity_parameters[p, 2], bulk_viscosity_parameters[p, 3],
                    T, 0.0) for T in T_range]
                data_functions.append(parameter_function)
            data_functions = np.array(data_functions)

            scaled_data = self.paramTrafoScaler_bulk.transform(data_functions)
            projected_parameters = self.paramTrafoPCA_bulk.transform(scaled_data)

            new_theta = np.delete(X, self.indices_zeta_s_parameters, axis=1)
            new_theta = np.concatenate((new_theta, projected_parameters), axis=1)

            if np.ndim(X) == 1:
                shear_viscosity_parameters = X[self.indices_eta_s_parameters]
            else:
                shear_viscosity_parameters = X[:,self.indices_eta_s_parameters]
            mu_B_range = np.linspace(0.0, 0.6, 100)
            data_functions = []
            for p in range(X.shape[0]):
                parameter_function = [self.parametrization_eta_over_s_vs_mu_B(
                    shear_viscosity_parameters[p, 0], shear_viscosity_parameters[p, 1],
                    shear_viscosity_parameters[p, 2], mu_B) for mu_B in mu_B_range]
                data_functions.append(parameter_function)
            data_functions = np.array(data_functions)

            scaled_data = self.paramTrafoScaler_shear.transform(data_functions)
            projected_parameters = self.paramTrafoPCA_shear.transform(scaled_data)

            new_theta = np.delete(new_theta, self.indices_eta_s_parameters, axis=1)
            new_theta = np.concatenate((new_theta, projected_parameters), axis=1)

            if np.ndim(X) == 1:
                yloss_viscosity_parameters = X[self.indices_yloss_parameters]
            else:
                yloss_viscosity_parameters = X[:,self.indices_yloss_parameters]
            yinit_range = np.linspace(0.0, 6.2, 100)
            data_functions = []
            for p in range(X.shape[0]):
                parameter_function = [self.parametrization_y_loss_vs_y_init(
                    yloss_viscosity_parameters[p, 0], yloss_viscosity_parameters[p, 1],
                    yloss_viscosity_parameters[p, 2], yinit) for yinit in yinit_range]
                data_functions.append(parameter_function)
            data_functions = np.array(data_functions)

            scaled_data = self.paramTrafoScaler_yloss.transform(data_functions)
            projected_parameters = self.paramTrafoPCA_yloss.transform(scaled_data)

            new_theta = np.delete(new_theta, self.indices_yloss_parameters, axis=1)
            new_theta = np.concatenate((new_theta, projected_parameters), axis=1)

            gp_mean = [gp.predict(new_theta, return_cov=return_cov) for gp in self.gps]
        else:
            gp_mean = [gp.predict(X, return_cov=return_cov) for gp in self.gps]

        if return_cov:
            gp_mean, gp_cov = zip(*gp_mean)

        mean = self._inverse_transform(
            np.concatenate([m[:, np.newaxis] for m in gp_mean], axis=1)
        )

        if return_cov:
            # Build array of the GP predictive variances at each sample point.
            # shape: (nsamples, npc)
            gp_var = np.concatenate([
                c.diagonal()[:, np.newaxis] for c in gp_cov
            ], axis=1)

            # Add extra uncertainty to predictive variance.
            extra_std = np.array(extra_std, copy=False).reshape(-1, 1)
            gp_var += extra_std**2

            # Compute the covariance at each sample point using the
            # pre-calculated arrays (see constructor).
            cov = np.dot(gp_var, self._var_trans).reshape(
                X.shape[0], self.nobs, self.nobs
            )
            cov += self._cov_trunc

            return mean, cov
        else:
            return mean


    def sample_y(self, X, n_samples=1, random_state=None):
        """
        Sample model output at `X`.

        Returns a nested dict of observable arrays, each with shape
        ``(n_samples_X, n_samples, n_cent_bins)``.

        """
        # Sample the GP for each emulated PC.  The remaining components are
        # assumed to have a standard normal distribution.
        return self._inverse_transform(
            np.concatenate([
                gp.sample_y(
                    X, n_samples=n_samples, random_state=random_state
                )[:, :, np.newaxis]
                for gp in self.gps
            ] + [
                np.random.standard_normal(
                    (X.shape[0], n_samples, self.pca.n_components_ - self.npc)
                )
            ], axis=2)
        )


    def testEmulatorErrors(self, nTestPoints=1):
        """
        This function uses (nev - nTestPoints) points to train the emulator
        and use nTestPoints points to test the emulator in each iteration.
        It returns the emulator predictions, their errors,
        the actual values of observables and their errors as four arrays.
        """
        emulatorPreds = []
        emulatorPredsErr = []
        validationData = []
        validationDataErr = []

        logging.info("Validating GP emulator ...")
        eventIdxList = range(self.nev - nTestPoints, self.nev)
        trainEventMask = [True]*self.nev
        for event_i in eventIdxList:
            trainEventMask[event_i] = False
        self.trainEmulator(trainEventMask)
        validateEventMask = [not i for i in trainEventMask]

        pred, predCov = self.predict(
            self.design_points_org_[validateEventMask, :], return_cov=True)
        pred_var = np.sqrt(np.array([predCov[i].diagonal() for i in range(predCov.shape[0])]))
        
        if self.logTrafo_:
            emulatorPreds = np.exp(pred)
            emulatorPredsErr = pred_var*np.exp(pred)
        else:
            emulatorPreds = pred
            emulatorPredsErr = pred_var
        
        if self.logTrafo_:
            validationData = np.exp(self.model_data[validateEventMask, :])
            validationDataErr = self.model_data_err[validateEventMask, :]*np.exp(self.model_data[validateEventMask, :])
        else:
            validationData = self.model_data[validateEventMask, :]
            validationDataErr = self.model_data_err[validateEventMask, :]
        
        emulatorPreds = np.array(emulatorPreds).reshape(-1, self.nobs)
        emulatorPredsErr = np.array(emulatorPredsErr).reshape(-1, self.nobs)
        validationData = np.array(validationData).reshape(-1, self.nobs)
        validationDataErr = np.array(validationDataErr).reshape(-1, self.nobs)
        return (emulatorPreds, emulatorPredsErr,
               validationData, validationDataErr)
    
    def testEmulatorErrorsWithTrainingPoints(self, nTestPoints=1):
        """
        This function uses number_test_points points to train the 
        emulator and the same points to test the emulator in each 
        iteration. The resulting errors should be very small.
        It returns the emulator predictions, their errors,
        the actual values of observables and their errors as four arrays.
        """
        emulatorPreds = []
        emulatorPredsErr = []
        validationData = []
        validationDataErr = []

        logging.info("Validating GP emulator ...")
        eventIdxList = range(self.nev - nTestPoints, self.nev)
        trainEventMask = [True]*self.nev
        for event_i in eventIdxList:
            trainEventMask[event_i] = False
        self.trainEmulator(trainEventMask)
        validateEventMask = [i for i in trainEventMask] # here is the difference to the previous function

        pred, predCov = self.predict(
            self.design_points_org_[validateEventMask, :], return_cov=True)
        pred_var = np.sqrt(np.array([predCov[i].diagonal() for i in range(predCov.shape[0])]))
        
        if self.logTrafo_:
            emulatorPreds = np.exp(pred)
            emulatorPredsErr = pred_var*np.exp(pred)
        else:
            emulatorPreds = pred
            emulatorPredsErr = pred_var
        
        if self.logTrafo_:
            validationData = np.exp(self.model_data[validateEventMask, :])
            validationDataErr = self.model_data_err[validateEventMask, :]*np.exp(self.model_data[validateEventMask, :])
        else:
            validationData = self.model_data[validateEventMask, :]
            validationDataErr = self.model_data_err[validateEventMask, :]
        
        emulatorPreds = np.array(emulatorPreds).reshape(-1, self.nobs)
        emulatorPredsErr = np.array(emulatorPredsErr).reshape(-1, self.nobs)
        validationData = np.array(validationData).reshape(-1, self.nobs)
        validationDataErr = np.array(validationDataErr).reshape(-1, self.nobs)
        return (emulatorPreds, emulatorPredsErr,
               validationData, validationDataErr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='train emulators with the model dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-par', '--parameter_file', type=str, default='ABCD.txt',
        help='model parameter filename')
    parser.add_argument(
        '-t', '--training_set_path', type=str, default=".",
        help='path for the training data set from model'
    )
    parser.add_argument(
        '--npc', type=int, default=10,
        help='number of principal components'
    )
    parser.add_argument(
        '--nrestarts', type=int, default=0,
        help='number of optimizer restarts'
    )

    parser.add_argument(
        '--retrain', action='store_true', default=False,
        help='retrain even if emulator is cached'
    )

    args = parser.parse_args()
    kwargs = vars(args)

    emu = Emulator(**kwargs)
