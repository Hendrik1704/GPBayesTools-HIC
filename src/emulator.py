"""
Training for Gaussian process emulators.

Uses the `scikit-learn <http://scikit-learn.org>`_ implementations of
`principal component analysis (PCA)
<http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
and `Gaussian process regression
<http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_.
"""

import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.model_selection import learning_curve

from .emulator_base import EmulatorBase, check_npc, number_of_pcs


class Emulator(EmulatorBase):
    """
    Multidimensional Gaussian process emulator using principal component
    analysis. There is the option to switch off the PCA transformation
    and use the raw data for the Gaussian process emulation.

    The model training data are standardized (subtract mean and scale to unit
    variance), then transformed through PCA.  The first `npc` principal
    components (PCs) are emulated by independent Gaussian processes (GPs),
    where `npc` is the number of PCs (int) or the fraction of the explained
    variance (float in (0, 1)).  The
    remaining components are neglected, which is equivalent to assuming they
    are standard zero-mean unit-variance GPs.

    This class has become a bit messy but it still does the job.  It would
    probably be better to refactor some of the data transformations /
    preprocessing into modular classes, to be used with an sklearn pipeline.
    The classes would also need to handle transforming uncertainties, which
    could be tricky.

    With `logTrafo` set to True, the emulator is trained on the log of the
    observables and predict() returns the mean and covariance in log space.
    Experimental data used with the emulator must then be log-transformed as
    well. The parameter `exp_and_cov_diagonal` can be set to True to
    exponentiate the mean and set the off-diagonal elements of the covariance
    matrix to zero. For log trained emulators, this will return predictions
    in the original scale of the observables, but with diagonal covariance
    matrices.

    The parameter `perform_no_PCA` can be set to True to switch off the PCA
    transformation and use the raw data for the Gaussian process emulation.
    """
    def __init__(self, training_set_path=".", parameter_file="ABCD.txt",
                 npc=10, nrestarts=0, logTrafo=False,
                 max_rel_uncertainty_data=None, exp_and_cov_diagonal=False,
                 perform_no_PCA=False):
        super().__init__(training_set_path, parameter_file, logTrafo,
                         max_rel_uncertainty_data, exp_and_cov_diagonal)
        self.perform_no_PCA_ = perform_no_PCA

        check_npc(npc)
        self.npc_requested_ = npc
        self.npc = npc
        self.nrestarts = nrestarts

        self.scaler = StandardScaler()
        self.pca = PCA(whiten=True, svd_solver='full')


    def _pca_of_all_data(self):
        """
        First npc PCs of all training data. Separate scaler and PCA objects
        are used, so that the trained emulator is not modified.
        """
        scaler = StandardScaler()
        pca = PCA(whiten=True, svd_solver='full')
        Z = pca.fit_transform(scaler.fit_transform(self.model_data))
        npc = number_of_pcs(getattr(self, 'npc_requested_', self.npc),
                            pca.explained_variance_ratio_)
        return Z[:, :npc]


    def outputPCAvsParam(self):
        logging.info('Performing PCA ...')
        Z = self._pca_of_all_data()
        return(self.design_points, Z.T)


    def trainEmulator(self, eventMask, kernel_type="RBF"):
        data_to_use = self.model_data[eventMask, :]
        # Standardize the input data. New scaler and PCA objects are used,
        # so that the previously trained ones are not modified.
        self.scaler = StandardScaler()
        self.pca = PCA(whiten=True, svd_solver='full')
        standardized_data = self.scaler.fit_transform(data_to_use)

        if self.perform_no_PCA_:
            logging.info('Skipping PCA. Using raw standardized data for GP training ...')
            Z = standardized_data
            logging.info('Standardized data shape: {}'.format(Z.shape))
        else:
            logging.info('Standardizing data and performing PCA ...')
            # Transform data with PCA. Use the first
            # `npc` components but save the full PC transformation for later.
            Z = self.pca.fit_transform(standardized_data)
            # the PCA has at most min(n_training_points, nobs) components
            self.npc = number_of_pcs(getattr(self, 'npc_requested_', self.npc),
                                     self.pca.explained_variance_ratio_)
            Z = Z[:, :self.npc]

            logging.info('{} PCs explain {:.5f} of variance'.format(
                self.npc, self.pca.explained_variance_ratio_[:self.npc].sum()
            ))

        nev, nobs = self.model_data[eventMask, :].shape
        logging.info(
            'Train GP emulators with {} training points ...'.format(nev))

        design_points = self.design_points[eventMask, :]

        # Define kernel (covariance function):
        # Gaussian correlation (RBF) plus a noise term.
        ptp = self.design_max - self.design_min
        if kernel_type == "RBF":
            rbf_kern = 1. * kernels.RBF(
                    length_scale=ptp,
                    length_scale_bounds=np.outer(ptp, (1e-1, 1e2)),
                    )
        elif kernel_type == "Matern":
            rbf_kern = 1. * kernels.Matern(
                    length_scale=ptp,
                    length_scale_bounds=np.outer(ptp, (1e-3, 1e5)),
                    nu=1.5
                    )
        else:
            raise ValueError("Unknown kernel type: {}".format(kernel_type))

        #homoscedastic noise kernel
        hom_white_kern = kernels.WhiteKernel(
                                 noise_level=.05,
                                 noise_level_bounds=(1e-2, 1e2)
                                 )
        kernel = (rbf_kern + hom_white_kern)
        
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
        logging.info('GP scores: {}'.format(gpScores))

        if not self.perform_no_PCA_:
            for n, gp in enumerate(self.gps):
                evr = self.pca.explained_variance_ratio_[n]
                logging.info(
                    'GP {}: {:.5f} of variance, LML = {:.5g}, Score = {:.2f}, kernel: {}'
                    .format(n, evr, gp.log_marginal_likelihood_value_,
                            gpScores[n], gp.kernel_)
                )

        if not self.perform_no_PCA_:
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


    def getAvgTrainingDataRelError(self,):
        relErr = np.mean(np.nan_to_num(self.model_data_err/self.model_data),
                         axis=0)
        return(relErr)


    def print_learning_curve(self):
        Z = self._pca_of_all_data()
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


    def predict(self, X, return_cov=True):
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

        """
        gp_mean = [gp.predict(X, return_cov=return_cov) for gp in self.gps]

        if return_cov:
            gp_mean, gp_cov = zip(*gp_mean)

        if not self.perform_no_PCA_:
            mean = self._inverse_transform(
                np.concatenate([m[:, np.newaxis] for m in gp_mean], axis=1)
            )
        else:
            mean = self.scaler.inverse_transform(
                np.concatenate([m[:, np.newaxis] for m in gp_mean], axis=1)
            )

        if self.exp_and_cov_diagonal_:
            mean = np.exp(mean)

        if return_cov:
            # Build array of the GP predictive variances at each sample point.
            # shape: (nsamples, npc)
            gp_var = np.concatenate([
                c.diagonal()[:, np.newaxis] for c in gp_cov
            ], axis=1)

            if not self.perform_no_PCA_:
                # Compute the covariance at each sample point using the
                # pre-calculated arrays (see constructor).
                cov = np.dot(gp_var, self._var_trans).reshape(
                    X.shape[0], self.nobs, self.nobs
                )
                cov += self._cov_trunc
            else:
                # Create a covariance matrix for each sample point from gp_var,
                # transformed from standardized units back to observable units
                cov = np.zeros((X.shape[0], self.nobs, self.nobs))
                for i in range(X.shape[0]):
                    cov[i] = np.diag(gp_var[i] * self.scaler.var_)

            if self.exp_and_cov_diagonal_:
                # For each prediction set the off-diagonal elements of the
                # covariance matrix to zero
                for i in range(cov.shape[0]):
                    new_cov = np.zeros((self.nobs, self.nobs))
                    fstd = np.sqrt(np.diag(cov[i]))
                    np.fill_diagonal(new_cov, (fstd * mean[i])**2)
                    cov[i] = new_cov

            return mean, cov
        else:
            return mean


    def sample_y(self, X, n_samples=1, random_state=None):
        """
        Sample model output at `X`.

        Returns a nested dict of observable arrays, each with shape
        ``(n_samples_X, n_samples, n_cent_bins)``.

        """
        if not self.perform_no_PCA_:
            rng = np.random.default_rng(random_state)
            # Sample the GP for each emulated PC, with independent random
            # numbers for each GP.  The remaining components are assumed to
            # have a standard normal distribution.
            samples = self._inverse_transform(
                np.concatenate([
                    gp.sample_y(
                        X, n_samples=n_samples,
                        random_state=int(rng.integers(2**32 - 1))
                    )[:, :, np.newaxis]
                    for gp in self.gps
                ] + [
                    rng.standard_normal(
                        (X.shape[0], n_samples, self.pca.n_components_ - self.npc)
                    )
                ], axis=2)
            )
            if self.exp_and_cov_diagonal_:
                samples = np.exp(samples)
            return samples
        else:
            logging.warning("Sampling from raw data is not implemented.")
            return None
