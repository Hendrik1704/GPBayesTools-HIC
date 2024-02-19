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
#from sklearn.externals import joblib
import joblib

from gp_extras.kernels import HeteroscedasticKernel
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
                 npc=10, nrestarts=0, retrain=False, transformDesign=False):
        self._vec_zeta_s = np.vectorize(self._zeta_over_s)
        self.transformDesign_ = transformDesign
        

        #self._load_training_data(training_set_path)
        self._load_training_data_pickle(training_set_path)

        if self.transformDesign_:
            self.design_min = np.min(self.design_points, axis=0)
            self.design_max = np.max(self.design_points, axis=0)
        else:
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


    def outputPCAvsParam(self):
        logging.info('Perforing PCA ...')
        Z = self.pca.fit_transform(
                self.scaler.fit_transform(self.model_data)
        )[:, :self.npc]
        return(self.design_points, Z.T)


    def trainEmulator(self, eventMask):
        # Standardize observables and transform through PCA.  Use the first
        # `npc` components but save the full PC transformation for later.
        logging.info('Perforing PCA ...')
        Z = self.pca.fit_transform(
                self.scaler.fit_transform(self.model_data[eventMask, :])
        )[:, :self.npc]

        logging.info('{} PCs explain {:.5f} of variance'.format(
            self.npc, self.pca.explained_variance_ratio_[:self.npc].sum()
        ))

        nev, nobs = self.model_data[eventMask, :].shape
        logging.info(
            'Train GP emulators with {} training points ...'.format(nev))


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
                        self.design_points[eventMask, :]).cluster_centers_
            het_noise_kern = HeteroscedasticKernel.construct(
                prototypes, 1., (1e-1, 1e1), gamma=1e-5, gamma_bounds="fixed")
            kernel = (rbf_kern + het_noise_kern)

        # Fit a GP (optimize the kernel hyperparameters) to each PC.
        self.gps = [
            GPR(kernel=kernel, alpha=0.1,
                n_restarts_optimizer=self.nrestarts,
                copy_X_train=False
            ).fit(self.design_points[eventMask, :], z)
            for z in Z.T
        ]
        gpScores = []
        for i, gp in enumerate(self.gps):
            gpScores.append(gp.score(self.design_points[eventMask, :], Z.T[i]))

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

        for event_id in dataDict.keys():
            temp_data = dataDict[event_id]["obs"].transpose()
            statErrMax = np.abs((temp_data[:, 1]/(temp_data[:, 0]+1e-16))).max()
            if statErrMax > 0.1:
                logging.info("Discard Parameter {}, stat err = {:.2f}".format(
                                                    event_id, statErrMax))
                continue
            self.design_points.append(dataDict[event_id]["parameter"])
            self.model_data.append(temp_data[:, 0])
            self.model_data_err.append(temp_data[:, 1])
        logging.info("Training dataset size: {}".format(len(self.model_data)))
        self.design_points = np.array(self.design_points)
        self.design_points_org_ = np.copy(self.design_points)
        self.model_data = np.array(self.model_data)
        self.model_data_err = np.nan_to_num(
                np.abs(np.array(self.model_data_err)))
        if self.transformDesign_:
            self.design_points = self._transform_design(self.design_points)
        logging.info("All training data are loaded.")


    def getAvgTrainingDataRelError(self,):
        relErr = np.mean(np.nan_to_num(self.model_data_err/self.model_data),
                         axis=0)
        return(relErr)


    def _transform_design(self, X):
        """This function transform the parameters of bulk viscosity
        to another representation for better emulation performance.
        """
        # pop out the bulk viscous parameters
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                   12, 13, 14, 16, 17,
                   21]
        new_design_X = X[:, indices]

        #now append the values of eta/s and zeta/s at various temperatures
        num_T = 10
        Temperature_grid = np.linspace(0.12, 0.35, num_T)
        #num_muB = 3
        #muB_grid = np.linspace(0.0, 0.4, num_muB)
        zeta_vals = []
        for T_i in Temperature_grid:
            #for muB_i in muB_grid:
            zeta_vals.append(
                self._vec_zeta_s(T_i, 0.0, X[:, 15], X[:, 15], X[:, 15],
                                 X[:, 18], X[:, 19], X[:, 20])
            )

        zeta_vals = np.array(zeta_vals).T

        new_design_X = np.concatenate( (new_design_X, zeta_vals), axis=1)
        return new_design_X


    def _zeta_over_s(self, T, muB, bulkMax0, bulkMax1, bulkMax2,
                     bulkTpeak0, bulkWhigh, bulkWlow):
        if muB < 0.2:
            bulkMax = bulkMax0 + (bulkMax1 - bulkMax0)/0.2*muB
        elif muB < 0.4:
            bulkMax = bulkMax1 + (bulkMax2 - bulkMax1)/0.2*(muB - 0.2)
        else:
            bulkMax = bulkMax2
        bulkTpeak = bulkTpeak0 - 0.15*muB*muB
        zeta_s = self._zeta_over_s_base(T, bulkMax, bulkTpeak,
                                        bulkWhigh, bulkWlow)
        return zeta_s


    def _zeta_over_s_base(self, T, bulkMax, bulkTpeak, bulkWhigh, bulkWlow):
        Tdiff = T - bulkTpeak
        if Tdiff > 0:
            Tdiff /= bulkWhigh
        else:
            Tdiff /= bulkWlow
        zeta_s = bulkMax*np.exp(-Tdiff*Tdiff)
        return zeta_s


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
        trainStatus = []
        for i, z in enumerate(Z.T):
            train_size_abs, train_scores, test_scores = learning_curve(
                GPR(kernel=kernel, alpha=0.,
                    copy_X_train=False),
                self.design_points, z, train_sizes=[0.2, 0.4, 0.6, 0.8, 0.9]
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
        if self.transformDesign_:
            X = self._transform_design(X)

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
        if self.transformDesign_:
            X = self._transform_design(X)
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


    def testEmulatorErrors(self, nTestPoints=1, nIters=1):
        """
        This function uses (nev - nTestPoints) points to train the emulator
        and use nTestPoints points to test the emulator in each iteration.
        In each iteraction, the nTestPoints data points are randomly
        chosen.
        It returns the emulator predictions, their errors,
        the actual values of observabels and their errors as four arrays.
        """
        rng = np.random.default_rng()
        emulatorPreds = []
        emulatorPredsErr = []
        validationData = []
        validationDataErr = []
        for iter_i in range(nIters):
            logging.info(
                    "Validating GP emulators iter = {} ...".format(iter_i))
            #eventIdxList = rng.choice(self.nev, nTestPoints, replace=False)
            eventIdxList = range(self.nev - nTestPoints, self.nev)
            trainEventMask = [True]*self.nev
            for event_i in eventIdxList:
                trainEventMask[event_i] = False
            self.trainEmulator(trainEventMask)
            validateEventMask = [not i for i in trainEventMask]

            pred, predCov = self.predict(
                self.design_points_org_[validateEventMask, :], return_cov=True)
            emulatorPreds.append(pred)
            '''
            predErr = np.zeros([nTestPoints, self.nobs])
            for iobs in range(self.nobs):
                predErr[:, iobs] = np.sqrt(predCov[:, iobs, iobs])
                '''
            emulatorPredsErr.append(predCov)
            
            validationData.append(self.model_data[validateEventMask, :])
            print(validationData)
            validationDataErr.append(
                        self.model_data_err[validateEventMask, :]
                )
        emulatorPreds = np.array(emulatorPreds).reshape(-1, self.nobs)
        emulatorPredsErr = np.array(emulatorPredsErr).reshape(-1, self.nobs)
        validationData = np.array(validationData).reshape(-1, self.nobs)
        validationDataErr = np.array(validationDataErr).reshape(-1, self.nobs)
        return(emulatorPreds, emulatorPredsErr,
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
