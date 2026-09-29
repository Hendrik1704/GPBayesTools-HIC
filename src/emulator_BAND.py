"""
Training for Gaussian process emulators.

Uses the `Gaussian process regression
<https://surmise.readthedocs.io/en/latest/index.html>`_ implemented by the BAND 
collaboration.
"""

import logging
import numpy as np
import pickle
import surmise
from surmise.emulation import emulator

from . import cachedir, parse_model_parameter_file

class EmulatorBAND:
    """
    Multidimensional Gaussian Process emulator wrapper for the GP emulators of 
    the BAND collaboration.
    """

    def __init__(self, training_set_path=".", parameter_file="ABCD.txt", 
                 method='PCGP',logTrafo=False,
                 max_rel_uncertainty_data=0.1, exp_and_cov_diagonal=False,
                 seed=None):
        self.method_ = method
        # surmise (>=1.0.0) requires a global RNG to be set before training
        self.rng_ = np.random.default_rng(seed)
        self.logTrafo_ = logTrafo 
        self.max_rel_uncertainty_data_ = max_rel_uncertainty_data
        self._load_training_data_pickle(training_set_path)
        self.exp_and_cov_diagonal_ = exp_and_cov_diagonal
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


    def predict_test_emu_errors(self,X,theta):
        """
        Predict model output.
        """
        gp = self.emu.predict(x=X,theta=theta)

        if self.exp_and_cov_diagonal_:
            # If the emulator is trained on the log of the data, we return the
            # predictions in the original scale with diagonal covariance matrix.
            fpredmean = np.exp(gp.mean())
        else:
            fpredmean = gp.mean()

        fpredcov = self._full_covariance(gp)

        if self.exp_and_cov_diagonal_:
            fcov = np.zeros_like(fpredcov)
            # Extract the diagonal of the covariance matrix for each prediction
            for i in range(fpredcov.shape[0]):
                diagonal_cov = np.zeros((self.nobs, self.nobs))
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
