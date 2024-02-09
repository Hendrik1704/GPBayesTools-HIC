"""
Trains Gaussian process emulators.

When run as a script, allows retraining emulators, specifying the number of
principal components, and other options (however it is not necessary to do this
explicitly --- the emulators will be trained automatically when needed).  Run
``python -m src.emulator --help`` for usage information.

Uses the `Gaussian process regression
<https://surmise.readthedocs.io/en/latest/index.html>`_ implemented by the BAND 
collaboration.
"""

import logging

import numpy as np
import pickle
from surmise.emulation import emulator

from . import cachedir, parse_model_parameter_file

class EmulatorBAND:
    """
    Multidimensional Gaussian Process emulator.
    """

    def __init__(self, training_set_path=".", parameter_file="ABCD.txt",
                 nrestarts=0, retrain=False, transformDesign=False,
                 logFlag=False):
        self._vec_zeta_s = np.vectorize(self._zeta_over_s)
        self.transformDesign_ = transformDesign
        self.logFlag_ = logFlag

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

        self.nrestarts = nrestarts
        self.nev, self.nobs = self.model_data.shape
        self.nparameters = self.design_points.shape[1]


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

    def _load_training_data_pickle(self, dataFile):
        """This function reads in training data sets at every sample point"""
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
            if self.logFlag_:
                self.model_data.append(np.log(np.abs(temp_data[:, 0]) + 1e-30))
                self.model_data_err.append(
                    np.abs(temp_data[:, 1]/(temp_data[:, 0] + 1e-30))
                )
            else:
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

    def trainEmulator(self, event_mask):
        logging.info('Performing emulator training ...')

        if isinstance(event_mask, int):
            event_idx_list = range(self.nev - event_mask)
            train_event_mask = [False]*self.nev
            for event_i in event_idx_list:
                train_event_mask[event_i] = True

            nev, nobs = self.model_data[train_event_mask, :].shape
            logging.info(
                'Train GP emulators with {} training points ...'.format(nev))

            x = np.arange(nobs).reshape(-1, 1)

            print("x = ",x.shape)
            print("theta = ",self.design_points[train_event_mask, :].shape)
            print("f = ",self.model_data[train_event_mask, :].T.shape)

            self.emu = emulator(x=self.x,theta=self.design_points[train_event_mask, :],
                                f=self.model_data[train_event_mask, :].T,
                                method='PCGP',
                                args={'warnings': True}
                                )
        else:
            nev, nobs = self.model_data[event_mask, :].shape
            logging.info(
                'Train GP emulators with {} training points ...'.format(nev))

            x = np.arange(nobs).reshape(-1, 1)

            print("x = ",x.shape)
            print("theta = ",self.design_points[event_mask, :].shape)
            print("f = ",self.model_data[event_mask, :].T.shape)

            self.emu = emulator(x=x,theta=self.design_points[event_mask, :],
                                f=self.model_data[event_mask, :].T,
                                method='PCGP',
                                args={'warnings': True}
                                )

        
    def predict(self,X,theta):
        """
        Predict model output at `X`.
        """
        gp = self.emu.predict(x=X,theta=theta)
        #print(gp._info)

        fpredmean = gp.mean()
        #fpredvar = gp.var()
        fpredcov = gp.covx().transpose((1, 0, 2))

        print(fpredmean.shape)
        print(fpredcov.shape)

        return (fpredmean, fpredcov)
    
    def testEmulatorErrors(self, number_test_points=1, number_iterations=1):
        """
        This function uses (nev - number_test_points) points to train the 
        emulator and use number_test_points points to test the emulator in each 
        iteration. In each iteration, the number_test_points data points are 
        randomly chosen.
        It returns the emulator predictions, their errors,
        the actual values of observables and their errors as four arrays.
        """
        emulator_predictions = []
        emulator_predictions_err = []
        validation_data = []
        validation_data_err = []

        for iter_i in range(number_iterations):
            logging.info(
                "Validation GP emulators iter = {} ...".format(iter_i))
            event_idx_list = range(self.nev - number_test_points, self.nev)
            train_event_mask = [True]*self.nev
            for event_i in event_idx_list:
                train_event_mask[event_i] = False
            self.trainEmulator(train_event_mask)
            validate_event_mask = [not i for i in train_event_mask]

            x = np.arange(self.nobs).reshape(-1, 1)
            pred_mean, pred_cov = self.predict(x,
                self.design_points[validate_event_mask, :])
            emulator_predictions.append(pred_mean)
            emulator_predictions_err.append(pred_cov)
            
            if self.logFlag_:
                validation_data.append(
                    np.exp(self.model_data[validate_event_mask, :])
                )
                validation_data_err.append(
                    self.model_data_err[validate_event_mask, :]
                    *np.exp(self.model_data[validate_event_mask, :])
                )
            else:
                validation_data.append(self.model_data[validate_event_mask, :])
                validation_data_err.append(
                    self.model_data_err[validate_event_mask, :]
                )
        emulator_predictions = np.array(emulator_predictions).reshape(-1, self.nobs)
        emulator_predictions_err = np.array(emulator_predictions_err).reshape(-1 , self.nobs, self.nobs)
        validation_data = np.array(validation_data).reshape(-1, self.nobs)
        validation_data_err = np.array(validation_data_err).reshape(-1, self.nobs)

        print(emulator_predictions.shape)
        print(emulator_predictions_err.shape)

        return (emulator_predictions, emulator_predictions_err, 
                    validation_data, validation_data_err)



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

    emu = EmulatorBAND(**kwargs)