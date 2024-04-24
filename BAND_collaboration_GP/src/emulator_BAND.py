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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import cachedir, parse_model_parameter_file

class EmulatorBAND:
    """
    Multidimensional Gaussian Process emulator wrapper for the GP emulators of 
    the BAND collaboration.
    """

    def __init__(self, training_set_path=".", parameter_file="ABCD.txt", 
                 method='PCGP',logTrafo=False,parameterTrafoPCA=False):
        self.method_ = method
        self.logTrafo_ = logTrafo 
        self.parameterTrafoPCA_ = parameterTrafoPCA
        self._load_training_data_pickle(training_set_path)

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

        if self.parameterTrafoPCA_:
            self.targetVariance = 0.99
            # the order of the PCA transformations is important here, since the second and
            # third transformation will update the PCA_new_design_points
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

            self.nparameters = self.PCA_new_design_points.shape[1]


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
            if statErrMax > 0.1:
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
        if self.parameterTrafoPCA_:
            design_points = self.PCA_new_design_points[event_mask, :]

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
                                method='PCGPwImpute',
                                args={'warnings': True})
        else:
            ValueError("Requested method not implemented!")


    def predict_test_emu_errors(self,X,theta):
        """
        Predict model output.
        """
        if self.parameterTrafoPCA_:
            if np.ndim(theta) == 1:
                bulk_viscosity_parameters = theta[self.indices_zeta_s_parameters]
            else:
                bulk_viscosity_parameters = theta[:,self.indices_zeta_s_parameters]
            T_range = np.linspace(0.0, 0.5, 100)
            data_functions = []
            for p in range(theta.shape[0]):
                parameter_function = [self.parametrization_zeta_over_s_vs_T(
                    bulk_viscosity_parameters[p, 0], bulk_viscosity_parameters[p, 1],
                    bulk_viscosity_parameters[p, 2], bulk_viscosity_parameters[p, 3],
                    T, 0.0) for T in T_range]
                data_functions.append(parameter_function)
            data_functions = np.array(data_functions)

            scaled_data = self.paramTrafoScaler_bulk.transform(data_functions)
            projected_parameters = self.paramTrafoPCA_bulk.transform(scaled_data)

            new_theta = np.delete(theta, self.indices_zeta_s_parameters, axis=1)
            new_theta = np.concatenate((new_theta, projected_parameters), axis=1)

            if np.ndim(theta) == 1:
                shear_viscosity_parameters = theta[self.indices_eta_s_parameters]
            else:
                shear_viscosity_parameters = theta[:,self.indices_eta_s_parameters]
            mu_B_range = np.linspace(0.0, 0.6, 100)
            data_functions = []
            for p in range(theta.shape[0]):
                parameter_function = [self.parametrization_eta_over_s_vs_mu_B(
                    shear_viscosity_parameters[p, 0], shear_viscosity_parameters[p, 1],
                    shear_viscosity_parameters[p, 2], mu_B) for mu_B in mu_B_range]
                data_functions.append(parameter_function)
            data_functions = np.array(data_functions)

            scaled_data = self.paramTrafoScaler_shear.transform(data_functions)
            projected_parameters = self.paramTrafoPCA_shear.transform(scaled_data)

            new_theta = np.delete(new_theta, self.indices_eta_s_parameters, axis=1)
            new_theta = np.concatenate((new_theta, projected_parameters), axis=1)

            if np.ndim(theta) == 1:
                yloss_viscosity_parameters = theta[self.indices_yloss_parameters]
            else:
                yloss_viscosity_parameters = theta[:,self.indices_yloss_parameters]
            yinit_range = np.linspace(0.0, 6.2, 100)
            data_functions = []
            for p in range(theta.shape[0]):
                parameter_function = [self.parametrization_y_loss_vs_y_init(
                    yloss_viscosity_parameters[p, 0], yloss_viscosity_parameters[p, 1],
                    yloss_viscosity_parameters[p, 2], yinit) for yinit in yinit_range]
                data_functions.append(parameter_function)
            data_functions = np.array(data_functions)

            scaled_data = self.paramTrafoScaler_yloss.transform(data_functions)
            projected_parameters = self.paramTrafoPCA_yloss.transform(scaled_data)

            new_theta = np.delete(new_theta, self.indices_yloss_parameters, axis=1)
            new_theta = np.concatenate((new_theta, projected_parameters), axis=1)

            gp = self.emu.predict(x=X,theta=new_theta)
        else:
            gp = self.emu.predict(x=X,theta=theta)

        fpredmean = gp.mean()
        fpredcov = gp.covx().transpose((1, 0, 2))

        return (fpredmean, fpredcov)


    def predict(self,X,return_cov=True, extra_std=0.0):
        """
        Predict model output. Here X is the parameter vector at the prediction
        point.
        """
        x = np.arange(self.nobs).reshape(-1, 1)

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

            gp = self.emu.predict(x=x,theta=new_theta)
        else:
            gp = self.emu.predict(x=x,theta=X)

        fpredmean = gp.mean()
        fpredcov = gp.covx().transpose((1, 0, 2))

        if return_cov:
            return (fpredmean.T, fpredcov)
        else:
            return fpredmean.T


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

        if self.logTrafo_:
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

        if self.logTrafo_:
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