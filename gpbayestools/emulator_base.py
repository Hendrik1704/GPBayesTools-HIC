"""
Common base class of the emulators.

It loads the training data and the model parameter file, applies the quality
filter and the log transformation of the training data, and implements the
validation functions. The emulators implement ``train_emulator(event_mask)``
and ``predict(X, return_cov=True)``.
"""

import logging
import numpy as np
import pickle

from . import keep_trained_state, parse_model_parameter_file


def check_npc(npc):
    """Check the number of principal components `npc`: an int >= 1 (number of
    PCs) or a float in (0, 1) (fraction of the explained variance)."""
    if isinstance(npc, (int, np.integer)) and not isinstance(npc, bool):
        if npc < 1:
            raise ValueError("npc must be >= 1, got {}".format(npc))
    elif isinstance(npc, (float, np.floating)):
        if not 0 < npc < 1:
            raise ValueError(
                "A float npc is the fraction of the explained variance and "
                "must be in (0, 1), got {}".format(npc)
            )
    else:
        raise TypeError("npc must be an int or a float, got {!r}".format(npc))


def number_of_pcs(npc, explained_variance_ratio):
    """Number of PCs for `npc` (see check_npc), given the explained variance
    ratios of all PCs. A float npc selects the smallest number of PCs that
    explain more than this fraction of the variance, as in sklearn's PCA."""
    n_available = len(explained_variance_ratio)
    if isinstance(npc, (float, np.floating)):
        n = np.searchsorted(np.cumsum(explained_variance_ratio), npc, side="right") + 1
        return int(min(n, n_available))
    if npc > n_available:
        logging.warning(
            "Only {} PCs available, using npc = {}".format(n_available, n_available)
        )
    return int(min(npc, n_available))


def truncation_signal(trunc_cov, noise_cov):
    """
    Remove the statistical noise of the training data from the truncation
    covariance `trunc_cov` of the discarded PCs. In each eigendirection of
    `trunc_cov`, the noise variance of `noise_cov` in that direction is
    subtracted, down to zero. The result is positive semi-definite and not
    larger than `trunc_cov`. Both matrices must be in the same (e.g.
    standardized) units.
    """
    vals, vecs = np.linalg.eigh(0.5 * (trunc_cov + trunc_cov.T))
    noise = np.einsum("ik,ij,jk->k", vecs, noise_cov, vecs)
    signal = np.clip(vals - noise, 0.0, None)
    return (vecs * signal) @ vecs.T


class EmulatorBase:
    """
    Base class of the emulators.

    Parameters
    ----------
    training_set_path : str
        Path to the pickle file with the training data, a dictionary
        {event_id: {'parameter': array (nparameters,),
        'obs': array (2, nobs) with the values and statistical errors}}.
    parameter_file : str
        Path to the model parameter file.
    log_trafo : bool
        If True, the emulator is trained on the log of the observables, which
        must be positive. predict() then returns the mean and covariance in log
        space, and experimental data used with the emulator must be
        log-transformed as well.
    max_rel_uncertainty_data : float or None
        Training points with a larger relative statistical error of any
        observable are discarded. None (default) disables this filter.
    exp_and_cov_diagonal : bool
        Only with log_trafo=True: predict() returns the predictions transformed
        back to the original scale of the observables.
    """

    def __init__(
        self,
        training_set_path=".",
        parameter_file="ABCD.txt",
        log_trafo=False,
        max_rel_uncertainty_data=None,
        exp_and_cov_diagonal=False,
    ):
        self.logTrafo_ = log_trafo
        self.max_rel_uncertainty_data_ = max_rel_uncertainty_data
        self.exp_and_cov_diagonal_ = exp_and_cov_diagonal
        if not self.logTrafo_ and self.exp_and_cov_diagonal_:
            raise ValueError(
                "exp_and_cov_diagonal can only be set to True if log_trafo is True."
            )

        self._load_training_data_pickle(training_set_path)

        self.pardict = parse_model_parameter_file(parameter_file)
        self.design_min = np.array([val[1] for val in self.pardict.values()])
        self.design_max = np.array([val[2] for val in self.pardict.values()])

        self.nev, self.nobs = self.model_data.shape
        self.nparameters = self.design_points.shape[1]
        if self.nparameters != len(self.pardict):
            raise ValueError(
                "The training data have {} parameters, but the parameter file "
                "{} has {}".format(self.nparameters, parameter_file, len(self.pardict))
            )

    # -------------------------
    # Training data
    # -------------------------
    @staticmethod
    def _max_rel_error(temp_data):
        """Largest relative statistical error of a training point. Observables
        that are exactly zero have no relative error and are ignored."""
        nonzero = temp_data[:, 0] != 0
        return np.max(
            np.abs(temp_data[nonzero, 1] / temp_data[nonzero, 0]), initial=0.0
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
            if not np.all(np.isfinite(temp_data[:, 0])):
                logging.info(
                    "Discard Parameter {}, non-finite observables".format(event_id)
                )
                discarded_points += 1
                continue
            if self.logTrafo_ and np.any(temp_data[:, 0] <= 0):
                raise ValueError(
                    "log_trafo requires positive observables, but "
                    "parameter point {} has values <= 0".format(event_id)
                )
            if self.max_rel_uncertainty_data_ is not None:
                statErrMax = self._max_rel_error(temp_data)
                if statErrMax > self.max_rel_uncertainty_data_:
                    logging.info(
                        "Discard Parameter {}, stat err = {:.2f}".format(
                            event_id, statErrMax
                        )
                    )
                    discarded_points += 1
                    continue
            self.design_points.append(dataDict[event_id]["parameter"])
            if not self.logTrafo_:
                self.model_data.append(temp_data[:, 0])
                self.model_data_err.append(temp_data[:, 1])
            else:
                # errors in log space are the relative errors
                self.model_data.append(np.log(temp_data[:, 0]))
                self.model_data_err.append(np.abs(temp_data[:, 1] / temp_data[:, 0]))
        if len(self.model_data) == 0:
            raise ValueError(
                "All training points in {} were discarded".format(dataFile)
            )
        self.design_points = np.array(self.design_points)
        self.model_data = np.array(self.model_data)
        self.model_data_err = np.nan_to_num(np.abs(np.array(self.model_data_err)))
        logging.info("All training data are loaded.")
        logging.info(
            "Training dataset size: {}, discarded points: {}".format(
                len(self.model_data), discarded_points
            )
        )

    def train_emulator_auto_mask(self, **train_kwargs):
        """Train the emulator on all training points."""
        self.train_emulator(np.ones(self.nev, dtype=bool), **train_kwargs)

    # -------------------------
    # Validation
    # -------------------------
    def _predictions_in_log_space(self):
        """True if predict() returns the mean and covariance in log space."""
        return self.logTrafo_ and not self.exp_and_cov_diagonal_

    def _predict_log_space(self, X, include_noise):
        """predict() of a log-transformed emulator in log space, also if it
        returns the predictions in the original scale."""
        had_flag = "exp_and_cov_diagonal_" in self.__dict__
        flag = self.__dict__.get("exp_and_cov_diagonal_")
        self.exp_and_cov_diagonal_ = False
        try:
            return self.predict(X, return_cov=True, include_noise=include_noise)
        finally:
            if had_flag:
                self.exp_and_cov_diagonal_ = flag
            else:
                del self.exp_and_cov_diagonal_

    def sample_y(self, X, n_samples=1, random_state=None, include_noise=False):
        """
        Sample model output at the parameter points `X` from the predicted
        Gaussian distribution, with the same uncertainty as predict() (see
        `include_noise`). The points are sampled independently, only the
        correlations between the observables are taken into account.

        The samples are in the same space as the predictions: in log space for
        log-transformed emulators, unless exp_and_cov_diagonal is set, in which
        case the samples are drawn in log space and exponentiated
        (log-normal).

        Returns an array with shape ``(nsamples_X, n_samples, nobs)``.
        """
        X = np.atleast_2d(X)
        rng = np.random.default_rng(random_state)
        back_transform = self.logTrafo_ and not self._predictions_in_log_space()
        if back_transform:
            mean, cov = self._predict_log_space(X, include_noise)
        else:
            mean, cov = self.predict(X, return_cov=True, include_noise=include_noise)
        samples = np.stack(
            [
                rng.multivariate_normal(m, c, size=n_samples, method="eigh")
                for m, c in zip(np.asarray(mean), np.asarray(cov))
            ]
        )
        if back_transform:
            samples = np.exp(samples)
        return samples

    def _validation_masks(self, number_test_points, random_points, seed):
        """Boolean masks of the training and test points. The test points are
        the last number_test_points points, or randomly chosen points if
        random_points is True."""
        if not 0 <= number_test_points < self.nev:
            raise ValueError(
                "number_test_points must be between 0 and {}, got {}".format(
                    self.nev - 1, number_test_points
                )
            )
        if random_points:
            rng = np.random.default_rng(seed)
            test_idx = rng.choice(self.nev, number_test_points, replace=False)
        else:
            test_idx = np.arange(self.nev - number_test_points, self.nev)
        test_mask = np.zeros(self.nev, dtype=bool)
        test_mask[test_idx] = True
        return ~test_mask, test_mask

    def _validation_output(self, mask):
        """Emulator predictions and their standard deviations, and the
        training data and their errors at the points in mask, all in the
        original scale of the observables."""
        # the test points are noisy simulations, so the noise of the emulator
        # is included in the predicted errors
        pred_mean, pred_cov = self.predict(
            self.design_points[mask, :], return_cov=True, include_noise=True
        )
        pred_mean = np.asarray(pred_mean)
        pred_std = np.sqrt(np.diagonal(np.asarray(pred_cov), axis1=1, axis2=2))

        if self._predictions_in_log_space():
            pred_std = pred_std * np.exp(pred_mean)
            pred_mean = np.exp(pred_mean)

        if self.logTrafo_:
            data = np.exp(self.model_data[mask, :])
            data_err = self.model_data_err[mask, :] * data
        else:
            data = self.model_data[mask, :]
            data_err = self.model_data_err[mask, :]

        return (
            pred_mean.reshape(-1, self.nobs),
            pred_std.reshape(-1, self.nobs),
            np.array(data).reshape(-1, self.nobs),
            np.array(data_err).reshape(-1, self.nobs),
        )

    @keep_trained_state
    def test_emulator_errors(
        self, number_test_points=1, random_points=False, seed=None, **train_kwargs
    ):
        """
        Train the emulator without number_test_points test points and predict
        at the test points. The test points are the last points of the
        training data, or randomly chosen points if random_points is True
        (reproducible with seed). train_kwargs are passed to train_emulator.
        The trained emulator is not changed.

        Returns the emulator predictions, their errors, the values of the
        observables and their errors at the test points, as four arrays of
        shape (number_test_points, nobs) in the original scale of the
        observables.
        """
        logging.info("Validating emulator ...")
        train_mask, test_mask = self._validation_masks(
            number_test_points, random_points, seed
        )
        self.train_emulator(train_mask, **train_kwargs)
        return self._validation_output(test_mask)

    @keep_trained_state
    def test_emulator_errors_with_training_points(
        self, number_test_points=1, random_points=False, seed=None, **train_kwargs
    ):
        """
        Train the emulator without number_test_points test points (chosen as
        in test_emulator_errors) and predict at the training points. The
        resulting errors should be very small. The trained emulator is not
        changed.

        Returns the same four arrays as test_emulator_errors, with
        (nev - number_test_points) rows.
        """
        logging.info("Validating emulator at the training points ...")
        train_mask, _ = self._validation_masks(number_test_points, random_points, seed)
        self.train_emulator(train_mask, **train_kwargs)
        return self._validation_output(train_mask)
