"""
Common base class of the emulators.

It loads the training data and the model parameter file, applies the quality
filter and the log transformation of the training data, and implements the
validation functions and ``sample_y``. The emulators implement
``train_emulator(event_mask)`` and ``predict(X, return_cov=True,
include_noise=False)``.
"""

import logging
import pickle

import numpy as np

from . import keep_trained_state, parse_model_parameter_file

logger = logging.getLogger(__name__)


def check_npc(npc):
    """
    Check the number of principal components `npc`.

    Parameters
    ----------
    npc : int or float
        Number of principal components (PCs), an int >= 1, or the fraction of
        the explained variance, a float in (0, 1).

    Raises
    ------
    ValueError
        If an int `npc` is < 1 or a float `npc` is not in (0, 1).
    TypeError
        If `npc` is neither an int nor a float (bool is not accepted).
    """
    if isinstance(npc, (int, np.integer)) and not isinstance(npc, bool):
        if npc < 1:
            raise ValueError(f"npc must be >= 1, got {npc}")
    elif isinstance(npc, (float, np.floating)):
        if not 0 < npc < 1:
            raise ValueError(
                "A float npc is the fraction of the explained variance and "
                f"must be in (0, 1), got {npc}"
            )
    else:
        raise TypeError(f"npc must be an int or a float, got {npc!r}")


def number_of_pcs(npc, explained_variance_ratio):
    """
    Return the number of PCs to use for `npc`.

    A float `npc` selects the smallest number of PCs that explain more than
    this fraction of the variance, as in sklearn's PCA. An int `npc` larger
    than the number of available PCs is reduced to that number, with a
    warning.

    Parameters
    ----------
    npc : int or float
        Number of PCs or fraction of the explained variance (see `check_npc`).
    explained_variance_ratio : array_like
        Explained variance ratios of all PCs.

    Returns
    -------
    int
        Number of PCs, at most the number of available PCs.
    """
    n_available = len(explained_variance_ratio)
    if isinstance(npc, (float, np.floating)):
        n = np.searchsorted(np.cumsum(explained_variance_ratio), npc, side="right") + 1
        return int(min(n, n_available))
    if npc > n_available:
        logger.warning(
            f"npc = {npc} is larger than the number of available PCs, using all "
            f"{n_available} PCs"
        )
    return int(min(npc, n_available))


def truncation_signal(trunc_cov, noise_cov):
    """
    Remove the statistical noise of the training data from a truncation
    covariance.

    In each eigendirection of `trunc_cov`, the noise variance of `noise_cov`
    in that direction is subtracted, down to zero. The result is positive
    semi-definite and not larger than `trunc_cov`.

    Parameters
    ----------
    trunc_cov : ndarray of shape (nobs, nobs)
        Truncation covariance of the discarded PCs.
    noise_cov : ndarray of shape (nobs, nobs)
        Covariance of the statistical noise of the training data, in the same
        (e.g. standardized) units as `trunc_cov`.

    Returns
    -------
    ndarray of shape (nobs, nobs)
        Signal part of the truncation covariance.
    """
    vals, vecs = np.linalg.eigh(0.5 * (trunc_cov + trunc_cov.T))
    noise = np.einsum("ik,ij,jk->k", vecs, noise_cov, vecs)
    signal = np.clip(vals - noise, 0.0, None)
    return (vecs * signal) @ vecs.T


class EmulatorBase:
    """
    Base class of the emulators.

    It loads the training data and the model parameter file and implements
    the validation functions and `sample_y`. Training points with non-finite
    observables are always discarded. All arguments except the two paths are
    keyword-only in all emulators. Subclasses implement
    ``train_emulator(event_mask)`` and ``predict(X, return_cov=True,
    include_noise=False)``.

    Parameters
    ----------
    training_set_path : str or path-like
        Path to the pickle file with the training data, a dictionary
        ``{event_id: {'parameter': array (nparameters,), 'obs': array (2,
        nobs) with the values and statistical errors}}``.
    parameter_file : str or path-like
        Path to the model parameter file.
    log_trafo : bool, default=False
        If True, the emulator is trained on the log of the observables, which
        must be positive. predict() then returns the mean and covariance in
        log space, and experimental data used with the emulator must be
        log-transformed as well.
    max_rel_uncertainty_data : float or None, default=None
        Training points with a larger relative statistical error of any
        observable are discarded. None disables this filter.
    exp_and_cov_diagonal : bool, default=False
        Only with ``log_trafo=True``: predict() returns the predictions
        transformed back to the original scale of the observables, with
        diagonal covariance matrices (EmulatorSparseGP keeps the correlations
        between the observables). The mean is exp(mean) of the log-space
        prediction, i.e. the median of the log-normal distribution, and the
        variance is the first-order (delta method) approximation
        exp(mean)^2 * var.

    Raises
    ------
    ValueError
        If `exp_and_cov_diagonal` is True but `log_trafo` is False, if the
        number of parameters of the training data and the parameter file
        differ, if `log_trafo` is True and a training point has observables
        <= 0, or if all training points are discarded.
    """

    def __init__(
        self,
        training_set_path,
        parameter_file,
        *,
        log_trafo=False,
        max_rel_uncertainty_data=None,
        exp_and_cov_diagonal=False,
    ):
        self.log_trafo = log_trafo
        self.max_rel_uncertainty_data = max_rel_uncertainty_data
        self.exp_and_cov_diagonal = exp_and_cov_diagonal
        if not self.log_trafo and self.exp_and_cov_diagonal:
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
                f"The training data have {self.nparameters} parameters, but the "
                f"parameter file {parameter_file} has {len(self.pardict)}"
            )

    # attributes of the emulators saved with versions < 3.0.0 and their
    # current names (None for removed attributes), in the order in which they
    # are renamed, and default values of attributes that did not exist in
    # these versions
    _legacy_attributes = []
    _legacy_defaults = {}

    def __setstate__(self, state):
        """Restore the state after unpickling, renaming legacy attributes."""
        self.__dict__.update(self._migrate_legacy_state(state))

    @classmethod
    def _migrate_legacy_state(cls, state):
        """Rename the attributes of emulators saved with versions < 3.0.0."""
        if "logTrafo_" not in state:
            return state
        if state.get("parameterTrafoPCA_", False):
            raise ValueError(
                "The emulator was trained with parameterTrafoPCA=True, which was "
                "removed in version 3.0.0. Use version v2.0.1 to load it."
            )
        state = dict(state)
        renames = [
            ("logTrafo_", "log_trafo"),
            ("max_rel_uncertainty_data_", "max_rel_uncertainty_data"),
            ("exp_and_cov_diagonal_", "exp_and_cov_diagonal"),
        ] + cls._legacy_attributes
        for old, new in renames:
            if new is None:
                state.pop(old, None)
            elif old in state and new not in state:
                state[new] = state.pop(old)
        for name, default in cls._legacy_defaults.items():
            if name not in state:
                state[name] = default(state) if callable(default) else default
        return state

    # -------------------------
    # Training data
    # -------------------------
    @staticmethod
    def _max_rel_error(temp_data):
        """
        Largest relative statistical error of a training point.

        Observables that are exactly zero have no relative error and are
        ignored.
        """
        nonzero = temp_data[:, 0] != 0
        return np.max(
            np.abs(temp_data[nonzero, 1] / temp_data[nonzero, 0]), initial=0.0
        )

    def _load_training_data_pickle(self, data_file):
        """Read the training data of all parameter points from a pickle file."""
        logger.info(f"Loading the training data from {data_file} ...")
        self.model_data = []
        self.model_data_err = []
        self.design_points = []
        with open(data_file, "rb") as fp:
            data_dict = pickle.load(fp)

        # Sort keys in ascending order
        sorted_event_ids = sorted(data_dict.keys(), key=lambda x: int(x))

        n_nonfinite = 0
        n_filtered = 0
        for event_id in sorted_event_ids:
            temp_data = data_dict[event_id]["obs"].transpose()
            if not np.all(np.isfinite(temp_data[:, 0])):
                logger.warning(
                    f"Discarding training point {event_id}: non-finite observables"
                )
                n_nonfinite += 1
                continue
            if self.max_rel_uncertainty_data is not None:
                stat_err_max = self._max_rel_error(temp_data)
                if stat_err_max > self.max_rel_uncertainty_data:
                    logger.info(
                        f"Discarding training point {event_id}: relative "
                        f"statistical error {stat_err_max:.3g} > "
                        f"{self.max_rel_uncertainty_data}"
                    )
                    n_filtered += 1
                    continue
            # after the error filter, which can discard such points
            if self.log_trafo and np.any(temp_data[:, 0] <= 0):
                raise ValueError(
                    "log_trafo requires positive observables, but "
                    f"parameter point {event_id} has values <= 0"
                )
            self.design_points.append(data_dict[event_id]["parameter"])
            if not self.log_trafo:
                self.model_data.append(temp_data[:, 0])
                self.model_data_err.append(temp_data[:, 1])
            else:
                # errors in log space are the relative errors
                self.model_data.append(np.log(temp_data[:, 0]))
                self.model_data_err.append(np.abs(temp_data[:, 1] / temp_data[:, 0]))
        if len(self.model_data) == 0:
            raise ValueError(f"All training points in {data_file} were discarded")
        self.design_points = np.array(self.design_points)
        self.model_data = np.array(self.model_data)
        self.model_data_err = np.abs(np.array(self.model_data_err))
        n_nonfinite_err = int(np.sum(~np.isfinite(self.model_data_err)))
        if n_nonfinite_err > 0:
            logger.warning(
                f"Setting {n_nonfinite_err} non-finite statistical errors of the "
                "training data to 0"
            )
            self.model_data_err[~np.isfinite(self.model_data_err)] = 0.0
        n_discarded = n_nonfinite + n_filtered
        logger.info(
            f"Loaded {len(self.model_data)} training points with "
            f"{self.model_data.shape[1]} observables, discarded {n_discarded} "
            f"({n_nonfinite} non-finite, {n_filtered} with too large errors)"
        )

    def train_emulator_auto_mask(self, **train_kwargs):
        """
        Train the emulator on all training points.

        Parameters
        ----------
        **train_kwargs
            Keyword arguments passed to ``train_emulator``.
        """
        self.train_emulator(np.ones(self.nev, dtype=bool), **train_kwargs)

    # -------------------------
    # Validation
    # -------------------------
    def _predictions_in_log_space(self):
        """True if predict() returns the mean and covariance in log space."""
        return self.log_trafo and not self.exp_and_cov_diagonal

    def _predict_log_space(self, X, include_noise):
        """
        Predict in log space with a log-transformed emulator.

        This also works if predict() returns the predictions in the original
        scale (exp_and_cov_diagonal).
        """
        flag = self.exp_and_cov_diagonal
        self.exp_and_cov_diagonal = False
        try:
            return self.predict(X, return_cov=True, include_noise=include_noise)
        finally:
            self.exp_and_cov_diagonal = flag

    def sample_y(self, X, n_samples=1, seed=None, include_noise=False):
        """
        Sample model output from the predicted Gaussian distribution.

        The samples have the same uncertainty as predict() (see
        `include_noise`). The points are sampled independently, only the
        correlations between the observables are taken into account.

        The samples are in the same space as the predictions: in log space for
        log-transformed emulators, unless exp_and_cov_diagonal is set, in
        which case the samples are drawn in log space and exponentiated
        (log-normal).

        Parameters
        ----------
        X : array_like of shape (nsamples_X, nparameters)
            Parameter points. A 1D array is treated as a single point.
        n_samples : int, default=1
            Number of samples per parameter point.
        seed : int, numpy.random.Generator or None, default=None
            Seed or generator passed to ``numpy.random.default_rng``.
        include_noise : bool, default=False
            If True, the noise fitted by the emulator is included in the
            uncertainty, as in predict().

        Returns
        -------
        ndarray of shape (nsamples_X, n_samples, nobs)
            Samples of the observables.
        """
        X = np.atleast_2d(X)
        rng = np.random.default_rng(seed)
        back_transform = self.log_trafo and not self._predictions_in_log_space()
        if back_transform:
            mean, cov = self._predict_log_space(X, include_noise)
        else:
            mean, cov = self.predict(X, return_cov=True, include_noise=include_noise)
        samples = np.stack(
            [
                rng.multivariate_normal(m, c, size=n_samples, method="eigh")
                for m, c in zip(np.asarray(mean), np.asarray(cov), strict=True)
            ]
        )
        if back_transform:
            samples = np.exp(samples)
        return samples

    def _validation_masks(self, n_test_points, random_points, seed, min_test_points):
        """
        Boolean masks of the training and test points.

        The test points are the last n_test_points points, or randomly
        chosen points if random_points is True. At least `min_test_points`
        test points and 2 training points are required.
        """
        if not min_test_points <= n_test_points <= self.nev - 2:
            raise ValueError(
                f"n_test_points must be between {min_test_points} and "
                f"{self.nev - 2} (at least 2 training points), got {n_test_points}"
            )
        if random_points:
            rng = np.random.default_rng(seed)
            test_idx = rng.choice(self.nev, n_test_points, replace=False)
        else:
            test_idx = np.arange(self.nev - n_test_points, self.nev)
        test_mask = np.zeros(self.nev, dtype=bool)
        test_mask[test_idx] = True
        return ~test_mask, test_mask

    def _validation_output(self, mask):
        """
        Predictions and training data at the points in `mask`.

        Returns the emulator predictions and their standard deviations, and
        the training data and their errors, all in the original scale of the
        observables.
        """
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

        if self.log_trafo:
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

    @staticmethod
    def _test_points_text(n_test_points, random_points):
        """Description of the chosen test points for the log messages."""
        points = "point" if n_test_points == 1 else "points"
        if random_points:
            return f"{n_test_points} random training {points}"
        return f"the last {n_test_points} training {points}"

    @keep_trained_state
    def test_emulator_errors(
        self, n_test_points=1, random_points=False, seed=None, **train_kwargs
    ):
        """
        Validate the emulator at test points excluded from the training.

        The emulator is trained without the test points and predicts at the
        test points. The predicted errors include the noise of the emulator,
        since the test points are noisy simulations. The trained emulator is
        not changed.

        Parameters
        ----------
        n_test_points : int, default=1
            Number of test points, between 1 and nev - 2.
        random_points : bool, default=False
            If False, the test points are the last points of the training
            data. If True, they are chosen randomly.
        seed : int or None, default=None
            Seed for the random choice of the test points.
        **train_kwargs
            Keyword arguments passed to ``train_emulator``.

        Returns
        -------
        pred_mean, pred_err, data, data_err : ndarray
            The emulator predictions, their errors, the values of the
            observables and their errors at the test points, each of shape
            (n_test_points, nobs), in the original scale of the
            observables.

        Raises
        ------
        ValueError
            If `n_test_points` is not between 1 and nev - 2.
        """
        train_mask, test_mask = self._validation_masks(
            n_test_points, random_points, seed, min_test_points=1
        )
        logger.info(
            "Validating the emulator with "
            f"{self._test_points_text(n_test_points, random_points)} as test "
            "points ..."
        )
        self.train_emulator(train_mask, **train_kwargs)
        return self._validation_output(test_mask)

    @keep_trained_state
    def test_emulator_errors_with_training_points(
        self, n_test_points=1, random_points=False, seed=None, **train_kwargs
    ):
        """
        Validate the emulator at its training points.

        The emulator is trained without n_test_points test points (chosen
        as in `test_emulator_errors`) and predicts at the training points. The
        resulting errors should be very small. The trained emulator is not
        changed.

        Parameters
        ----------
        n_test_points : int, default=1
            Number of test points excluded from the training, between 0 and
            nev - 2.
        random_points : bool, default=False
            If True, the test points are chosen randomly, otherwise they are
            the last points of the training data.
        seed : int or None, default=None
            Seed for the random choice of the test points.
        **train_kwargs
            Keyword arguments passed to ``train_emulator``.

        Returns
        -------
        pred_mean, pred_err, data, data_err : ndarray
            The same four arrays as `test_emulator_errors`, with
            (nev - n_test_points) rows.

        Raises
        ------
        ValueError
            If `n_test_points` is not between 0 and nev - 2.
        """
        train_mask, _ = self._validation_masks(
            n_test_points, random_points, seed, min_test_points=0
        )
        logger.info(
            "Validating the emulator at the training points, without "
            f"{self._test_points_text(n_test_points, random_points)} as test "
            "points ..."
        )
        self.train_emulator(train_mask, **train_kwargs)
        return self._validation_output(train_mask)
