"""
Training for Gaussian process emulators.

Uses the `Gaussian process regression with heteroskedastic emulator
<https://hetgpy.readthedocs.io/en/v1.0.4/>`_ implemented in the `hetgpy` package.
"""

import contextlib
import io
import logging
import pickle

import numpy as np
from hetgpy import hetGP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .emulator_base import EmulatorBase, check_npc, number_of_pcs, truncation_signal

logger = logging.getLogger(__name__)


class EmulatorHetGP(EmulatorBase):
    """
    Emulator with heteroskedastic GPs of the hetgpy package.

    The GPs emulate the principal components of the (standardized)
    observables.

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
        must be positive, and predict() returns the mean and covariance in log
        space. Experimental data used with the emulator must then be
        log-transformed as well.
    max_rel_uncertainty_data : float or None, default=None
        Training points with a larger relative statistical error of any
        observable are discarded. None disables this filter.
    exp_and_cov_diagonal : bool, default=False
        If True, predict() returns exp(mean) and a diagonal covariance in the
        original scale of the observables (see `EmulatorBase`). Requires
        ``log_trafo=True``.
    npc : int or float, default=0.99
        Number of PCs (int >= 1) or fraction of the explained variance (float
        in (0, 1)).

    Raises
    ------
    ValueError
        If `npc` is out of range, or for invalid training data or options
        (see `EmulatorBase`).
    TypeError
        If `npc` is neither an int nor a float.
    """

    def __init__(
        self,
        training_set_path,
        parameter_file,
        *,
        log_trafo=False,
        max_rel_uncertainty_data=None,
        exp_and_cov_diagonal=False,
        npc=0.99,
    ):
        super().__init__(
            training_set_path,
            parameter_file,
            log_trafo=log_trafo,
            max_rel_uncertainty_data=max_rel_uncertainty_data,
            exp_and_cov_diagonal=exp_and_cov_diagonal,
        )

        # The outputs are standardized and transformed with a PCA in
        # train_emulator(), keeping npc PCs (int) or the PCs explaining the
        # fraction npc (float) of the variance. The GP emulators are then
        # trained on the resulting principal components.
        check_npc(npc)
        self.npc = npc

    def _fit_output_pca(self, data, data_err=None):
        """
        Fit the output standardization and PCA to the training data.

        Also compute the truncation covariance. `data` are the training data
        and `data_err` their statistical errors.
        """
        self.scaler_ = StandardScaler()
        standardized_outputs = self.scaler_.fit_transform(data)
        # the exact (full) SVD is used as in EmulatorSklearn, since sklearn's
        # default can choose a randomized, approximate solver
        full_pca = PCA(svd_solver="full").fit(standardized_outputs)
        self.npc_ = number_of_pcs(self.npc, full_pca.explained_variance_ratio_)
        self.pca_ = PCA(n_components=self.npc_, svd_solver="full")
        self.train_pcs_ = self.pca_.fit_transform(standardized_outputs)
        self._compute_truncation_cov(data, self.train_pcs_, data_err)
        logger.info(
            f"Using {self.npc_} PCs, which explain "
            f"{self.pca_.explained_variance_ratio_.sum():.5f} of the variance"
        )

    def _compute_truncation_cov(self, data, data_pca, data_err=None):
        """
        Compute the covariance of the PCs discarded by the output PCA.

        The covariance is in observable units. It is added to the predicted
        covariance, since the emulator cannot resolve this part of the
        variance. `data` are the training data, `data_pca` their principal
        components and `data_err` their statistical errors.

        The truncation covariance also contains the statistical noise of the
        training data in the discarded PC directions. Its signal part without
        this noise, _cov_trunc_signal, is used for predictions of the model
        function (include_noise=False).
        """
        standardized_outputs = self.scaler_.transform(data)
        residuals = standardized_outputs - self.pca_.inverse_transform(data_pca)
        scale = self.scaler_.scale_
        # covariances in standardized units
        trunc_cov_scaled = np.cov(residuals, rowvar=False)
        self._cov_trunc = trunc_cov_scaled * np.outer(scale, scale)
        self._cov_trunc_signal = self._cov_trunc
        if data_err is not None:
            noise_cov_scaled = np.diag(np.mean((data_err / scale) ** 2, axis=0))
            self._cov_trunc_signal = truncation_signal(
                trunc_cov_scaled, noise_cov_scaled
            ) * np.outer(scale, scale)

    def __getstate__(self):
        """
        Prepare a pickleable state.

        The fitted hetgpy models can be serialized with the standard
        ``pickle`` module, but not with ``dill``, which is used to save the
        emulators. We therefore store the models as a ``pickle`` byte string,
        which ``dill`` can serialize, so that the unpickled emulator contains
        exactly the trained models.
        """
        state = self.__dict__.copy()
        gps = state.pop("gps_", None)
        if gps is not None:
            state["_gps_pickle"] = pickle.dumps(gps)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        gps_pickle = state.pop("_gps_pickle", None)
        self.__dict__.update(state)
        if gps_pickle is not None:
            self.gps_ = pickle.loads(gps_pickle)
        # otherwise the emulator was not trained and stays untrained

    def train_emulator(self, event_mask):
        """
        Train the emulator on the training points selected by `event_mask`.

        The output standardization and PCA are fitted to the selected training
        points only, then one hetGP model is trained per PC.

        Parameters
        ----------
        event_mask : array_like of bool of shape (nev,)
            Mask of the training points to use.
        """
        # Subselect training data
        event_mask = np.asarray(event_mask, dtype=bool)
        design_points = self.design_points[event_mask, :]
        # fit the output PCA only to the training points
        self._fit_output_pca(
            self.model_data[event_mask, :], self.model_data_err[event_mask, :]
        )
        Z = self.train_pcs_

        nev = design_points.shape[0]
        logger.info(
            f"Training {self.npc_} hetGP models for the PCs with {nev} "
            "training points ..."
        )

        # Train one hetGP model per principal component of the outputs.
        self.gps_ = []
        for j in range(self.npc_):
            gp = hetGP()
            # hetgpy prints messages (e.g. when it returns a homoskedastic
            # model) to stdout, they are logged instead
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                gp.mleHetGP(
                    X=design_points,
                    Z=Z[:, j],
                    covtype="Matern3_2",
                    maxit=100,
                )
            if output.getvalue().strip():
                logger.debug(f"hetGP model {j + 1}: {output.getvalue().strip()}")
            # a failed fit gives non-finite predictions
            pred = gp.predict(x=design_points)
            if not (
                np.all(np.isfinite(pred["mean"])) and np.all(np.isfinite(pred["sd2"]))
            ):
                logger.warning(
                    f"The hetGP model of PC {j + 1} gives non-finite predictions at "
                    "the training points, the fit failed"
                )
            self.gps_.append(gp)
            logger.debug(f"hetGP model {j + 1}/{self.npc_} trained")
        logger.info("Emulator training finished")

    def predict(self, X, return_cov=True, include_noise=False):
        """
        Predict model output at the parameter points `X`.

        By default, the covariance is the uncertainty of the emulated model
        function. With `include_noise`, the noise variance estimated by the
        hetGP models (nugs) is included, i.e. the uncertainty of a new noisy
        simulation.

        Parameters
        ----------
        X : array_like of shape (nsamples, nparameters)
            Parameter points. A 1D array is treated as a single point.
        return_cov : bool, default=True
            If True, the covariance is returned as well.
        include_noise : bool, default=False
            If True, the noise variance estimated by the hetGP models is
            included in the covariance.

        Returns
        -------
        mean : ndarray of shape (nsamples, nobs)
            Predicted mean.
        cov : ndarray of shape (nsamples, nobs, nobs)
            Covariance between the observables. Only returned if `return_cov`
            is True.
        """
        X = np.atleast_2d(X)
        n_theta = X.shape[0]

        # Predict principal components at given parameter points
        pc_means = np.zeros((self.npc_, n_theta))
        pc_vars = np.zeros((self.npc_, n_theta))

        for j, gp in enumerate(self.gps_):
            pred = gp.predict(x=X)
            pc_means[j] = np.asarray(pred["mean"]).reshape(-1)
            pc_vars[j] = np.asarray(pred["sd2"]).reshape(-1)
            if include_noise:
                pc_vars[j] += np.asarray(pred["nugs"]).reshape(-1)

        # Reconstruct observables from PCs
        mean = self.scaler_.inverse_transform(self.pca_.inverse_transform(pc_means.T))

        # Covariance in the space of the observables: the PC variances are
        # transformed with the PCA components and the standardization scales
        W = self.pca_.components_.T * self.scaler_.scale_[:, None]  # (nobs, npc)
        pc_vars = np.maximum(pc_vars, 0.0)
        cov = np.einsum("ik,kn,jk->nij", W, pc_vars, W)
        cov += self._cov_trunc if include_noise else self._cov_trunc_signal

        if self.exp_and_cov_diagonal:
            # If the emulator is trained on the log of the data, we return the
            # predictions in the original scale with diagonal covariance matrix.
            mean = np.exp(mean)
            std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
            cov = np.zeros_like(cov)
            idx = np.arange(self.nobs)
            cov[:, idx, idx] = (std * mean) ** 2

        if return_cov:
            return mean, cov
        return mean
