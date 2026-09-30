"""
Sparse variational Gaussian processes (SVGP) with a PCA of the outputs.

Implements a PCA-reduced sparse variational GP emulator using JAX and optax:
the single emulator PCASparseGPEmulator and the ensemble PCASparseGPEnsemble
(optionally trained on bootstrap samples). The emulator interface of the package is
EmulatorSparseGP in emulator_sparse_gp.py.

Importing this module enables 64-bit floats in JAX (``jax_enable_x64``) for
the whole Python process. In single precision, the Cholesky decompositions
and the predicted covariances are not accurate enough for the likelihood in
the MCMC.

References
----------
.. [1] Hensman et al. (2015), "Scalable Variational Gaussian Process
   Classification".
.. [2] Lakshminarayanan et al. (2017), "Simple and Scalable Predictive
   Uncertainty Estimation using Deep Ensembles".
"""

import functools
import logging

import jax
import numpy as np

# must be set before any JAX arrays are created
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .emulator_base import check_npc, number_of_pcs, truncation_signal

_INIT_STRATEGIES = ("maxmin", "kmeans", "kmeans_pp", "random", "sobol")

# independent random subkeys of a key for the different random steps
_SUBKEY_IDS = {"init": 0, "numpy": 1, "train": 2, "bootstrap": 3}


def _subkey(key, purpose):
    """Independent subkey of `key` for `purpose` (see `_SUBKEY_IDS`)."""
    return jax.random.fold_in(key, _SUBKEY_IDS[purpose])


def _restore_state_on_error(fit):
    """
    Restore the attributes of the object if the decorated `fit` fails, so
    that a failed refit keeps the previously trained state.
    """

    @functools.wraps(fit)
    def wrapper(self, *args, **kwargs):
        state = self.__dict__.copy()
        try:
            return fit(self, *args, **kwargs)
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(state)
            raise

    return wrapper


def _check_y_err(Y_err, Y):
    """
    Check the shape of the observation errors `Y_err` of the outputs `Y`,
    (N, P) standard deviations or (N, P, P) covariance matrices.
    """
    N, P = np.shape(Y)
    Y_err = np.asarray(Y_err, dtype=float)
    if Y_err.ndim == 2:
        if Y_err.shape != (N, P):
            raise ValueError(
                f"Y_err shape {Y_err.shape} must match Y shape {(N, P)} for the "
                "(N, P) diagonal-error format."
            )
    elif Y_err.ndim == 3:
        if Y_err.shape != (N, P, P):
            raise ValueError(
                f"Y_err shape {Y_err.shape} expected {(N, P, P)} for the (N, P, P) "
                "full-covariance format."
            )
        if np.any(np.diagonal(Y_err, axis1=1, axis2=2) < 0):
            raise ValueError(
                "Y_err contains covariance matrices with negative diagonal "
                "entries. Check your input."
            )
    else:
        raise ValueError(f"Y_err must be shape (N, P) or (N, P, P); got {Y_err.shape}.")


def _fit_pca(Yn, n_pc):
    """
    Fit the PCA of the standardized outputs `Yn` with `n_pc` components.

    `n_pc` is the number of PCs or the fraction of the explained variance
    (see `number_of_pcs`); it is capped at the number of available PCs. The
    exact (full) SVD is used, since sklearn's default can choose a
    randomized, approximate solver. Returns the fitted PCA and the PCs.
    """
    Yn = np.asarray(Yn)
    full_pca = PCA(svd_solver="full").fit(Yn)
    pca = PCA(
        n_components=number_of_pcs(n_pc, full_pca.explained_variance_ratio_),
        svd_solver="full",
    )
    return pca, pca.fit_transform(Yn)


logger = logging.getLogger(__name__)


# =============================================================================
# Kernel functions
# =============================================================================


def rbf_kernel(x1, x2, ls, var):
    """
    Squared exponential (RBF) kernel with ARD lengthscales.

    Parameters
    ----------
    x1 : array (N1, D)
        First set of input points.
    x2 : array (N2, D)
        Second set of input points.
    ls : array (D,) or float
        Lengthscales.
    var : float
        Kernel variance.

    Returns
    -------
    array (N1, N2)
        Kernel matrix ``var * exp(-0.5 * r^2)``, where ``r`` is the distance
        of the points scaled by the lengthscales.
    """
    x1 = x1 / ls
    x2 = x2 / ls
    sq = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return var * jnp.exp(-0.5 * sq)


def matern32_kernel(x1, x2, ls, var):
    """
    Matern-3/2 kernel with ARD lengthscales.

    Parameters
    ----------
    x1 : array (N1, D)
        First set of input points.
    x2 : array (N2, D)
        Second set of input points.
    ls : array (D,) or float
        Lengthscales.
    var : float
        Kernel variance.

    Returns
    -------
    array (N1, N2)
        Kernel matrix ``var * (1 + sqrt(3) r) * exp(-sqrt(3) r)``, where ``r``
        is the distance of the points scaled by the lengthscales. A small
        constant (1e-12) is added to ``r^2`` to keep the gradient finite.
    """
    x1 = x1 / ls
    x2 = x2 / ls
    dist = jnp.sqrt(jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1) + 1e-12)
    sqrt3 = jnp.sqrt(3.0)
    return var * (1 + sqrt3 * dist) * jnp.exp(-sqrt3 * dist)


# =============================================================================
# PCASparseGPEmulator
# =============================================================================


class PCASparseGPEmulator:
    """
    PCA-reduced sparse variational Gaussian process emulator.

    The outputs are standardized and reduced with PCA. Each retained
    principal component is modeled by an independent sparse variational GP
    with inducing points. The kernel is the sum of an RBF and a Matern-3/2
    kernel with shared ARD lengthscales.

    Notes
    -----
    Parameterization: whitened SVGP (GPflow convention).

    - Prior on inducing outputs: ``p(u) = N(0, Kzz)``
    - Whitened variable: ``v = Lz^{-1} u``, ``p(v) = N(0, I)``
    - Variational distribution: ``q(v) = N(m, S)``, ``S = L L^T``
    - Posterior mean at X*: ``Kxz @ Lz^{-T} @ m``
    - Posterior variance at X*:
      ``Kxx - sum(A_half^2, axis=0) + sum((A_half.T @ L)^2, axis=1)``,
      where ``A_half = Lz^{-1} @ Kxz.T`` (one triangular solve only).

    Hyperparameters are point estimates, not posteriors. The kernel
    parameters (log_lengthscale, log_var_rbf, log_var_mat) and the nugget
    (log_noise) are optimized with ADAM on the ELBO, without a hyperprior
    (type-II maximum likelihood). Despite their names, these parameters are
    unconstrained values that are mapped to positive values with a softplus.
    The predictive distribution p(f* | X*, X, Y) is therefore conditioned on
    fixed hyperparameters θ̂ and does NOT integrate over p(θ | X, Y).
    Consequences:

    - The predictive variance is the GP posterior variance at fixed θ, which
      systematically underestimates the total uncertainty when the
      likelihood surface is broad or multi-modal in hyperparameter space.
    - This is most problematic with short lengthscales (overfitting), very
      small datasets, or when the kernel family is misspecified.
    - PCASparseGPEnsemble partially compensates: different random seeds
      produce different θ̂ values, so the spread between members
      captures some hyperparameter uncertainty. However, if all members
      collapse to the same optimum (common with large N), this compensation
      vanishes.

    Principled alternatives (not implemented):

    - MCMC over θ (e.g. HMC in NumPyro/BlackJAX): exact but expensive.
    - Laplace approximation around θ̂: adds a Gaussian correction to the
      predictive variance with cost O(P_θ²), where P_θ is the number of
      kernel parameters.
    - VOGP / hyperparameter variational inference: jointly optimizes q(θ)
      alongside the variational GP posterior; supported in GPflux/GPyTorch.

    For Bayesian inference / MCMC-based calibration this is usually
    acceptable: the emulator is evaluated at many θ_physics proposals, and
    the hyperparameter uncertainty is small relative to the parameter-space
    uncertainty being inferred. Re-evaluate if the emulator is used for
    decisions sensitive to the tails of the predictive distribution (e.g.
    rare-event probabilities, tight design margins).
    """

    def __init__(self, n_pc=0.999, M=200, key=None, init_strategy="maxmin"):
        """
        Initialize the emulator.

        Parameters
        ----------
        n_pc : float or int
            Number of PCA components: float in (0, 1) for the fraction of the
            explained variance, int for a fixed number (default 0.999).
        M : int
            Number of inducing points (default 200). Rule of thumb: M ≈ N/5
            for D ≤ 10, M ≈ N/3 for D = 20-35. Increasing M improves the
            accuracy at O(M²) cost in memory and O(M³) per Cholesky
            decomposition.
        key : jax.random.PRNGKey or None
            Random key for reproducibility. Independent subkeys are derived
            from it for the inducing-point initialization, the seeds of
            numpy/scikit-learn/scipy and the mini-batches. None uses
            ``PRNGKey(0)`` (default None).
        init_strategy : str
            Inducing-point initialization strategy: 'maxmin' (default, best
            coverage in moderate D), 'kmeans', 'kmeans_pp', 'random' or
            'sobol'.
        """
        # n_pc is the argument, n_pc_ the number of PCs after fit()
        self.n_pc = n_pc
        self.n_pc_ = None
        self.M = M
        self.key = jax.random.PRNGKey(0) if key is None else key
        self.init_strategy = init_strategy
        self.training_history_ = None
        self.trunc_cov_yn_ = (
            None  # set in fit(); exact PCA truncation covariance in Yn space
        )
        # set in fit(); mean obs-noise covariance (n_pc, n_pc) in standardized
        # PC space
        self.mean_obs_cov_pc_ = None
        # set in fit(); truncation covariance without the observation noise
        self.trunc_cov_signal_yn_ = None

    # -------------------------
    # Kernel
    # -------------------------
    def kernel(self, x1, x2, p):
        """
        Sum of the RBF and Matern-3/2 kernels with the current parameters.

        The lengthscales and variances are obtained from the unconstrained
        parameters with a softplus and clipped to [1e-3, 1e4] and
        [1e-7, 10], respectively.

        Parameters
        ----------
        x1 : array (N1, D)
            First set of (normalized) input points.
        x2 : array (N2, D)
            Second set of (normalized) input points.
        p : dict
            Parameters with the keys 'log_lengthscale', 'log_var_rbf' and
            'log_var_mat'.

        Returns
        -------
        array (N1, N2)
            Kernel matrix.
        """
        ls = jax.nn.softplus(p["log_lengthscale"]) + 1e-6
        ls = jnp.clip(ls, 1e-3, 1e4)
        vr = jax.nn.softplus(p["log_var_rbf"]) + 1e-6
        vm = jax.nn.softplus(p["log_var_mat"]) + 1e-6
        vr = jnp.clip(vr, 1e-7, 10.0)
        vm = jnp.clip(vm, 1e-7, 10.0)
        return rbf_kernel(x1, x2, ls, vr) + matern32_kernel(x1, x2, ls, vm)

    def kernel_diag(self, x, p):
        """
        Diagonal of the kernel matrix ``kernel(x, x, p)``.

        Parameters
        ----------
        x : array (N, D)
            Input points.
        p : dict
            Parameters with the keys 'log_var_rbf' and 'log_var_mat'.

        Returns
        -------
        array (N,)
            Sum of the RBF and Matern-3/2 variances at every point.
        """
        vr = jax.nn.softplus(p["log_var_rbf"]) + 1e-6
        vm = jax.nn.softplus(p["log_var_mat"]) + 1e-6
        vr = jnp.clip(vr, 1e-7, 10.0)
        vm = jnp.clip(vm, 1e-7, 10.0)
        return (vr + vm) * jnp.ones(x.shape[0])

    # -------------------------
    # Build variational L (Cholesky factor)
    # -------------------------
    def _build_cholesky_factor(self, L_unconstrained):
        """
        Build lower triangular Cholesky factors from unconstrained parameters.

        The upper triangle is discarded and a softplus (plus 1e-6) is applied
        to the diagonal, so that the factors have a positive diagonal.

        Parameters
        ----------
        L_unconstrained : array (n_pc, M, M)
            Unconstrained parameters.

        Returns
        -------
        array (n_pc, M, M)
            Lower triangular matrices with a positive diagonal.
        """
        L = jnp.tril(L_unconstrained)
        raw_diag = jnp.diagonal(L, axis1=-2, axis2=-1)
        pos_diag = jax.nn.softplus(raw_diag) + 1e-6
        diag_raw = jax.vmap(jnp.diag)(raw_diag)
        diag_pos = jax.vmap(jnp.diag)(pos_diag)
        return L - diag_raw + diag_pos

    # -------------------------
    # Inducing point initialisation strategies
    # -------------------------
    def _numpy_seed(self):
        """
        Integer seed derived from self.key for numpy/sklearn/scipy.

        Ensemble members with different keys thus get different
        initializations.
        """
        return int(jax.random.randint(_subkey(self.key, "numpy"), (), 0, 2**31 - 1))

    def _init_inducing_maxmin(self, X):
        """Select M training points by greedy max-min distance selection."""
        N, _ = X.shape
        idx = jax.random.randint(_subkey(self.key, "init"), (), 0, N)
        Z = X[idx : idx + 1]
        min_dists = jnp.sum((X - Z[0]) ** 2, axis=-1)
        for _ in range(1, self.M):
            idx = jnp.argmax(min_dists)
            Z = jnp.vstack([Z, X[idx]])
            dists_to_new = jnp.sum((X - X[idx]) ** 2, axis=-1)
            min_dists = jnp.minimum(min_dists, dists_to_new)
        return Z

    def _init_inducing_kmeans(self, X):
        """Use k-means cluster centers (10 inits) with small random jitter."""
        kmeans = KMeans(self.M, n_init=10, random_state=self._numpy_seed()).fit(
            np.array(X)
        )
        Z = jnp.array(kmeans.cluster_centers_)
        Z += 0.01 * jax.random.normal(_subkey(self.key, "init"), Z.shape)
        return Z

    def _init_inducing_kmeans_pp(self, X):
        """Use k-means cluster centers with a single k-means++ init."""
        kmeans = KMeans(
            self.M, init="k-means++", n_init=1, random_state=self._numpy_seed()
        ).fit(np.array(X))
        return jnp.array(kmeans.cluster_centers_)

    def _init_inducing_random(self, X):
        """Select M training points at random without replacement."""
        N = X.shape[0]
        idx = jax.random.choice(_subkey(self.key, "init"), N, (self.M,), replace=False)
        return X[idx]

    def _init_inducing_sobol(self, X):
        """Place M scrambled Sobol points in the bounding box of X."""
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=X.shape[1], scramble=True, seed=self._numpy_seed())
        sample = sampler.random(self.M)
        X_min = jnp.min(X, axis=0)
        X_max = jnp.max(X, axis=0)
        return jnp.array(X_min + sample * (X_max - X_min))

    # -------------------------
    # Fit
    # -------------------------
    @_restore_state_on_error
    def fit(
        self,
        X,
        Y,
        Y_err=None,
        steps=25000,
        batch_size=None,
        kernel_lr=1e-3,
        variational_lr=1e-3,
        inducing_lr=3e-4,
        _fixed_pca_state=None,
        print_every=200,
        jitter_init=1e-5,
        jitter_max=1e-1,
        verbose=True,
        early_stopping=False,
        patience=20,
        es_rel_tol=1e-4,
        ema_alpha=0.95,
        auto_lr_backoff=True,
        lr_backoff_factor=0.3,
        max_lr_backoff_retries=3,
        nan_patience=10,
    ):
        """
        Fit the emulator to training data.

        The inputs and outputs are standardized, the outputs are reduced with
        PCA, and the kernel, variational and inducing-point parameters are
        optimized with ADAM on the ELBO. The parameters with the best ELBO
        (or the best EMA of the ELBO with mini-batches) are kept.

        Parameters
        ----------
        X : array (N, D)
            Input training data.
        Y : array (N, P)
            Output training data.
        Y_err : array (N, P) or (N, P, P) or None
            Per-training-point observation uncertainty in original Y units.

            **(N, P)**: independent (diagonal) errors. ``Y_err[n, j]`` is the
            standard deviation of output j at training point n. Projected to
            standardized PC space as a diagonal covariance:
            ``Var_pc[n, i] = sum_j (W[i,j] / (pc_std[i]*Ys[j]))^2 * Y_err[n,j]^2``.
            Entries larger than 1e5 times the standard deviation of the
            training data are capped.

            **(N, P, P)**: correlated errors. ``Y_err[n]`` is the full
            ``(P, P)`` observation covariance matrix at training point n.
            Full propagation: ``Cov_pc[n] = W_scaled @ Y_err[n] @ W_scaled^T``,
            where ``W_scaled[i,j] = W[i,j] / (pc_std[i]*Ys[j])``.

            The diagonal of ``Cov_pc[n]`` is used in the per-PC ELBO
            likelihood (capped at 1e10 in standardized PC units). The mean
            over n of the full ``Cov_pc`` matrix is stored and used in
            ``predict()`` for the back-projection to the output-space
            covariance (include_obs_noise). The mean noise covariance is also
            removed from the truncation covariance for predictions without
            noise (see `predict`). When None, no extra observation noise is
            added and the truncation covariance is used unchanged.
            Default None.
        steps : int
            Maximum number of ADAM steps (default 25000).
        batch_size : int or None
            Mini-batch size for the stochastic ELBO. None uses the full
            dataset (default). The likelihood term is scaled by N/batch_size,
            so that the ELBO remains comparable across batch sizes.
            Recommended: 256-1024 for large N.
        kernel_lr : float
            Learning rate for the kernel parameters and the nugget
            (default 1e-3).
        variational_lr : float
            Learning rate for the variational parameters (default 1e-3).
        inducing_lr : float
            Learning rate for the inducing points (default 3e-4).
        _fixed_pca_state : dict or None
            Internal parameter used by PCASparseGPEnsemble. When provided,
            the normalization and PCA fit are skipped and the pre-computed
            shared PCA state is used instead (default None).
        print_every : int
            Log the progress every print_every steps (default 200).
        jitter_init : float
            Initial diagonal jitter for the stability of Kzz (default 1e-5).
            Increased automatically by factors of 10 if the Cholesky
            decomposition fails at startup or the loss is NaN during
            training.
        jitter_max : float
            Maximum allowed jitter (default 1e-1). At startup, a RuntimeError
            is raised if it is exceeded; during training, the jitter is
            capped at this value.
        verbose : bool
            Log the training progress (default True). Warnings, e.g. of the
            NaN recovery, are always logged.
        early_stopping : bool
            Enable EMA-based early stopping (default False). The EMA gain
            over a sliding window of about 1/(1 - ema_alpha) steps (at least
            20) is checked at every step. Training stops when it stays below
            ``es_rel_tol`` for ``patience`` consecutive steps.
        patience : int
            Consecutive steps without meaningful EMA improvement before
            stopping (default 20).
        es_rel_tol : float
            Minimum relative EMA improvement over the window required to
            reset the patience (default 1e-4).
        ema_alpha : float
            EMA smoothing factor in [0, 1). Higher means heavier smoothing
            (default 0.95). The effective window is about 1/(1 - ema_alpha)
            steps. With mini-batches (batch_size < N), the parameters with
            the best EMA of the ELBO are returned, with the full batch the
            ones with the best ELBO.
        auto_lr_backoff : bool
            Automatically reduce the learning rates and retry when repeated
            NaN losses occur, instead of failing immediately (default True).
        lr_backoff_factor : float
            Multiplicative factor in (0, 1) applied to all learning rates on
            each retry (default 0.3).
        max_lr_backoff_retries : int
            Maximum number of learning-rate backoff retries after NaN bursts
            (default 3).
        nan_patience : int
            After a non-finite ELBO, the jitter is increased (up to
            jitter_max), and the parameters and the optimizer state are reset
            to the best parameters so far (or the initial ones). After more
            than `nan_patience` consecutive non-finite steps, the learning
            rates are reduced (if auto_lr_backoff) or a RuntimeError is
            raised (default 10).

        Returns
        -------
        PCASparseGPEmulator
            The fitted emulator (self). The training history is stored as
            the dict ``self.training_history_``:

            - 'elbos': the finite ELBO values (mini-batch estimates with
              batch_size < N), 'steps': their iteration numbers, 'n_steps':
              their number. The constant -0.5 * N * n_pc * log(2 pi) of the
              Gaussian log-likelihood is left out, and the ELBO refers to the
              standardized PCs.
            - 'best_step', 'best_score': iteration and score of the returned
              parameters (the ELBO, or its EMA with mini-batches), None if no
              step had a finite ELBO.
            - 'converged': True if the early stopping stopped the training.
            - 'jitter': the jitter of the returned parameters, which predict()
              uses, 'jitter_final': the jitter at the end of the training
              (larger after a NaN recovery), 'lr_backoff_retries': the number
              of learning-rate reductions, 'kernel_lr_final',
              'variational_lr_final', 'inducing_lr_final': the final learning
              rates.

        Raises
        ------
        ValueError
            If Y_err has an invalid shape or negative diagonal covariance
            entries, if M exceeds the number of training points, if
            init_strategy is unknown, or if n_pc, steps, batch_size,
            lr_backoff_factor, max_lr_backoff_retries, nan_patience,
            ema_alpha, jitter_init or print_every are out of range. The
            attributes of the emulator are restored if fit() fails, so that a
            failed refit keeps the previously trained emulator.
        RuntimeError
            If the Cholesky decomposition of Kzz fails at jitter_max at
            startup, or if the loss stays NaN after all jitter increases and
            learning-rate backoff retries.
        """
        if not (0.0 < lr_backoff_factor < 1.0):
            raise ValueError(
                f"lr_backoff_factor must be in (0, 1), got {lr_backoff_factor}"
            )
        if max_lr_backoff_retries < 0:
            raise ValueError("max_lr_backoff_retries must be >= 0")
        if nan_patience < 1:
            raise ValueError("nan_patience must be >= 1")
        if not (0.0 <= ema_alpha < 1.0):
            raise ValueError(f"ema_alpha must be in [0, 1), got {ema_alpha}")
        if not (0.0 < jitter_init <= jitter_max):
            raise ValueError(
                f"jitter_init must be in (0, jitter_max], got jitter_init="
                f"{jitter_init} and jitter_max={jitter_max}"
            )
        if print_every < 1:
            raise ValueError(f"print_every must be >= 1, got {print_every}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 or None, got {batch_size}")
        check_npc(self.n_pc)
        if Y_err is not None:
            _check_y_err(Y_err, Y)
        if self.init_strategy not in _INIT_STRATEGIES:
            raise ValueError(
                f"Unknown init_strategy {self.init_strategy!r}, use one of "
                f"{_INIT_STRATEGIES}"
            )
        # checked before the PCA and the other fitted attributes are replaced
        if self.M > X.shape[0]:
            raise ValueError(
                f"M={self.M} inducing points cannot exceed N={X.shape[0]} training "
                f"points. Reduce M or provide more training data."
            )
        if _fixed_pca_state is None:
            self.Xm_, self.Xs_ = X.mean(0), X.std(0) + 1e-8
            Xn = (X - self.Xm_) / self.Xs_

            self.Ym_, self.Ys_ = Y.mean(0), Y.std(0) + 1e-8
            Yn = (Y - self.Ym_) / self.Ys_

            logger.debug(
                f"Output means in [{self.Ym_.min():.3g}, {self.Ym_.max():.3g}], "
                f"standard deviations in [{self.Ys_.min():.3g}, {self.Ys_.max():.3g}]"
            )

            self.pca_, Yp = _fit_pca(Yn, self.n_pc)
            self.n_pc_ = self.pca_.n_components_
            explained_var = np.sum(self.pca_.explained_variance_ratio_)

            if verbose:
                logger.info(
                    f"Using {self.n_pc_} PCs, which explain {explained_var:.5f} of "
                    "the variance"
                )

            self.pc_mean_ = jnp.mean(Yp, axis=0)
            self.pc_std_ = jnp.std(Yp, axis=0) + 1e-8
            Yp = jnp.array((Yp - np.array(self.pc_mean_)) / np.array(self.pc_std_))

            Yn_np = np.array(Yn)
            P = Yn_np.shape[1]
            W_ret = self.pca_.components_
            lam_ret = self.pca_.explained_variance_
            if self.n_pc_ < P:
                Sigma_data = np.cov(Yn_np.T)
                Sigma_ret = (W_ret * lam_ret[:, None]).T @ W_ret
                Sigma_trunc = Sigma_data - Sigma_ret
                vals, vecs = np.linalg.eigh(Sigma_trunc)
                vals = np.maximum(vals, 0.0)
                self.trunc_cov_yn_ = jnp.array(vecs @ (vals[:, None] * vecs.T))
                ppca_approx = float(self.pca_.noise_variance_) * (P - self.n_pc_)
                logger.debug(
                    f"Trace of the truncation covariance: {float(np.sum(vals)):.4g} "
                    f"(PPCA approximation: {ppca_approx:.4g})"
                )
            else:
                self.trunc_cov_yn_ = jnp.zeros((P, P))
                logger.debug("No truncation covariance, all PCs are retained")
        else:
            self.Xm_ = _fixed_pca_state["Xm"]
            self.Xs_ = _fixed_pca_state["Xs"]
            self.Ym_ = _fixed_pca_state["Ym"]
            self.Ys_ = _fixed_pca_state["Ys"]
            self.pca_ = _fixed_pca_state["pca"]
            self.n_pc_ = _fixed_pca_state["n_pc"]
            self.pc_mean_ = _fixed_pca_state["pc_mean"]
            self.pc_std_ = _fixed_pca_state["pc_std"]
            self.trunc_cov_yn_ = _fixed_pca_state["trunc_cov_yn"]

            Xn = (X - self.Xm_) / self.Xs_
            Yn = (Y - self.Ym_) / self.Ys_
            Yp_r = self.pca_.transform(np.array(Yn))
            Yp = jnp.array((Yp_r - np.array(self.pc_mean_)) / np.array(self.pc_std_))

        N_full = Xn.shape[0]

        _W_np = self.pca_.components_
        _pc_std_np = np.array(self.pc_std_)
        _Ys_np = np.array(self.Ys_)
        _W_scaled = _W_np / (_pc_std_np[:, None] * _Ys_np[None, :])

        # Observation-noise variances in standardized PC units (data variance
        # ~1) are capped at this value. Larger errors carry no information, but
        # can overflow and make the ELBO non-finite.
        max_obs_var = 1e10

        if Y_err is not None:
            _yerr = np.asarray(Y_err, dtype=float)
            if _yerr.ndim == 2:
                _yerr_max = np.sqrt(max_obs_var) * _Ys_np
                n_capped = int(np.sum(_yerr > _yerr_max[None, :]))
                if n_capped > 0:
                    logger.warning(
                        f"Capping {n_capped} entries of Y_err that are larger than "
                        "1e5 times the standard deviation of the training data"
                    )
                    _yerr = np.minimum(_yerr, _yerr_max[None, :])
                obs_var_full = jnp.array(
                    np.minimum((_yerr**2) @ (_W_scaled**2).T, max_obs_var)
                )
                _mean_C_Y = np.diag(np.mean(_yerr**2, axis=0))
                _mean_obs_cov_pc = _W_scaled @ _mean_C_Y @ _W_scaled.T
                logger.debug(
                    "RMS of Y_err: "
                    f"{float(np.sqrt(np.mean(_yerr**2))):.4g} (original Y units)"
                )
            else:
                # (N, P, P) covariance matrices, checked by _check_y_err
                obs_var_full = jnp.array(
                    np.clip(
                        np.einsum("ij,njk,ik->ni", _W_scaled, _yerr, _W_scaled),
                        0.0,
                        max_obs_var,
                    )
                )
                _mean_C_Y = np.mean(_yerr, axis=0)
                _mean_obs_cov_pc = _W_scaled @ _mean_C_Y @ _W_scaled.T
                _mean_obs_cov_pc = 0.5 * (_mean_obs_cov_pc + _mean_obs_cov_pc.T)
                _evals, _evecs = np.linalg.eigh(_mean_obs_cov_pc)
                _mean_obs_cov_pc = _evecs @ (
                    np.maximum(_evals, 0.0)[:, None] * _evecs.T
                )
            self.mean_obs_cov_pc_ = jnp.array(_mean_obs_cov_pc)
        else:
            obs_var_full = jnp.zeros((N_full, self.n_pc_))
            self.mean_obs_cov_pc_ = None

        # The truncation covariance also contains the observation noise of the
        # training data in the discarded PCA directions. Its signal part, the
        # truncation covariance without this noise, is used for predictions
        # of the model function (include_noise=False).
        self.trunc_cov_signal_yn_ = self.trunc_cov_yn_
        if Y_err is not None:
            _noise_yn = _mean_C_Y / np.outer(_Ys_np, _Ys_np)
            self.trunc_cov_signal_yn_ = jnp.array(
                truncation_signal(np.array(self.trunc_cov_yn_), _noise_yn)
            )

        B = min(batch_size, N_full) if batch_size is not None else N_full

        if verbose:
            batches = f"mini-batches of {B}" if B < N_full else "the full batch"
            logger.info(
                f"Training the SVGPs of {self.n_pc_} PCs with {N_full} training "
                f"points, {self.M} inducing points and {batches} for at most "
                f"{steps} steps ..."
            )

        if self.init_strategy == "maxmin":
            Z = self._init_inducing_maxmin(Xn)
        elif self.init_strategy == "kmeans":
            Z = self._init_inducing_kmeans(Xn)
        elif self.init_strategy == "kmeans_pp":
            Z = self._init_inducing_kmeans_pp(Xn)
        elif self.init_strategy == "random":
            Z = self._init_inducing_random(Xn)
        elif self.init_strategy == "sobol":
            Z = self._init_inducing_sobol(Xn)
        else:
            raise AssertionError(self.init_strategy)

        self.params_ = {
            "Z": Z,
            "log_lengthscale": jnp.full(
                (X.shape[1],), float(np.log(np.expm1(float(np.sqrt(X.shape[1])))))
            ),
            "log_var_rbf": jnp.array(0.0),
            "log_var_mat": jnp.array(-0.5),
            "log_noise": jnp.full((self.n_pc_,), -2.0),
            "m": jnp.zeros((self.n_pc_, self.M)),
            "L_unconstrained": jnp.zeros((self.n_pc_, self.M, self.M)),
        }

        jitter = jitter_init
        while True:
            Kzz_test = self.kernel(Z, Z, self.params_) + jitter * jnp.eye(self.M)
            Lz_test = jnp.linalg.cholesky(Kzz_test)
            if not bool(jnp.any(~jnp.isfinite(Lz_test))):
                break
            if jitter >= jitter_max:
                raise RuntimeError(
                    f"Kzz Cholesky failed up to jitter={jitter:.1e} (jitter_max). "
                    f"Try reducing M or using a different init_strategy."
                )
            jitter = min(jitter * 10.0, jitter_max)
        if jitter > jitter_init:
            logger.warning(
                f"Increased the jitter from {jitter_init:.1e} to {jitter:.1e} for a "
                "stable Cholesky decomposition of Kzz"
            )

        self.jitter_ = jitter
        self.n_train_ = N_full

        def elbo_fn(p, Xb, Yb, obs_noise_b, jitter_arr):
            Z = p["Z"]
            noise = jax.nn.softplus(p["log_noise"]) + 1e-6
            Kzz = self.kernel(Z, Z, p) + jitter_arr * jnp.eye(self.M)
            Lz = jnp.linalg.cholesky(Kzz)
            Kxz = self.kernel(Xb, Z, p)
            A_half = jax.scipy.linalg.solve_triangular(Lz, Kxz.T, lower=True)
            m = p["m"]
            L = self._build_cholesky_factor(p["L_unconstrained"])
            base_var = self.kernel_diag(Xb, p)
            qdiag = jnp.sum(A_half**2, axis=0)

            def pc_term(i):
                y_i = Yb[:, i]
                m_i = m[i]
                L_i = L[i]
                noise_i = noise[i]
                f_mean_i = A_half.T @ m_i
                B_i = A_half.T @ L_i
                f_var_i = jnp.clip(
                    base_var - qdiag + jnp.sum(B_i**2, axis=1), 1e-7, None
                )
                obs_var_i = noise_i + obs_noise_b[:, i]
                ll_i = -0.5 * jnp.sum(((y_i - f_mean_i) ** 2 + f_var_i) / obs_var_i)
                ll_i -= 0.5 * jnp.sum(jnp.log(obs_var_i))
                L_i_diag = jnp.clip(jnp.diag(L_i), 1e-8, None)
                kl_i = 0.5 * (
                    jnp.sum(m_i**2)
                    + jnp.sum(L_i**2)
                    - self.M
                    - 2.0 * jnp.sum(jnp.log(L_i_diag))
                )
                return ll_i, kl_i

            ll_per_pc, kl_per_pc = jax.vmap(pc_term)(jnp.arange(self.n_pc_))
            total_elbo = (N_full / Xb.shape[0]) * jnp.sum(ll_per_pc) - jnp.sum(
                kl_per_pc
            )
            return total_elbo

        param_labels = {
            "Z": "inducing",
            "log_lengthscale": "kernel",
            "log_var_rbf": "kernel",
            "log_var_mat": "kernel",
            "log_noise": "kernel",
            "m": "variational",
            "L_unconstrained": "variational",
        }

        current_kernel_lr = float(kernel_lr)
        current_variational_lr = float(variational_lr)
        current_inducing_lr = float(inducing_lr)
        lr_backoff_count = 0

        def make_optimizer_and_step(k_lr, v_lr, i_lr):
            tx_local = optax.multi_transform(
                {
                    "kernel": optax.adam(k_lr),
                    "variational": optax.adam(v_lr),
                    "inducing": optax.adam(i_lr),
                },
                param_labels,
            )

            @jax.jit
            def step_local(p_local, opt_state_local, Xb, Yb, obs_noise_b, jitter_arr):
                loss_val, grads = jax.value_and_grad(
                    lambda params: -elbo_fn(params, Xb, Yb, obs_noise_b, jitter_arr)
                )(p_local)
                updates, new_opt_state_local = tx_local.update(grads, opt_state_local)
                new_p_local = optax.apply_updates(p_local, updates)
                return new_p_local, new_opt_state_local, -loss_val

            return tx_local, step_local

        tx, step = make_optimizer_and_step(
            current_kernel_lr, current_variational_lr, current_inducing_lr
        )
        opt_state = tx.init(self.params_)

        p = self.params_
        p_init = {k: np.array(v) for k, v in p.items()}
        elbos = []
        # best_score is the ELBO, or its EMA for mini-batches; step_ids are
        # the iteration numbers of the finite ELBO values
        best_score = -np.inf
        best_step = None
        step_ids = []
        best_params = None
        # jitter with which the best parameters were trained; the jitter can
        # be increased later by the NaN recovery, and predict() must use the
        # jitter of the returned parameters
        best_jitter = jitter
        converged = False
        nan_count = 0
        ema = None
        es_patience_count = 0
        es_check_interval = max(20, int(round(1.0 / (1.0 - ema_alpha))))
        ema_history = []
        # mini-batches, independent of the random initialization
        key = _subkey(self.key, "train")

        if early_stopping:
            logger.debug(
                f"Early stopping: patience={patience}, es_rel_tol={es_rel_tol:.1e}, "
                f"ema_alpha={ema_alpha} (window of {es_check_interval} steps)"
            )
        logger.debug(
            f"NaN recovery: nan_patience={nan_patience}, jitter_max={jitter_max:.1e}"
            + (
                f", max_lr_backoff_retries={max_lr_backoff_retries}, "
                f"lr_backoff_factor={lr_backoff_factor:.3f}"
                if auto_lr_backoff
                else ", no learning-rate backoff"
            )
        )

        n_iterations = 0
        for i in range(steps):
            n_iterations = i + 1
            key, subkey = jax.random.split(key)
            if B < N_full:
                idx = jax.random.choice(subkey, N_full, (B,), replace=False)
                Xb, Yb, obs_noise_b = Xn[idx], Yp[idx], obs_var_full[idx]
            else:
                Xb, Yb, obs_noise_b = Xn, Yp, obs_var_full

            # step() evaluates the ELBO at p_eval and returns the updated
            # parameters, so elbo_val belongs to p_eval, not to the new p
            p_eval = p
            p, opt_state, elbo_val = step(
                p_eval, opt_state, Xb, Yb, obs_noise_b, jnp.array(jitter)
            )

            if not jnp.isfinite(elbo_val):
                new_jitter = min(jitter * 10.0, jitter_max)
                if new_jitter > jitter:
                    jitter_str = f"the jitter {jitter:.1e} -> {new_jitter:.1e}"
                else:
                    jitter_str = f"the jitter already at jitter_max = {jitter:.1e}"
                logger.warning(
                    f"Non-finite ELBO at step {i}, restarting from the best "
                    f"parameters with {jitter_str}"
                )
                jitter = new_jitter
                # restart from the best parameters with a finite ELBO, or
                # from the initial parameters if there are none yet
                restart_params = best_params if best_params is not None else p_init
                p = {k: jnp.array(v) for k, v in restart_params.items()}
                opt_state = tx.init(p)
                nan_count += 1
                if nan_count > nan_patience:
                    if auto_lr_backoff and lr_backoff_count < max_lr_backoff_retries:
                        lr_backoff_count += 1
                        current_kernel_lr *= lr_backoff_factor
                        current_variational_lr *= lr_backoff_factor
                        current_inducing_lr *= lr_backoff_factor
                        logger.warning(
                            f"NaN recovery attempt {lr_backoff_count}/"
                            f"{max_lr_backoff_retries}: lowering the learning rates "
                            f"to kernel={current_kernel_lr:.3e}, "
                            f"variational={current_variational_lr:.3e}, "
                            f"inducing={current_inducing_lr:.3e}"
                        )
                        tx, step = make_optimizer_and_step(
                            current_kernel_lr,
                            current_variational_lr,
                            current_inducing_lr,
                        )
                        opt_state = tx.init(p)
                        nan_count = 0
                        continue
                    raise RuntimeError(
                        f"Non-finite ELBO in {nan_count} consecutive steps "
                        f"(jitter={jitter:.1e}) after {lr_backoff_count} "
                        f"LR backoff retries. "
                        f"Current LRs: kernel={current_kernel_lr:.3e}, "
                        f"variational={current_variational_lr:.3e}, "
                        f"inducing={current_inducing_lr:.3e}. "
                        f"Try smaller initial learning rates or larger "
                        f"jitter_init/jitter_max."
                    )
                continue

            elbo_val_f = float(elbo_val)
            elbos.append(elbo_val_f)
            step_ids.append(i)
            # nan_patience counts consecutive non-finite steps
            nan_count = 0

            if ema is None:
                ema = elbo_val_f
            else:
                ema = ema_alpha * ema + (1.0 - ema_alpha) * elbo_val_f
            ema_history.append(ema)

            # With mini-batches the ELBO of a single step is noisy, and its
            # maximum would select the parameters of the luckiest batch.
            # The best parameters are therefore selected on the EMA. With the
            # full batch the ELBO is exact and used directly.
            score = ema if B < N_full else elbo_val_f
            if score > best_score:
                best_score = score
                best_step = i
                best_params = {k: np.array(v) for k, v in p_eval.items()}
                best_jitter = jitter

            if early_stopping and len(ema_history) > es_check_interval:
                prev_ema = ema_history[-es_check_interval - 1]
                scale = max(1.0, abs(prev_ema), abs(ema))
                rel_gain = (ema - prev_ema) / scale
                if rel_gain < es_rel_tol:
                    es_patience_count += 1
                else:
                    es_patience_count = 0
                if es_patience_count >= patience:
                    converged = True
                    if verbose:
                        logger.info(
                            f"Early stopping at step {i}: the relative EMA gain "
                            f"over {es_check_interval} steps stayed below "
                            f"{es_rel_tol:.1e} for {patience} steps (ELBO = "
                            f"{elbo_val_f:.3f}, EMA = {ema:.3f})"
                        )
                    break

            if verbose and (i % print_every == 0 or i == steps - 1):
                ema_str = f", EMA={ema:.3f}"
                es_str = (
                    f", pat={es_patience_count}/{patience}" if early_stopping else ""
                )
                logger.info(
                    f"Step {i:5d}/{steps}: ELBO = {elbo_val_f:10.3f}{ema_str}{es_str}"
                )

        if best_params is not None:
            self.params_ = {k: jnp.array(v) for k, v in best_params.items()}
            self.jitter_ = best_jitter
        else:
            logger.warning(
                f"No step of the {n_iterations} training steps had a finite ELBO, "
                "returning the last parameters"
            )
            self.params_ = p
            self.jitter_ = jitter

        actual_steps = len(elbos)
        self.training_history_ = {
            "elbos": elbos,
            "steps": step_ids,
            "converged": converged,
            "n_steps": actual_steps,
            "best_step": best_step,
            "best_score": best_score if best_step is not None else None,
            "jitter": self.jitter_,
            "jitter_final": jitter,
            "lr_backoff_retries": lr_backoff_count,
            "kernel_lr_final": current_kernel_lr,
            "variational_lr_final": current_variational_lr,
            "inducing_lr_final": current_inducing_lr,
        }

        if verbose and best_step is not None:
            elbo_str = "EMA of the ELBO" if B < N_full else "ELBO"
            logger.info(
                f"Training finished after {n_iterations} steps ({actual_steps} with "
                f"a finite ELBO, converged: {converged}): best {elbo_str} = "
                f"{best_score:.3f} at step {best_step}, jitter = {self.jitter_:.1e}"
            )

        return self

    # -------------------------
    # Predict
    # -------------------------
    def predict(
        self,
        X_star,
        include_noise=False,
        include_truncation=True,
        include_pca_sampling=False,
        include_obs_noise=False,
        return_var_decomposition=False,
    ):
        """
        Predict at new inputs with full uncertainty quantification.

        Only the covariance between the outputs at the same test point is
        computed, not the joint covariance between different test points.

        Parameters
        ----------
        X_star : array (N_test, D)
            Test inputs.
        include_noise : bool
            Add the learned per-PC nugget (log_noise) to the predictive
            variance (default False). The nugget is fitted in addition to the
            known observation noise Y_err, so it describes the variance of
            the training data beyond Y_err: unmodelled noise and emulation
            error that the inducing points cannot represent. Without it, the
            prediction is the uncertainty of the emulated model function, as
            needed for the comparison with experimental data. With it (and
            include_obs_noise for the known noise), the prediction is the
            uncertainty of a new noisy simulation. If the nugget is large
            compared with the GP variance, the emulator does not describe the
            training data well, and include_noise=True gives more
            conservative intervals.
        include_truncation : bool
            Add the exact PCA truncation uncertainty (default True): the
            covariance contribution of all discarded PCA components, computed
            in fit() as ``Sigma_trunc = Sigma_data - W_ret^T diag(Lambda_ret)
            W_ret``. This is exact under the linear PCA model (no PPCA
            isotropy assumption). With Y_err, the observation noise of the
            training data in the discarded directions is removed from it,
            unless include_obs_noise is True.
        include_pca_sampling : bool
            Add the finite-training-data uncertainty of the PCA mean
            estimate, ``Var(pc_mean_i) = pc_std_i^2 / n_train_`` per component
            (default False).
        include_obs_noise : bool
            Include the observation/statistical uncertainty propagated from
            Y_err (default False). For calibration against experimental means
            this should typically be False, because the experimental
            uncertainties are already handled in the likelihood. Set True
            when predicting noisy finite-statistics observables.
        return_var_decomposition : bool
            If True, also return a dict of the individual covariance
            contributions (default False).

        Returns
        -------
        Y_pred : numpy.ndarray (N_test, P)
            Predictive mean in original Y units.
        full_cov : numpy.ndarray (N_test, P, P)
            Predictive covariance of the outputs at each test point.
        var_decomp : dict
            Only returned if return_var_decomposition=True. Keys:

            - 'gp_posterior' (N_test, P, P): GP posterior covariance.
            - 'nugget' (1, P, P): learned log_noise term (zeros if
              include_noise=False).
            - 'obs_noise' (1, P, P): Y_err noise projected to output space
              (zeros if include_obs_noise=False or Y_err was not provided).
            - 'pca_truncation' (1, P, P): PCA truncation covariance, without
              the observation noise of the training data unless
              include_obs_noise is True (zeros if include_truncation=False).
            - 'pca_sampling' (1, P, P): PCA sampling covariance (zeros if
              include_pca_sampling=False).

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if not hasattr(self, "params_"):
            raise RuntimeError(
                "Call fit() before predict(). The emulator has not been trained yet."
            )
        Xn = (X_star - self.Xm_) / self.Xs_
        p = self.params_
        Z = p["Z"]

        Kzz = self.kernel(Z, Z, p) + self.jitter_ * jnp.eye(self.M)
        Lz = jnp.linalg.cholesky(Kzz)
        Kxz = self.kernel(Xn, Z, p)

        # Same whitened SVGP formulation as in elbo_fn()
        A_half = jax.scipy.linalg.solve_triangular(Lz, Kxz.T, lower=True)  # (M, N_test)
        qdiag = jnp.sum(A_half**2, axis=0)  # (N_test,) = diag(Kxz Kzz^{-1} Kxz^T)
        # Returns per-test-point output-output covariance (N_test, P, P) — sufficient
        # for single-proposal MCMC / Bayesian calibration.  Does NOT compute the joint
        # covariance Cov(f(x_a), f(x_b)) for a≠b (needed for active learning / BALD).

        m = p["m"]
        L = self._build_cholesky_factor(p["L_unconstrained"])
        base_var = self.kernel_diag(Xn, p)

        def pc_predict(i):
            m_i = m[i]
            L_i = L[i]
            mean_i = A_half.T @ m_i
            B_i = A_half.T @ L_i
            var_i = base_var - qdiag + jnp.sum(B_i**2, axis=1)
            return mean_i, jnp.clip(var_i, 1e-7, None)

        means_pc, vars_pc = jax.vmap(pc_predict)(jnp.arange(self.n_pc_))
        means_pc = means_pc.T  # (N_test, n_pc), standardized PC space
        vars_gp = vars_pc.T  # (N_test, n_pc), GP posterior variance only
        # Accumulate variance in standardized PC space
        vars_total = vars_gp

        # 1. Learned nugget: variance of the training data beyond Y_err, only
        #    included with include_noise (see the docstring).
        noise = jax.nn.softplus(p["log_noise"]) + 1e-6  # (n_pc,)
        if include_noise:
            vars_total = vars_total + noise[None, :]

        # 2. Finite-data PCA sampling: standard error of pc_mean estimate
        #    Var(pc_mean_i) = pc_std_i^2 / n_train_, i.e. 1/n_train_ in
        #    standardized space
        if include_pca_sampling:
            vars_total = vars_total + (1.0 / self.n_train_)

        # Undo PC normalization -> original PC space
        means_pc = means_pc * self.pc_std_ + self.pc_mean_
        vars_total_orig = vars_total * (self.pc_std_**2)  # (N_test, n_pc)
        vars_gp_orig = vars_gp * (self.pc_std_**2)  # for decomposition

        # Back-project from original PC space -> normalized output (Yn) space
        W = jnp.array(self.pca_.components_)  # (n_pc, P)
        Wt = W.T  # (P, n_pc)
        P = W.shape[1]
        full_cov = jnp.einsum(
            "pi,ni,qi->npq", Wt, vars_total_orig, Wt
        )  # (N_test, P, P)

        # 3. PCA truncation uncertainty — exact Sigma_trunc computed in fit().
        #
        # Model note on cross-PC posterior covariance:
        # The GP posterior is diagonal in PC space: each PC is an independent
        # mean-field variational GP, so Cov_q(f_i(x*), f_j(x*)) = 0 for i != j
        # (their cross-covariance = A_half^T Cov_q(v_i, v_j) A_half = 0).
        # The output-space covariance W^T diag(sigma^2_i) W is already fully dense
        # in P-dimensional output space — it is NOT diagonal there.
        # To capture non-zero cross-PC posterior correlations one would need
        # a Linear Model of Coregionalization (LMC) with a joint variational
        # distribution over all (n_pc * M) inducing variables simultaneously,
        # which is a fundamental architectural change to the ELBO.
        # (P, P), exact, PSD, set in fit(). Without the observation noise,
        # the observation noise of the training data in the discarded
        # directions is removed as well.
        if include_obs_noise:
            trunc_cov_yn = self.trunc_cov_yn_
        else:
            trunc_cov_yn = self.trunc_cov_signal_yn_
        if include_truncation:
            full_cov = full_cov + trunc_cov_yn[None, :, :]
        else:
            trunc_cov_yn = jnp.zeros((P, P))

        # Scale from Yn space to original Y space: Cov_Y[p,q] = Ys[p]*Cov_Yn[p,q]*Ys[q]
        Ys = jnp.array(self.Ys_)
        Ys_outer = jnp.outer(Ys, Ys)
        full_cov = full_cov * Ys_outer[None, :, :]

        # 4. Observation noise from Y_err — back-projected via the stored
        # (n_pc, n_pc) mean covariance. Handles both (N,P) and (N,P,P) inputs.
        if include_obs_noise and self.mean_obs_cov_pc_ is not None:
            obs_cov_pc_orig = self.mean_obs_cov_pc_ * jnp.outer(
                self.pc_std_, self.pc_std_
            )  # (n_pc, n_pc)
            obs_cov_yn = Wt @ obs_cov_pc_orig @ W  # (P, P)
            full_cov = full_cov + (obs_cov_yn * Ys_outer)[None, :, :]

        Y_pred = self.pca_.inverse_transform(np.array(means_pc))
        Y_pred = np.asarray(Y_pred * self.Ys_ + self.Ym_)
        full_cov = np.asarray(full_cov)

        if return_var_decomposition:
            # GP posterior covariance in Y space
            gp_cov = (
                jnp.einsum("pi,ni,qi->npq", Wt, vars_gp_orig, Wt) * Ys_outer[None, :, :]
            )

            # Nugget covariance in Y space (gated by include_noise)
            if include_noise:
                nugget_orig = noise * (self.pc_std_**2)  # (n_pc,)
                nugget_cov_yn = jnp.einsum("pi,i,qi->pq", Wt, nugget_orig, Wt)  # (P, P)
                nugget_cov = (nugget_cov_yn * Ys_outer)[None, :, :]
            else:
                nugget_cov = jnp.zeros((1, P, P))

            # Known observation noise from Y_err (optional, gated by include_obs_noise).
            # Uses the stored full (n_pc, n_pc) covariance — correct for both
            # diagonal (N,P) and full-covariance (N,P,P) Y_err inputs.
            if include_obs_noise and self.mean_obs_cov_pc_ is not None:
                obs_cov_pc_orig = self.mean_obs_cov_pc_ * jnp.outer(
                    self.pc_std_, self.pc_std_
                )
                obs_cov_yn = Wt @ obs_cov_pc_orig @ W
                obs_noise_cov = (obs_cov_yn * Ys_outer)[None, :, :]
            else:
                obs_noise_cov = jnp.zeros((1, P, P))

            # Truncation covariance in Y space (exact, not PPCA)
            if include_truncation:
                trunc_cov_y = (trunc_cov_yn * Ys_outer)[None, :, :]
            else:
                trunc_cov_y = jnp.zeros((1, P, P))

            # PCA sampling covariance in Y space
            if include_pca_sampling:
                pca_samp_pc = (self.pc_std_**2) / self.n_train_  # (n_pc,)
                pca_samp_cov_yn = jnp.einsum("pi,i,qi->pq", Wt, pca_samp_pc, Wt)
                pca_samp_cov = (pca_samp_cov_yn * Ys_outer)[None, :, :]
            else:
                pca_samp_cov = jnp.zeros((1, P, P))

            return (
                Y_pred,
                full_cov,
                {
                    "gp_posterior": gp_cov,
                    "nugget": nugget_cov,
                    "obs_noise": obs_noise_cov,
                    "pca_truncation": trunc_cov_y,
                    "pca_sampling": pca_samp_cov,
                },
            )

        return Y_pred, full_cov


# =============================================================================
# PCASparseGPEnsemble
# =============================================================================


class PCASparseGPEnsemble:
    """
    Ensemble of PCASparseGPEmulator for an estimate of the emulator
    uncertainty from the spread between the members.

    All members share one PCA basis, fitted on the full training data. Each
    member is trained with a different JAX random key, which gives
    different:

    - inducing-point initializations,
    - mini-batch orderings (stochastic ELBO),
    - ADAM optimizer trajectories / local optima,
    - bootstrap resamples of the training data (with ``bootstrap=True``).

    The predictions are the mean and covariance of the equal-weight mixture
    of the members (law of total variance)::

        E[Y | x*]   = (1/K) sum_k  mu_k(x*)
        Cov[Y | x*] = (1/K) sum_k  Sigma_k(x*)                  # within members
                    + (1/K) sum_k (mu_k - E[Y])(mu_k - E[Y])^T  # between members

    The within-members term is the mean of the individual predictive
    covariances (GP posterior and, depending on the options, nugget,
    observation noise, PCA truncation and PCA sampling). The between-members
    term is the covariance of the per-member means.

    Notes
    -----
    This is a deep-ensemble heuristic, not a posterior marginalization.
    Members differ because of different random seeds, not because they are
    draws from a Bayesian posterior over hyperparameters or variational
    parameters. Consequently, the sample covariance between members
    conflates several distinct sources of variation:

    1. Genuine posterior uncertainty: in regions with little training data,
       different inducing-point layouts produce meaningfully different
       posterior means (the "good" signal).
    2. Optimizer instability: ADAM with mini-batching can converge to
       slightly different local optima; under-trained members inflate the
       spread.
    3. Hyperparameter uncertainty: each member learns its own kernel
       parameters; their spread reflects optimization noise as much as true
       uncertainty.
    4. Initialization sensitivity: the max-min inducing-point initialization
       is deterministic given the key, so members get genuinely different
       geometric placements.

    This is exactly the Deep Ensembles approach (Lakshminarayanan et al.
    2017). It is empirically well calibrated and often outperforms
    single-model uncertainty estimates, but it is NOT a principled
    approximation to the Bayesian model average. In particular:

    - The uncertainty can be artificially inflated if the training is noisy
      or too short.
    - The uncertainty can be artificially deflated if all members collapse
      to the same optimum (common with large N and many inducing points).
    - Increasing the ensemble size K reduces the estimator variance of the
      spread, but does NOT reduce the bias from conflating the sources
      above.

    For Bayesian inference (MCMC / likelihood emulation) this is generally
    fine: slightly over-dispersed uncertainty is conservative and safe. If
    you need a calibrated emulator uncertainty for active learning or
    decision-making, consider treating the ensemble spread as an upper bound
    and validating on held-out data.
    """

    def __init__(
        self,
        n_ensemble=5,
        n_pc=0.999,
        M=200,
        base_key=None,
        init_strategy="maxmin",
        bootstrap=False,
    ):
        """
        Initialize the ensemble.

        Parameters
        ----------
        n_ensemble : int
            Number of ensemble members (default 5). The estimate of the
            spread between the members converges as 1/sqrt(K); 5-10 members
            are usually sufficient.
        n_pc : float or int
            Number of PCA components of the shared PCA basis: float in (0, 1)
            for the fraction of the explained variance, int for a fixed
            number (default 0.999).
        M : int
            Number of inducing points per member (default 200).
        base_key : jax.random.PRNGKey or None
            Master key; the member keys are derived by splitting it. None
            uses ``PRNGKey(42)`` (default None).
        init_strategy : str
            Inducing-point initialization strategy for all members, see
            PCASparseGPEmulator (default 'maxmin').
        bootstrap : bool
            If True, each member is trained on a bootstrap resample (N draws
            with replacement from the N training points) instead of the full
            dataset (default False). This increases the diversity between
            members and typically improves the calibration of the uncertainty
            estimate from their spread, at the cost of each member seeing ~63%
            unique points on average. The shared PCA basis is always fitted
            on the FULL dataset, regardless of this flag, to keep all members
            in a common output space.
        """
        self.n_ensemble = n_ensemble
        self.n_pc = n_pc
        self.M = M
        self.base_key = jax.random.PRNGKey(42) if base_key is None else base_key
        self.init_strategy = init_strategy
        self.bootstrap = bootstrap
        # set in fit(): the number of PCs and the trained members
        self.n_pc_ = None
        self.members_ = []
        self.training_histories_ = []

    @_restore_state_on_error
    def fit(self, X, Y, Y_err=None, verbose=True, verbose_members=False, **fit_kwargs):
        """
        Train all ensemble members.

        Parameters
        ----------
        X : array (N, D)
            Input training data.
        Y : array (N, P)
            Output training data.
        Y_err : array (N, P) or (N, P, P) or None
            Per-point observation uncertainty, forwarded to each member
            (default None). (N, P): independent standard deviations.
            (N, P, P): full per-point covariance matrices (see
            PCASparseGPEmulator.fit).
        verbose : bool
            Log an ensemble-level progress summary (default True).
        verbose_members : bool
            Log the training output of the individual members (default
            False).
        **fit_kwargs
            Forwarded to PCASparseGPEmulator.fit() (steps, batch_size,
            kernel_lr, early_stopping, ...); its verbose argument is set by
            `verbose_members`.

        Returns
        -------
        PCASparseGPEnsemble
            The fitted ensemble (self).

        Raises
        ------
        ValueError
            If n_pc or Y_err is invalid, or for the errors of
            PCASparseGPEmulator.fit(). The attributes of the ensemble are
            restored if fit() fails.
        """
        check_npc(self.n_pc)
        if Y_err is not None:
            _check_y_err(Y_err, Y)
        self.members_ = []
        self.training_histories_ = []
        keys = jax.random.split(self.base_key, self.n_ensemble)

        # ------------------------------------------------------------------
        # Compute ONE shared preprocessing state for all members.
        # If each member fit its own PCA the spread between the members would
        # mix genuine emulator uncertainty with PCA gauge freedom (sign flips,
        # arbitrary rotations).  By fixing the basis here, the per-member
        # predictions lie in the same output space and the sample covariance
        # of their means is a statistically clean uncertainty estimate.
        # ------------------------------------------------------------------
        _Xm = X.mean(0)
        _Xs = X.std(0) + 1e-8
        _Ym = Y.mean(0)
        _Ys = Y.std(0) + 1e-8
        _Yn = (Y - _Ym) / _Ys
        _pca, _Yp_raw = _fit_pca(_Yn, self.n_pc)
        _n_pc = _pca.n_components_
        self.n_pc_ = _n_pc
        _pc_mean = jnp.mean(_Yp_raw, axis=0)
        _pc_std = jnp.std(_Yp_raw, axis=0) + 1e-8
        _P_out = np.array(_Yn).shape[1]
        _W_ret = _pca.components_
        _lam_ret = _pca.explained_variance_
        if _n_pc < _P_out:
            _Sigma_data = np.cov(np.array(_Yn).T)
            _Sigma_ret = (_W_ret * _lam_ret[:, None]).T @ _W_ret
            _Sigma_trunc = _Sigma_data - _Sigma_ret
            _vals, _vecs = np.linalg.eigh(_Sigma_trunc)
            _vals = np.maximum(_vals, 0.0)
            _trunc_cov = jnp.array(_vecs @ (_vals[:, None] * _vecs.T))
        else:
            _trunc_cov = jnp.zeros((_P_out, _P_out))
        self.pca_state_ = {
            "Xm": _Xm,
            "Xs": _Xs,
            "Ym": _Ym,
            "Ys": _Ys,
            "pca": _pca,
            "n_pc": _n_pc,
            "pc_mean": _pc_mean,
            "pc_std": _pc_std,
            "trunc_cov_yn": _trunc_cov,
        }
        if verbose:
            ev = float(np.sum(_pca.explained_variance_ratio_))
            logger.info(
                f"Training an ensemble of {self.n_ensemble} sparse GP emulators "
                f"with {X.shape[0]} training points and {self.M} inducing points "
                f"using {_n_pc} PCs, which explain {ev:.5f} of the variance ..."
            )
        member_kwargs = {
            **fit_kwargs,
            "verbose": verbose_members,
            "_fixed_pca_state": self.pca_state_,
        }

        for k, key in enumerate(keys):
            # Bootstrap resample: draw N indices with replacement using the
            # member's own key so each member gets a deterministic but distinct
            # subset.  PCA state was fitted on full data and is unchanged.
            if self.bootstrap:
                N = X.shape[0]
                boot_idx = np.array(
                    jax.random.choice(_subkey(key, "bootstrap"), N, (N,), replace=True)
                )
                X_fit = X[boot_idx]
                Y_fit = Y[boot_idx]
                Y_err_fit = Y_err[boot_idx] if Y_err is not None else None
                n_unique = len(np.unique(boot_idx))
                if self.M > n_unique:
                    logger.warning(
                        f"[Member {k + 1}/{self.n_ensemble}] The bootstrap sample "
                        f"has only {n_unique} unique points, fewer than the "
                        f"{self.M} inducing points, which are then partly "
                        "duplicated"
                    )
                if verbose:
                    logger.info(
                        f"[Member {k + 1}/{self.n_ensemble}] Training on a "
                        f"bootstrap sample with {n_unique}/{N} unique points ..."
                    )
            else:
                X_fit, Y_fit, Y_err_fit = X, Y, Y_err
                if verbose:
                    logger.info(f"[Member {k + 1}/{self.n_ensemble}] Training ...")
            emu = PCASparseGPEmulator(
                n_pc=_n_pc,
                M=self.M,
                key=key,
                init_strategy=self.init_strategy,
            )
            emu.fit(X_fit, Y_fit, Y_err=Y_err_fit, **member_kwargs)
            self.members_.append(emu)
            self.training_histories_.append(emu.training_history_)
            if verbose:
                h = emu.training_history_
                best_score = h.get("best_score")
                if best_score is None:
                    best_score = float("nan")
                logger.info(
                    f"[Member {k + 1}/{self.n_ensemble}] Best ELBO (EMA for "
                    f"mini-batches) = {best_score:.3f} at step {h['best_step']}, "
                    f"{h['n_steps']} finite steps, converged: {h['converged']}, "
                    f"jitter = {h['jitter']:.1e}"
                )

        if verbose:
            logger.info(f"Training of the {self.n_ensemble} members finished")
        return self

    def predict(
        self,
        X_star,
        include_noise=False,
        include_truncation=True,
        include_pca_sampling=False,
        include_obs_noise=False,
        return_var_decomposition=False,
    ):
        """
        Combined ensemble prediction via the law of total variance.

        Parameters
        ----------
        X_star : array (N_test, D)
            Test inputs.
        include_noise : bool
            Add the learned nugget of each member (default False). See
            PCASparseGPEmulator.predict().
        include_truncation : bool
            Add the PCA truncation covariance (default True).
        include_pca_sampling : bool
            Add the finite-data PCA sampling uncertainty (default False).
        include_obs_noise : bool
            Add the observation noise propagated from Y_err (default False).
        return_var_decomposition : bool
            If True, also return a dict with the 'within_members' and
            'between_members' covariances (default False).

        Returns
        -------
        Y_pred : array (N_test, P)
            Ensemble mean.
        full_cov : array (N_test, P, P)
            Total predictive covariance (within + between members).
        var_decomp : dict
            Only returned if return_var_decomposition=True. Keys:

            - 'within_members' (N_test, P, P): average of the per-member
              predictive covariances.
            - 'between_members' (N_test, P, P): covariance of the per-member
              means (zeros for a single member).

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if not self.members_:
            raise RuntimeError("Call fit() before predict().")

        predict_kw = dict(
            include_noise=include_noise,
            include_truncation=include_truncation,
            include_pca_sampling=include_pca_sampling,
            include_obs_noise=include_obs_noise,
            return_var_decomposition=False,
        )
        all_means, all_covs = [], []
        for emu in self.members_:
            mu, cov = emu.predict(X_star, **predict_kw)
            all_means.append(np.array(mu))
            all_covs.append(np.array(cov))

        K = len(self.members_)
        means_arr = np.stack(all_means, axis=0)  # (K, N, P)
        covs_arr = np.stack(all_covs, axis=0)  # (K, N, P, P)
        # Ensemble mean
        Y_pred = means_arr.mean(axis=0)  # (N, P)
        # average of the per-member covariances
        within_members = covs_arr.mean(axis=0)  # (N, P, P)
        # covariance of the per-member means of the equal-weight mixture
        residuals = means_arr - Y_pred[None]  # (K, N, P)
        between_members = np.einsum("knp,knq->npq", residuals, residuals) / K

        full_cov = within_members + between_members  # (N, P, P)

        if return_var_decomposition:
            return (
                Y_pred,
                full_cov,
                {
                    "within_members": within_members,
                    "between_members": between_members,
                },
            )
        return Y_pred, full_cov

    # -------------------------
    # Per-member predictions (for diagnostics)
    # -------------------------
    def predict_members(self, X_star, **predict_kwargs):
        """
        Return the individual member predictions for diagnostics / plotting.

        Parameters
        ----------
        X_star : array (N_test, D)
            Test inputs.
        **predict_kwargs
            Forwarded to PCASparseGPEmulator.predict().

        Returns
        -------
        list of tuple
            One (Y_pred, full_cov) tuple per ensemble member, or a triple
            including var_decomp with return_var_decomposition=True.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if not self.members_:
            raise RuntimeError("Call fit() before predict_members().")
        return [emu.predict(X_star, **predict_kwargs) for emu in self.members_]
