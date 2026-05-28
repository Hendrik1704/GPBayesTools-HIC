"""
Training for Sparse Variational Gaussian Process emulators.

Implements a PCA-reduced Sparse Variational GP (SVGP) emulator using JAX and
optax.  Both a single emulator and a bootstrap ensemble variant are provided.

References:
  - Hensman et al. (2015), "Scalable Variational Gaussian Process Classification"
  - Lakshminarayanan et al. (2017), "Simple and Scalable Predictive Uncertainty
    Estimation using Deep Ensembles"
"""

import logging
import numpy as np
import pickle

import jax
import jax.numpy as jnp
import optax
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from . import cachedir, parse_model_parameter_file


# =============================================================================
# Kernel functions
# =============================================================================

def rbf_kernel(x1, x2, ls, var):
    x1 = x1 / ls
    x2 = x2 / ls
    sq = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return var * jnp.exp(-0.5 * sq)


def matern32_kernel(x1, x2, ls, var):
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
    PCA-reduced Sparse Variational Gaussian Process Emulator.

    Uses inducing point approximation with PCA-based dimensionality reduction
    for multi-output modeling. Combines RBF and Matern-3/2 kernels.

    Parameterization: whitened SVGP (GPflow convention).
      Prior on inducing outputs:  p(u) = N(0, Kzz)
      Whitened variable:          v = Lz^{-1} u,  p(v) = N(0, I)
      Variational distribution:   q(v) = N(m, S),  S = L L.T
      Posterior mean at X*:       Kxz @ Lz^{-T} @ m
      Posterior variance at X*:   Kxx - sum(A_half^2, axis=0)
                                       + sum((A_half.T @ L)^2, axis=1)
      where A_half = Lz^{-1} @ Kxz.T  (one triangular solve only)

    Hyperparameter treatment — MAP point estimates, not posteriors
    --------------------------------------------------------------
    The kernel parameters (log_lengthscale, log_var_rbf, log_var_mat, log_noise)
    are optimised to a single MAP point via ADAM on the ELBO.  The predictive
    distribution p(f* | X*, X, Y) is therefore conditioned on fixed θ_MAP and
    does NOT integrate over p(θ | X, Y).

    Consequences:
      - Predictive variance is the GP posterior variance at fixed θ, which
        systematically underestimates total uncertainty when the likelihood
        surface is broad or multi-modal in hyperparameter space.
      - This is most problematic with short lengthscales (overfitting), very
        small datasets, or when the kernel family is misspecified.
      - The PCASparseGPEnsemble partially compensates: different random seeds
        produce different θ_MAP values, so the inter-member spread captures
        some hyperparameter uncertainty.  However, if all members collapse to
        the same optimum (common with large N), this compensation vanishes.

    Principled alternatives (not implemented):
      - MCMC over θ (e.g. HMC in NumPyro/BlackJAX) — exact but expensive.
      - Laplace approximation around θ_MAP — adds a Gaussian correction to
        the predictive variance with cost O(P_θ²) where P_θ is the number
        of kernel parameters.
      - VOGP / hyperparameter variational inference — jointly optimises q(θ)
        alongside the variational GP posterior; supported in GPflux/GPyTorch.

    For Bayesian inference / MCMC-based calibration this is usually acceptable:
    the emulator is evaluated at many θ_physics proposals and the hyperparameter
    uncertainty is small relative to the parameter-space uncertainty being inferred.
    Re-evaluate if the emulator is used for decisions sensitive to tails of the
    predictive distribution (e.g. rare-event probabilities, tight design margins).
    """

    def __init__(self, n_pc=0.999, M=200, key=jax.random.PRNGKey(0), init_strategy='maxmin'):
        """
        Initialize the emulator.

        Parameters
        ----------
        n_pc : float or int
            Number of PCA components: float in (0,1) for explained variance, int for fixed count
        M : int
            Number of inducing points (default 200).  Rule of thumb: M ≈ N/5 for
            D≤10, M ≈ N/3 for D=20-35.  Increasing M improves accuracy at O(M²)
            cost in memory and O(M³) per Cholesky.
        key : jax.random.PRNGKey
            Random key for reproducibility
        init_strategy : str
            Inducing point init strategy: 'maxmin' (default, best coverage in
            moderate D), 'kmeans', 'kmeans_pp', 'random', 'sobol'
        """
        self.n_pc = n_pc
        self.M = M
        self.key = key
        self.init_strategy = init_strategy
        self.training_history = None
        self.trunc_cov_yn_ = None   # set in fit(); exact PCA truncation covariance in Yn space
        self.mean_obs_cov_pc_ = None  # set in fit(); mean obs-noise covariance (n_pc, n_pc) in standardized PC space

    # -------------------------
    # Kernel
    # -------------------------
    def kernel(self, x1, x2, p):
        ls = jax.nn.softplus(p["log_lengthscale"]) + 1e-6
        ls = jnp.clip(ls, 1e-3, 1e4)
        vr = jax.nn.softplus(p["log_var_rbf"]) + 1e-6
        vm = jax.nn.softplus(p["log_var_mat"]) + 1e-6
        vr = jnp.clip(vr, 1e-7, 10.0)
        vm = jnp.clip(vm, 1e-7, 10.0)
        return rbf_kernel(x1, x2, ls, vr) + matern32_kernel(x1, x2, ls, vm)

    def kernel_diag(self, x, p):
        vr = jax.nn.softplus(p["log_var_rbf"]) + 1e-6
        vm = jax.nn.softplus(p["log_var_mat"]) + 1e-6
        vr = jnp.clip(vr, 1e-7, 10.0)
        vm = jnp.clip(vm, 1e-7, 10.0)
        return (vr + vm) * jnp.ones(x.shape[0])

    # -------------------------
    # Build variational L (Cholesky factor)
    # -------------------------
    def build_L(self, L_unconstrained):
        """Construct positive-definite lower triangular matrix from unconstrained params."""
        L = jnp.tril(L_unconstrained)
        raw_diag = jnp.diagonal(L, axis1=-2, axis2=-1)
        pos_diag = jax.nn.softplus(raw_diag) + 1e-6
        diag_raw = jax.vmap(jnp.diag)(raw_diag)
        diag_pos = jax.vmap(jnp.diag)(pos_diag)
        return L - diag_raw + diag_pos

    # -------------------------
    # Inducing point initialisation strategies
    # -------------------------
    def init_Z_maxmin(self, X):
        N, _ = X.shape
        idx = jax.random.randint(self.key, (), 0, N)
        Z = X[idx:idx + 1]
        min_dists = jnp.sum((X - Z[0]) ** 2, axis=-1)
        for _ in range(1, self.M):
            idx = jnp.argmax(min_dists)
            Z = jnp.vstack([Z, X[idx]])
            dists_to_new = jnp.sum((X - X[idx]) ** 2, axis=-1)
            min_dists = jnp.minimum(min_dists, dists_to_new)
        return Z

    def init_Z_kmeans(self, X):
        kmeans = KMeans(self.M, n_init=10, random_state=0).fit(np.array(X))
        Z = jnp.array(kmeans.cluster_centers_)
        Z += 0.01 * jax.random.normal(self.key, Z.shape)
        return Z

    def init_Z_kmeans_pp(self, X):
        kmeans = KMeans(self.M, init='k-means++', n_init=1,
                        random_state=0).fit(np.array(X))
        return jnp.array(kmeans.cluster_centers_)

    def init_Z_random(self, X):
        N = X.shape[0]
        idx = jax.random.choice(self.key, N, (self.M,), replace=False)
        return X[idx]

    def init_Z_sobol(self, X):
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=X.shape[1], scramble=True)
        sample = sampler.random(self.M)
        X_min = jnp.min(X, axis=0)
        X_max = jnp.max(X, axis=0)
        return jnp.array(X_min + sample * (X_max - X_min))

    # -------------------------
    # Fit
    # -------------------------
    def fit(self, X, Y, Y_err=None, steps=2000, batch_size=None,
            kernel_lr=1e-3, variational_lr=1e-3, inducing_lr=3e-4,
            _fixed_pca_state=None,
            print_every=200, jitter_init=1e-5, jitter_max=1e-1,
            verbose=True, early_stopping=False, patience=50,
            es_rel_tol=1e-5, ema_alpha=0.98):
        """
        Fit the emulator to training data.

        Parameters
        ----------
        X : array (N, D)
            Input training data.
        Y : array (N, P)
            Output training data.
        Y_err : array (N, P) or (N, P, P) or None
            Per-training-point observation uncertainty in original Y units.

            **(N, P)** — independent (diagonal) errors: ``Y_err[n, j]`` is the
            standard deviation of output j at training point n.  Projected to
            standardised PC space as a diagonal covariance:
            ``Var_pc[n, i] = sum_j (W[i,j] / (pc_std[i]*Ys[j]))^2 * Y_err[n,j]^2``.

            **(N, P, P)** — correlated errors: ``Y_err[n]`` is the full ``(P, P)``
            observation covariance matrix at training point n.  Full propagation:
            ``Cov_pc[n] = W_scaled @ Y_err[n] @ W_scaled^T``
            where ``W_scaled[i,j] = W[i,j] / (pc_std[i]*Ys[j])``.
            The diagonal of ``Cov_pc[n]`` is used in the per-PC ELBO likelihood;
            the mean over n of the full ``Cov_pc`` matrix is stored and used in
            ``predict()`` for correct back-projection to output-space covariance.

            When ``None``, no extra observation noise is added.  Default None.
        _fixed_pca_state : dict or None
            Internal parameter used by PCASparseGPEnsemble.  When provided,
            skips normalization and PCA fitting and uses the pre-computed shared
            Number of optimization steps (default 2000).  For N~1000, M~200,
            convergence typically needs 1500-3000 steps; use early_stopping=True
            to avoid over-running.
        batch_size : int or None
            Mini-batch size for stochastic ELBO. None = full dataset (default).
            The likelihood term is scaled by N/batch_size so the ELBO remains
            comparable across batch sizes. Recommended: 256-1024 for large N.
        kernel_lr : float
            Learning rate for kernel parameters (default 1e-3)
        variational_lr : float
            Learning rate for variational parameters (default 1e-3)
        inducing_lr : float
            Learning rate for inducing points (default 3e-4)
        print_every : int
            Print progress every N steps (default 200)
        jitter_init : float
            Initial diagonal jitter for Kzz stability (default 1e-5).
            Auto-increased if Cholesky fails at startup or during training.
        jitter_max : float
            Maximum allowed jitter. Raises RuntimeError if exceeded (default 1e-1).
        verbose : bool
            Print training progress (default True)
        early_stopping : bool
            Enable EMA-based early stopping (default False).
            Stops when the exponential moving average of the ELBO plateaus.
        patience : int
            Consecutive steps with EMA rel. change < es_rel_tol before stopping (default 50)
        es_rel_tol : float
            EMA relative change threshold counted toward patience (default 1e-5)
        ema_alpha : float
            EMA smoothing factor in [0, 1). Higher = heavier smoothing (default 0.98).
            Effective window approx 1/(1 - ema_alpha) steps.

        Returns
        -------
        dict
            Training history: 'elbos', 'steps', 'converged', 'n_steps', 'jitter'
        """
        if verbose:
            print("=" * 60)
            print("PCASparseGPEmulator Training")
            print("=" * 60)

        if _fixed_pca_state is None:
            self.Xm, self.Xs = X.mean(0), X.std(0) + 1e-8
            Xn = (X - self.Xm) / self.Xs

            self.Ym, self.Ys = Y.mean(0), Y.std(0) + 1e-8
            Yn = (Y - self.Ym) / self.Ys

            if verbose:
                print(f"Input shape: {X.shape}, Output shape: {Y.shape}")
                print(f"Output stats - mean: [{self.Ym.min():.3f}, {self.Ym.max():.3f}], "
                      f"std: [{self.Ys.min():.3f}, {self.Ys.max():.3f}]")

            self.pca = PCA(n_components=self.n_pc)
            Yp = self.pca.fit_transform(np.array(Yn))
            self.n_pc = self.pca.n_components_
            explained_var = np.sum(self.pca.explained_variance_ratio_)

            if verbose:
                print(f"PCA: {self.n_pc} components, explained variance: {explained_var:.4f}")

            self.pc_mean = jnp.mean(Yp, axis=0)
            self.pc_std = jnp.std(Yp, axis=0) + 1e-8
            Yp = jnp.array((Yp - np.array(self.pc_mean)) / np.array(self.pc_std))

            Yn_np = np.array(Yn)
            P_out = Yn_np.shape[1]
            W_ret = self.pca.components_
            lam_ret = self.pca.explained_variance_
            if self.n_pc < P_out:
                Sigma_data = np.cov(Yn_np.T)
                Sigma_ret = (W_ret * lam_ret[:, None]).T @ W_ret
                Sigma_trunc = Sigma_data - Sigma_ret
                vals, vecs = np.linalg.eigh(Sigma_trunc)
                vals = np.maximum(vals, 0.0)
                self.trunc_cov_yn_ = jnp.array(vecs @ (vals[:, None] * vecs.T))
                if verbose:
                    ppca_approx = float(self.pca.noise_variance_) * (P_out - self.n_pc)
                    exact_trace = float(np.sum(vals))
                    print(f"Truncation covariance: exact trace={exact_trace:.4f} "
                          f"(PPCA approx trace={ppca_approx:.4f})")
            else:
                self.trunc_cov_yn_ = jnp.zeros((P_out, P_out))
                if verbose:
                    print("Truncation covariance: zero (all PCA components retained)")
        else:
            self.Xm = _fixed_pca_state['Xm']
            self.Xs = _fixed_pca_state['Xs']
            self.Ym = _fixed_pca_state['Ym']
            self.Ys = _fixed_pca_state['Ys']
            self.pca = _fixed_pca_state['pca']
            self.n_pc = _fixed_pca_state['n_pc']
            self.pc_mean = _fixed_pca_state['pc_mean']
            self.pc_std = _fixed_pca_state['pc_std']
            self.trunc_cov_yn_ = _fixed_pca_state['trunc_cov_yn_']

            Xn = (X - self.Xm) / self.Xs
            Yn = (Y - self.Ym) / self.Ys
            Yp_r = self.pca.transform(np.array(Yn))
            Yp = jnp.array((Yp_r - np.array(self.pc_mean)) / np.array(self.pc_std))

        N_full = Xn.shape[0]

        _W_np = self.pca.components_
        _pc_std_np = np.array(self.pc_std)
        _Ys_np = np.array(self.Ys)
        _W_scaled = _W_np / (_pc_std_np[:, None] * _Ys_np[None, :])

        if Y_err is not None:
            _yerr = np.asarray(Y_err, dtype=float)
            if _yerr.ndim == 2:
                if _yerr.shape != np.array(Y).shape:
                    raise ValueError(
                        f"Y_err shape {_yerr.shape} must match Y shape "
                        f"{np.array(Y).shape} for the (N, P) diagonal-error format.")
                obs_var_full = jnp.array((_yerr ** 2) @ (_W_scaled ** 2).T)
                _mean_C_Y = np.diag(np.mean(_yerr ** 2, axis=0))
                _mean_obs_cov_pc = _W_scaled @ _mean_C_Y @ _W_scaled.T
                if verbose:
                    print(f"Y_err (N,P): mean obs std = "
                          f"{float(np.sqrt(np.mean(_yerr**2))):.4g} (original Y units)")
            elif _yerr.ndim == 3:
                _P = np.array(Y).shape[1]
                if _yerr.shape != (N_full, _P, _P):
                    raise ValueError(
                        f"Y_err shape {_yerr.shape} expected ({N_full}, {_P}, {_P}) "
                        f"for the (N, P, P) full-covariance format.")
                _diags = np.array([np.diag(c) for c in _yerr])
                if np.any(_diags < 0):
                    raise ValueError(
                        "Y_err contains covariance matrices with negative diagonal "
                        "entries. Check your input.")
                obs_var_full = jnp.array(
                    np.einsum('ij,njk,ik->ni', _W_scaled, _yerr, _W_scaled))
                obs_var_full = jnp.clip(obs_var_full, 0.0, None)
                _mean_C_Y = np.mean(_yerr, axis=0)
                _mean_obs_cov_pc = _W_scaled @ _mean_C_Y @ _W_scaled.T
                _mean_obs_cov_pc = 0.5 * (_mean_obs_cov_pc + _mean_obs_cov_pc.T)
                _evals, _evecs = np.linalg.eigh(_mean_obs_cov_pc)
                _mean_obs_cov_pc = _evecs @ (np.maximum(_evals, 0.0)[:, None] * _evecs.T)
            else:
                raise ValueError(
                    f"Y_err must be shape (N, P) or (N, P, P); got {_yerr.shape}.")
            self.mean_obs_cov_pc_ = jnp.array(_mean_obs_cov_pc)
        else:
            obs_var_full = jnp.zeros((N_full, self.n_pc))
            self.mean_obs_cov_pc_ = None

        B = min(batch_size, N_full) if batch_size is not None else N_full

        if verbose:
            if B < N_full:
                print(f"Mini-batching: batch_size={B} "
                      f"(N={N_full}, scale={N_full/B:.1f}x per step)")
            else:
                print(f"Full-batch training (N={N_full})")

        if self.M > N_full:
            raise ValueError(
                f"M={self.M} inducing points cannot exceed N={N_full} training "
                f"points. Reduce M or provide more training data.")

        if self.init_strategy == 'maxmin':
            Z = self.init_Z_maxmin(Xn)
        elif self.init_strategy == 'kmeans':
            Z = self.init_Z_kmeans(Xn)
        elif self.init_strategy == 'kmeans_pp':
            Z = self.init_Z_kmeans_pp(Xn)
        elif self.init_strategy == 'random':
            Z = self.init_Z_random(Xn)
        elif self.init_strategy == 'sobol':
            Z = self.init_Z_sobol(Xn)
        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")

        self.params = {
            "Z": Z,
            "log_lengthscale": jnp.full(
                (X.shape[1],),
                float(np.log(np.expm1(float(np.sqrt(X.shape[1])))))),
            "log_var_rbf": jnp.array(0.0),
            "log_var_mat": jnp.array(-0.5),
            "log_noise": jnp.full((self.n_pc,), -2.0),
            "m": jnp.zeros((self.n_pc, self.M)),
            "L_unconstrained": jnp.zeros((self.n_pc, self.M, self.M)),
        }

        jitter = jitter_init
        if verbose:
            print(f"\nChecking Kzz stability "
                  f"(range: {jitter_init:.1e} - {jitter_max:.1e})")
        while True:
            Kzz_test = (self.kernel(Z, Z, self.params)
                        + jitter * jnp.eye(self.M))
            Lz_test = jnp.linalg.cholesky(Kzz_test)
            if not bool(jnp.any(~jnp.isfinite(Lz_test))):
                break
            jitter *= 10.0
            if jitter > jitter_max:
                raise RuntimeError(
                    f"Kzz Cholesky failed at jitter={jitter_max:.1e}. "
                    f"Try reducing M or using a different init_strategy.")
        if verbose:
            if jitter > jitter_init:
                print(f"  Increased jitter to {jitter:.1e} for stable Cholesky")
            else:
                print(f"  Kzz stable at jitter={jitter:.1e}")

        self.jitter = jitter
        self.N_train = N_full

        def elbo_fn(p, Xb, Yb, obs_noise_b, jitter_arr):
            Z = p["Z"]
            noise = jax.nn.softplus(p["log_noise"]) + 1e-6
            Kzz = self.kernel(Z, Z, p) + jitter_arr * jnp.eye(self.M)
            Lz = jnp.linalg.cholesky(Kzz)
            Kxz = self.kernel(Xb, Z, p)
            A_half = jax.scipy.linalg.solve_triangular(Lz, Kxz.T, lower=True)
            m = p["m"]
            L = self.build_L(p["L_unconstrained"])
            base_var = self.kernel_diag(Xb, p)
            qdiag = jnp.sum(A_half ** 2, axis=0)

            def pc_term(i):
                y_i = Yb[:, i]
                m_i = m[i]
                L_i = L[i]
                noise_i = noise[i]
                f_mean_i = A_half.T @ m_i
                B_i = A_half.T @ L_i
                f_var_i = jnp.clip(base_var - qdiag + jnp.sum(B_i ** 2, axis=1),
                                   1e-7, None)
                obs_var_i = noise_i + obs_noise_b[:, i]
                ll_i = -0.5 * jnp.sum(((y_i - f_mean_i) ** 2 + f_var_i) / obs_var_i)
                ll_i -= 0.5 * jnp.sum(jnp.log(obs_var_i))
                L_i_diag = jnp.clip(jnp.diag(L_i), 1e-8, None)
                kl_i = 0.5 * (
                    jnp.sum(m_i ** 2)
                    + jnp.sum(L_i ** 2)
                    - self.M
                    - 2.0 * jnp.sum(jnp.log(L_i_diag))
                )
                return ll_i, kl_i

            ll_per_pc, kl_per_pc = jax.vmap(pc_term)(jnp.arange(self.n_pc))
            total_elbo = (N_full / Xb.shape[0]) * jnp.sum(ll_per_pc) - jnp.sum(kl_per_pc)
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
        tx = optax.multi_transform(
            {
                "kernel":      optax.adam(kernel_lr),
                "variational": optax.adam(variational_lr),
                "inducing":    optax.adam(inducing_lr),
            },
            param_labels,
        )
        opt_state = tx.init(self.params)

        @jax.jit
        def step(p, opt_state, Xb, Yb, obs_noise_b, jitter_arr):
            loss_val, grads = jax.value_and_grad(
                lambda params: -elbo_fn(params, Xb, Yb, obs_noise_b, jitter_arr)
            )(p)
            updates, new_opt_state = tx.update(grads, opt_state)
            new_p = optax.apply_updates(p, updates)
            return new_p, new_opt_state, -loss_val

        p = self.params
        elbos = []
        best_elbo = -np.inf
        best_params = None
        converged = False
        nan_count = 0
        ema = None
        es_patience_count = 0
        key = self.key

        if verbose:
            print(f"\nTraining progress:")
            if early_stopping:
                print(f"Early stopping: patience={patience}, "
                      f"es_rel_tol={es_rel_tol:.1e}, "
                      f"ema_alpha={ema_alpha} (window~{1/(1-ema_alpha):.0f} steps)")

        for i in range(steps):
            key, subkey = jax.random.split(key)
            if B < N_full:
                idx = jax.random.choice(subkey, N_full, (B,), replace=False)
                Xb, Yb, obs_noise_b = Xn[idx], Yp[idx], obs_var_full[idx]
            else:
                Xb, Yb, obs_noise_b = Xn, Yp, obs_var_full

            p, opt_state, elbo_val = step(
                p, opt_state, Xb, Yb, obs_noise_b, jnp.array(jitter))

            if not jnp.isfinite(elbo_val):
                new_jitter = min(jitter * 10.0, jitter_max)
                if verbose:
                    print(f"  Step {i:5d}: NaN loss -- "
                          f"jitter {jitter:.1e} -> {new_jitter:.1e}")
                jitter = new_jitter
                self.jitter = jitter
                if best_params is not None:
                    p = {k: jnp.array(v) for k, v in best_params.items()}
                opt_state = tx.init(p)
                nan_count += 1
                if nan_count > 10:
                    raise RuntimeError(
                        f"NaN loss after {nan_count} jitter increases "
                        f"(jitter={jitter:.1e}). Consider lower learning rates "
                        f"or larger jitter_init.")
                continue

            elbo_val_f = float(elbo_val)
            elbos.append(elbo_val_f)

            if elbo_val_f > best_elbo:
                best_elbo = elbo_val_f
                best_params = {k: np.array(v) for k, v in p.items()}

            if early_stopping:
                if ema is None:
                    ema = elbo_val_f
                else:
                    ema_new = ema_alpha * ema + (1.0 - ema_alpha) * elbo_val_f
                    rel_change = abs(ema_new - ema) / (abs(ema) + 1e-8)
                    ema = ema_new
                    if rel_change < es_rel_tol:
                        es_patience_count += 1
                    else:
                        es_patience_count = 0
                    if es_patience_count >= patience:
                        converged = True
                        if verbose:
                            print(f"  Step {i:5d}/{steps}: ELBO = {elbo_val_f:10.3f} "
                                  f"(EMA={ema:.3f})")
                            print(f"\nEarly stopping: EMA plateau at step {i+1}")
                        break

            if verbose and (i % print_every == 0 or i == steps - 1):
                ema_str = f", EMA={ema:.3f}" if ema is not None else ""
                es_str = (f", pat={es_patience_count}/{patience}"
                          if early_stopping else "")
                print(f"  Step {i:5d}/{steps}: "
                      f"ELBO = {elbo_val_f:10.3f}{ema_str}{es_str}")

        if best_params is not None:
            self.params = {k: jnp.array(v) for k, v in best_params.items()}
        else:
            self.params = p

        actual_steps = len(elbos)
        self.training_history = {
            "elbos":     elbos,
            "steps":     list(range(actual_steps)),
            "converged": converged,
            "n_steps":   actual_steps,
            "jitter":    jitter,
        }

        if verbose:
            print(f"\nTraining complete. Best ELBO: {best_elbo:.3f} "
                  f"(converged: {converged})")
            print(f"Total steps: {actual_steps}/{steps}, "
                  f"jitter used: {jitter:.1e}")
            print("=" * 60)

        return self.training_history

    # -------------------------
    # Predict
    # -------------------------
    def predict(self, X_star, include_noise=True, include_truncation=True,
                include_pca_sampling=True, return_var_decomposition=False):
        """
        Make predictions on new data with full uncertainty quantification.

        Parameters
        ----------
        X_star : array (N_test, D)
            Test inputs
        include_noise : bool
            Whether to add the learned per-PC nugget/noise term (log_noise) to
            predictive variance.  The nugget plays a dual role during training:
            (a) it models observation noise, and (b) it absorbs emulation error
            that M inducing points cannot represent exactly.  Guidance:

              - Y_err provided AND emulation RMSE >> obs noise (typical for
                physics simulators with dense design/small measurement error):
                use include_noise=True.  The nugget mostly captures emulation
                uncertainty and MUST be in σ_pred for calibrated intervals.
                The Y_err obs-noise contribution (via mean_obs_cov_pc_) is also
                always included; minor double-counting is negligible compared to
                the emulation uncertainty.
              - Y_err provided AND emulation RMSE ≈ obs noise (well-converged
                emulator with M → N): use include_noise=False to avoid
                double-counting the observation noise term.
              - Y_err not provided: keep include_noise=True (default); the
                nugget is the only noise floor estimate available.
            Default True.
        include_truncation : bool
            Add exact PCA truncation uncertainty: the covariance contribution
            from all discarded PCA components, computed in fit() as
            Sigma_trunc = Sigma_data - W_ret^T diag(Lambda_ret) W_ret.
            This is exact under the linear PCA model (no PPCA isotropy
            assumption). Default True.
        include_pca_sampling : bool
            Add finite-training-data uncertainty from PCA mean estimation:
            Var(pc_mean_i) = pc_std_i^2 / N_train per component. Default True.
        return_var_decomposition : bool
            If True, return a dict of individual covariance contributions.
            Keys: "gp_posterior", "nugget", "obs_noise", "pca_truncation" (exact), "pca_sampling"

            'nugget'   : learned log_noise term (zeros if include_noise=False).
            'obs_noise': Y_err noise projected to output space (zeros if Y_err
                         was not provided; always included regardless of include_noise).

        Returns
        -------
        Y_pred : array (N_test, P)
        full_cov : array (N_test, P, P)
        var_decomp : dict, only if return_var_decomposition=True
        """
        if not hasattr(self, 'params'):
            raise RuntimeError(
                "Call fit() before predict(). The emulator has not been trained yet.")
        Xn = (X_star - self.Xm) / self.Xs
        p = self.params
        Z = p["Z"]

        Kzz = self.kernel(Z, Z, p) + self.jitter * jnp.eye(self.M)
        Lz = jnp.linalg.cholesky(Kzz)
        Ksz = self.kernel(Xn, Z, p)

        # Same whitened SVGP formulation as in elbo_fn()
        A_half = jax.scipy.linalg.solve_triangular(Lz, Ksz.T, lower=True)  # (M, N_test)
        qdiag = jnp.sum(A_half ** 2, axis=0)  # (N_test,) = diag(Ksz Kzz^{-1} Kzs)
        # Returns per-test-point output-output covariance (N_test, P, P) — sufficient
        # for single-proposal MCMC / Bayesian calibration.  Does NOT compute the joint
        # covariance Cov(f(x_a), f(x_b)) for a≠b (needed for active learning / BALD).

        m = p["m"]
        L = self.build_L(p["L_unconstrained"])
        base_var = self.kernel_diag(Xn, p)

        def pc_predict(i):
            m_i = m[i]
            L_i = L[i]
            mean_i = A_half.T @ m_i
            B_i = A_half.T @ L_i
            var_i = base_var - qdiag + jnp.sum(B_i ** 2, axis=1)
            return mean_i, jnp.clip(var_i, 1e-7, None)

        means_pc, vars_pc = jax.vmap(pc_predict)(jnp.arange(self.n_pc))
        means_pc = means_pc.T   # (N_test, n_pc), standardized PC space
        vars_gp  = vars_pc.T    # (N_test, n_pc), GP posterior variance only
        # Accumulate variance in standardized PC space
        vars_total = vars_gp

        # 1. Learned nugget / homoscedastic noise.
        # For deterministic simulators this is a training regularizer, NOT real
        # observation noise -> set include_noise=False to exclude it from predictions.
        # For stochastic simulators / experiments without Y_err, keep True (default).
        noise = jax.nn.softplus(p["log_noise"]) + 1e-6  # (n_pc,)
        if include_noise:
            vars_total = vars_total + noise[None, :]

        # 2. Finite-data PCA sampling: standard error of pc_mean estimate
        #    Var(pc_mean_i) = pc_std_i^2 / N_train  ->  1/N_train in standardized space
        if include_pca_sampling:
            vars_total = vars_total + (1.0 / self.N_train)

        # Undo PC normalization -> original PC space
        means_pc        = means_pc * self.pc_std + self.pc_mean
        vars_total_orig = vars_total * (self.pc_std ** 2)   # (N_test, n_pc)
        vars_gp_orig    = vars_gp    * (self.pc_std ** 2)   # for decomposition

        # Back-project from original PC space -> normalized output (Yn) space
        W      = jnp.array(self.pca.components_)  # (n_pc, P)
        Wt     = W.T                               # (P, n_pc)
        P_size = W.shape[1]
        full_cov = jnp.einsum("pi,ni,qi->npq", Wt, vars_total_orig, Wt)  # (N_test, P, P)

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
        trunc_cov_yn = self.trunc_cov_yn_  # (P, P), exact, PSD, set in fit()
        if include_truncation and trunc_cov_yn is not None:
            full_cov = full_cov + trunc_cov_yn[None, :, :]
        else:
            trunc_cov_yn = jnp.zeros((P_size, P_size))

        # Scale from Yn space to original Y space: Cov_Y[p,q] = Ys[p]*Cov_Yn[p,q]*Ys[q]
        Ys = jnp.array(self.Ys)
        Ys_outer = jnp.outer(Ys, Ys)
        full_cov = full_cov * Ys_outer[None, :, :]

        # 4. Observation noise from Y_err — back-projected via the stored (n_pc, n_pc)
        # mean covariance.  Always added when Y_err was provided, regardless of
        # include_noise.  Handles both (N,P) and (N,P,P) Y_err inputs uniformly.
        if self.mean_obs_cov_pc_ is not None:
            obs_cov_pc_orig = self.mean_obs_cov_pc_ * jnp.outer(self.pc_std, self.pc_std)  # (n_pc, n_pc)
            obs_cov_yn      = Wt @ obs_cov_pc_orig @ W                                     # (P, P)
            full_cov        = full_cov + (obs_cov_yn * Ys_outer)[None, :, :]

        Y_pred = self.pca.inverse_transform(np.array(means_pc))
        Y_pred = Y_pred * self.Ys + self.Ym

        if return_var_decomposition:
            # GP posterior covariance in Y space
            gp_cov = (jnp.einsum("pi,ni,qi->npq", Wt, vars_gp_orig, Wt)
                      * Ys_outer[None, :, :])

            # Nugget covariance in Y space (gated by include_noise)
            if include_noise:
                nugget_orig   = noise * (self.pc_std ** 2)                       # (n_pc,)
                nugget_cov_yn = jnp.einsum("pi,i,qi->pq", Wt, nugget_orig, Wt)  # (P, P)
                nugget_cov    = (nugget_cov_yn * Ys_outer)[None, :, :]
            else:
                nugget_cov = jnp.zeros((1, P_size, P_size))

            # Known observation noise from Y_err (always included when provided).
            # Uses the stored full (n_pc, n_pc) covariance — correct for both
            # diagonal (N,P) and full-covariance (N,P,P) Y_err inputs.
            if self.mean_obs_cov_pc_ is not None:
                obs_cov_pc_orig = self.mean_obs_cov_pc_ * jnp.outer(self.pc_std, self.pc_std)
                obs_cov_yn      = Wt @ obs_cov_pc_orig @ W
                obs_noise_cov   = (obs_cov_yn * Ys_outer)[None, :, :]
            else:
                obs_noise_cov = jnp.zeros((1, P_size, P_size))

            # Truncation covariance in Y space (exact, not PPCA)
            if include_truncation and trunc_cov_yn is not None:
                trunc_cov_y = (trunc_cov_yn * Ys_outer)[None, :, :]
            else:
                trunc_cov_y = jnp.zeros((1, P_size, P_size))

            # PCA sampling covariance in Y space
            if include_pca_sampling:
                pca_samp_pc     = (self.pc_std ** 2) / self.N_train        # (n_pc,)
                pca_samp_cov_yn = jnp.einsum("pi,i,qi->pq", Wt, pca_samp_pc, Wt)
                pca_samp_cov    = (pca_samp_cov_yn * Ys_outer)[None, :, :]
            else:
                pca_samp_cov = jnp.zeros((1, P_size, P_size))

            return Y_pred, full_cov, {
                "gp_posterior":  gp_cov,
                "nugget":        nugget_cov,
                "obs_noise":     obs_noise_cov,
                "pca_truncation": trunc_cov_y,
                "pca_sampling":  pca_samp_cov,
            }

        return Y_pred, full_cov


# =============================================================================
# PCASparseGPEnsemble
# =============================================================================

class PCASparseGPEnsemble:
    """
    Bootstrap ensemble of PCASparseGPEmulator for epistemic uncertainty quantification.

    Each member is trained with a different JAX random key, giving different:
      - inducing-point initialisation
      - mini-batch orderings (stochastic ELBO)
      - ADAM optimiser trajectories / local optima
      - PCA decomposition (when N is small)

    Predictions are combined via the law of total variance:

        E[Y | x*]   = (1/K) sum_k  mu_k(x*)
        Cov[Y | x*] = (1/K) sum_k  Sigma_k(x*)          # aleatoric
                    + (1/(K-1)) sum_k (mu_k - E[Y])(mu_k - E[Y])^T  # epistemic

    The aleatoric term is the mean of individual predictive covariances (GP posterior,
    observation noise, PCA truncation, PCA sampling). The epistemic term is the sample
    covariance of the per-member means.

    Statistical caveat — deep-ensemble heuristic, not posterior marginalization
    ---------------------------------------------------------------------------
    Members differ because of different random seeds, not because they are draws
    from a Bayesian posterior over hyperparameters or variational parameters.
    Consequently the inter-member sample covariance conflates several distinct
    sources of variation:

      1. Genuine posterior uncertainty — regions with little training data where
         different inducing-point layouts produce meaningfully different posterior
         means (the "good" signal).
      2. Optimiser instability — ADAM with mini-batching can converge to slightly
         different local optima; under-trained members inflate the spread.
      3. Hyperparameter uncertainty — each member learns its own kernel parameters;
         their spread reflects optimisation noise as much as true uncertainty.
      4. Initialisation sensitivity — max-min inducing-point init is deterministic
         given the key, so members get genuinely different geometric placements.

    This is exactly the Deep Ensembles approach (Lakshminarayanan et al. 2017).
    It is empirically well-calibrated and often outperforms single-model uncertainty
    estimates, but it is NOT a principled approximation to the Bayesian model average.
    In particular:
      - Uncertainty can be artificially inflated if training is noisy or too short.
      - Uncertainty can be artificially deflated if all members collapse to the same
        optimum (common with large N and many inducing points).
      - Increasing ensemble size K reduces estimator variance of the spread but does
        NOT reduce the bias from conflating the sources above.

    For Bayesian inference (MCMC / likelihood emulation) this is generally fine:
    slightly over-dispersed uncertainty is conservative and safe.  If you need
    calibrated epistemic uncertainty for active learning or decision-making, consider
    treating the ensemble spread as an upper bound and validating on held-out data.
    """

    def __init__(self, n_ensemble=5, n_pc=0.999, M=200,
                 base_key=jax.random.PRNGKey(42), init_strategy='maxmin',
                 bootstrap=False):
        """
        Parameters
        ----------
        n_ensemble : int
            Number of ensemble members (default 5). Epistemic uncertainty
            estimate converges as 1/sqrt(K); 5-10 members is usually sufficient.
        n_pc : float or int
            Forwarded to each PCASparseGPEmulator.
        M : int
            Number of inducing points per member.
        base_key : jax.random.PRNGKey
            Master key; member keys are derived by splitting this.
        init_strategy : str
            Inducing-point initialisation strategy for all members.
        bootstrap : bool
            If True, each member is trained on a bootstrap resample (N draws with
            replacement from the N training points) instead of the full dataset.
            This increases diversity between members and typically improves
            calibration of the epistemic uncertainty estimate, at the cost of each
            member seeing ~63% unique points on average.  Default False.
            Note: the shared PCA basis is always fitted on the FULL dataset,
            regardless of this flag, to keep all members in a common output space.
        """
        self.n_ensemble = n_ensemble
        self.n_pc = n_pc
        self.M = M
        self.base_key = base_key
        self.init_strategy = init_strategy
        self.bootstrap = bootstrap
        self.members = []
        self.training_histories = []

    def fit(self, X, Y, Y_err=None, verbose=True, verbose_members=False,
            **fit_kwargs):
        """
        Train all ensemble members.

        Parameters
        ----------
        X : array (N, D)
        Y : array (N, P)
        Y_err : array (N, P), (N, P, P), or None
            Per-point observation uncertainty (forwarded to each member).
            (N, P)   — independent standard deviations.
            (N, P, P) — full per-point covariance matrices.
            When provided, use include_noise=False in predict() so the nugget
            (training regularizer) does not double-count the known noise floor.
        verbose : bool
            Print ensemble-level progress summary (default True).
        verbose_members : bool
            Print individual member training output (default False).
        **fit_kwargs
            Forwarded verbatim to PCASparseGPEmulator.fit()
            (steps, batch_size, kernel_lr, early_stopping, ...).

        Returns
        -------
        self
        """
        self.members = []
        self.training_histories = []
        keys = jax.random.split(self.base_key, self.n_ensemble)

        # ------------------------------------------------------------------
        # Compute ONE shared preprocessing state for all members.
        # If each member fit its own PCA the ensemble epistemic variance would
        # mix genuine emulator uncertainty with PCA gauge freedom (sign flips,
        # arbitrary rotations).  By fixing the basis here, the per-member
        # predictions lie in the same output space and the sample covariance
        # of their means is a statistically clean epistemic uncertainty estimate.
        # ------------------------------------------------------------------
        _Xm  = X.mean(0); _Xs = X.std(0) + 1e-8
        _Ym  = Y.mean(0); _Ys = Y.std(0) + 1e-8
        _Yn  = (Y - _Ym) / _Ys
        _pca = PCA(n_components=self.n_pc)
        _Yp_raw  = _pca.fit_transform(np.array(_Yn))
        _n_pc    = _pca.n_components_
        _pc_mean = jnp.mean(_Yp_raw, axis=0)
        _pc_std  = jnp.std(_Yp_raw,  axis=0) + 1e-8
        _P_out   = np.array(_Yn).shape[1]
        _W_ret   = _pca.components_
        _lam_ret = _pca.explained_variance_
        if _n_pc < _P_out:
            _Sigma_data  = np.cov(np.array(_Yn).T)
            _Sigma_ret   = (_W_ret * _lam_ret[:, None]).T @ _W_ret
            _Sigma_trunc = _Sigma_data - _Sigma_ret
            _vals, _vecs = np.linalg.eigh(_Sigma_trunc)
            _vals        = np.maximum(_vals, 0.0)
            _trunc_cov   = jnp.array(_vecs @ (_vals[:, None] * _vecs.T))
        else:
            _trunc_cov = jnp.zeros((_P_out, _P_out))
        self.pca_state_ = {
            'Xm': _Xm, 'Xs': _Xs, 'Ym': _Ym, 'Ys': _Ys,
            'pca': _pca, 'n_pc': _n_pc,
            'pc_mean': _pc_mean, 'pc_std': _pc_std,
            'trunc_cov_yn_': _trunc_cov,
        }
        if verbose:
            ev = float(np.sum(_pca.explained_variance_ratio_))
            print(f"Shared PCA: {_n_pc} components, explained variance: {ev:.4f} "
                  f"(fixed for all {self.n_ensemble} members)")
        member_kwargs = {**fit_kwargs, 'verbose': verbose_members,
                         '_fixed_pca_state': self.pca_state_}

        for k, key in enumerate(keys):
            # Bootstrap resample: draw N indices with replacement using the
            # member's own key so each member gets a deterministic but distinct
            # subset.  PCA state was fitted on full data and is unchanged.
            if self.bootstrap:
                N = X.shape[0]
                boot_idx = np.array(jax.random.choice(key, N, (N,), replace=True))
                X_fit     = X[boot_idx]
                Y_fit     = Y[boot_idx]
                Y_err_fit = Y_err[boot_idx] if Y_err is not None else None
                if verbose:
                    n_unique = len(np.unique(boot_idx))
                    print(f"[Ensemble {k+1}/{self.n_ensemble}] Bootstrap: "
                          f"{n_unique}/{N} unique points ({100*n_unique/N:.0f}%)",
                          flush=True)
            else:
                X_fit, Y_fit, Y_err_fit = X, Y, Y_err
                if verbose:
                    print(f"[Ensemble {k+1}/{self.n_ensemble}] Training ...", flush=True)
            emu = PCASparseGPEmulator(
                n_pc=_n_pc, M=self.M,
                key=key, init_strategy=self.init_strategy,
            )
            emu.fit(X_fit, Y_fit, Y_err=Y_err_fit, **member_kwargs)
            self.members.append(emu)
            self.training_histories.append(emu.training_history)
            if verbose:
                h = emu.training_history
                print(f"  best ELBO={max(h['elbos']):.2f}, "
                      f"steps={h['n_steps']}, "
                      f"converged={h['converged']}, "
                      f"jitter={h['jitter']:.1e}")

        if verbose:
            print(f"Ensemble of {self.n_ensemble} members trained.")
        return self

    def predict(self, X_star, include_noise=True, include_truncation=True,
                include_pca_sampling=True, return_var_decomposition=False):
        """
        Combined ensemble prediction via the law of total variance.

        Parameters
        ----------
        X_star : array (N_test, D)
        include_noise : bool
        include_truncation : bool
        include_pca_sampling : bool
        return_var_decomposition : bool
            If True, also return a dict with 'aleatoric' and 'epistemic' covariances.

        Returns
        -------
        Y_pred : array (N_test, P)
            Ensemble mean.
        full_cov : array (N_test, P, P)
            Total predictive covariance (aleatoric + epistemic).
        var_decomp : dict, only if return_var_decomposition=True
            Keys:
              'aleatoric' (N_test, P, P): average of per-member predictive covariances.
              'epistemic'  (N_test, P, P): sample covariance of per-member means.
        """
        if not self.members:
            raise RuntimeError("Call fit() before predict().")

        predict_kw = dict(
            include_noise=include_noise,
            include_truncation=include_truncation,
            include_pca_sampling=include_pca_sampling,
            return_var_decomposition=False,
        )
        all_means, all_covs = [], []
        for emu in self.members:
            mu, cov = emu.predict(X_star, **predict_kw)
            all_means.append(np.array(mu))
            all_covs.append(np.array(cov))

        K          = len(self.members)
        means_arr  = np.stack(all_means, axis=0)   # (K, N, P)
        covs_arr   = np.stack(all_covs,  axis=0)   # (K, N, P, P)
        # Ensemble mean
        Y_pred     = means_arr.mean(axis=0)          # (N, P)
        # Aleatoric: average of per-member covariances
        aleatoric  = covs_arr.mean(axis=0)           # (N, P, P)

        # Epistemic: sample covariance of per-member means (Bessel-corrected)
        residuals  = means_arr - Y_pred[None]        # (K, N, P)
        if K > 1:
            epistemic = np.einsum('knp,knq->npq', residuals, residuals) / (K - 1)
        else:
            epistemic = np.zeros_like(aleatoric)

        full_cov   = aleatoric + epistemic            # (N, P, P)

        if return_var_decomposition:
            return Y_pred, full_cov, {
                "aleatoric": aleatoric,
                "epistemic": epistemic,
            }
        return Y_pred, full_cov

    # -------------------------
    # Per-member predictions (for diagnostics)
    # -------------------------
    def predict_members(self, X_star, **predict_kwargs):
        """
        Return individual member predictions for diagnostics / plotting.

        Returns
        -------
        list of (Y_pred, full_cov) tuples, one per ensemble member.
        """
        if not self.members:
            raise RuntimeError("Call fit() before predict_members().")
        return [emu.predict(X_star, **predict_kwargs) for emu in self.members]


# =============================================================================
# EmulatorSparseGP — high-level wrapper (same interface as EmulatorBAND)
# =============================================================================

class EmulatorSparseGP:
    """
    High-level wrapper around PCASparseGPEmulator / PCASparseGPEnsemble that
    follows the same interface as EmulatorBAND.

    Can operate in two modes controlled by ``n_ensemble``:

    * ``n_ensemble=1`` — single PCASparseGPEmulator.
    * ``n_ensemble>1`` — PCASparseGPEnsemble of that many members.
    """

    def __init__(self, training_set_path=".", parameter_file="ABCD.txt",
                 n_pc=0.999, M=200, n_ensemble=1,
                 init_strategy='maxmin', bootstrap=False,
                 logTrafo=False, max_rel_uncertainty_data=0.1):
        """
        Parameters
        ----------
        training_set_path : str
            Path to the pickle file with training data.
        parameter_file : str
            Path to the model parameter file.
        n_pc : float or int
            Number of PCA components passed to the inner emulator.
        M : int
            Number of inducing points.
        n_ensemble : int
            Number of ensemble members.  Use 1 for a single emulator.
        init_strategy : str
            Inducing-point initialisation strategy.
        bootstrap : bool
            Bootstrap resampling for ensemble members.
        logTrafo : bool
            If True, log-transform outputs before training and inverse-transform
            predictions.
        max_rel_uncertainty_data : float
            Maximum relative statistical uncertainty; training points with
            larger values are discarded.
        """
        self.n_pc_ = n_pc
        self.M_ = M
        self.n_ensemble_ = n_ensemble
        self.init_strategy_ = init_strategy
        self.bootstrap_ = bootstrap
        self.logTrafo_ = logTrafo
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

        self.nev, self.nobs = self.model_data.shape
        self.nparameters = self.design_points.shape[1]

    # -------------------------
    # Data loading
    # -------------------------
    def _load_training_data_pickle(self, dataFile):
        """Load training data from a pickle file."""
        logging.info("loading training data from {} ...".format(dataFile))
        self.model_data = []
        self.model_data_err = []
        self.design_points = []
        with open(dataFile, "rb") as fp:
            dataDict = pickle.load(fp)

        sorted_event_ids = sorted(dataDict.keys(), key=lambda x: int(x))

        discarded_points = 0
        for event_id in sorted_event_ids:
            temp_data = dataDict[event_id]["obs"].transpose()
            statErrMax = np.abs(
                (temp_data[:, 1] / (temp_data[:, 0] + 1e-16))).max()
            if statErrMax > self.max_rel_uncertainty_data_:
                logging.info(
                    "Discard Parameter {}, stat err = {:.2f}".format(
                        event_id, statErrMax))
                discarded_points += 1
                continue
            self.design_points.append(dataDict[event_id]["parameter"])
            if not self.logTrafo_:
                self.model_data.append(temp_data[:, 0])
                self.model_data_err.append(temp_data[:, 1])
            else:
                self.model_data.append(
                    np.log(np.abs(temp_data[:, 0]) + 1e-30))
                self.model_data_err.append(
                    np.abs(temp_data[:, 1] / (temp_data[:, 0] + 1e-30))
                )

        self.design_points = np.array(self.design_points)
        self.model_data = np.array(self.model_data)
        self.model_data_err = np.nan_to_num(np.abs(np.array(self.model_data_err)))
        logging.info("All training data are loaded.")
        logging.info(
            "Training dataset size: {}, discarded points: {}".format(
                len(self.model_data), discarded_points))

    # -------------------------
    # Training
    # -------------------------
    def trainEmulatorAutoMask(self, **fit_kwargs):
        """Train on all available training points."""
        train_event_mask = [True] * self.nev
        self.trainEmulator(train_event_mask, **fit_kwargs)

    def trainEmulator(self, event_mask, **fit_kwargs):
        """
        Train the (ensemble) emulator on the masked subset of training data.

        Parameters
        ----------
        event_mask : list of bool, length nev
            True entries are included in training.
        **fit_kwargs
            Forwarded to PCASparseGPEmulator.fit() or PCASparseGPEnsemble.fit().
        """
        logging.info('Performing sparse GP emulator training ...')
        X = self.design_points[event_mask, :]
        Y = self.model_data[event_mask, :]
        Y_err = self.model_data_err[event_mask, :]
        logging.info('Train sparse GP with {} training points ...'.format(
            X.shape[0]))

        if self.n_ensemble_ <= 1:
            self.emu_ = PCASparseGPEmulator(
                n_pc=self.n_pc_, M=self.M_,
                init_strategy=self.init_strategy_,
            )
            self.emu_.fit(X, Y, Y_err=Y_err, **fit_kwargs)
        else:
            self.emu_ = PCASparseGPEnsemble(
                n_ensemble=self.n_ensemble_,
                n_pc=self.n_pc_, M=self.M_,
                init_strategy=self.init_strategy_,
                bootstrap=self.bootstrap_,
            )
            self.emu_.fit(X, Y, Y_err=Y_err, **fit_kwargs)

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, X, return_cov=True, extra_std=0.0,
                include_noise=True, include_truncation=True,
                include_pca_sampling=True):
        """
        Predict model output at parameter points ``X``.

        Parameters
        ----------
        X : array (N_test, nparameters)
            Parameter points in original (non-normalised) space.
        return_cov : bool
            If True, return covariance matrices (default True).
        extra_std : float or array (N_test,)
            Additional standard deviation added to the diagonal of each
            predictive covariance (for use in MCMC calibration).
        include_noise : bool
            Include learned nugget in predictive variance.
        include_truncation : bool
            Include PCA truncation covariance.
        include_pca_sampling : bool
            Include finite-data PCA sampling uncertainty.

        Returns
        -------
        fpredmean : array (N_test, nobs)
        fpredcov  : array (N_test, nobs, nobs), only when return_cov=True
        """
        if not hasattr(self, 'emu_'):
            raise RuntimeError(
                "Call trainEmulator() before predict().")

        X = np.atleast_2d(X)
        Y_pred, full_cov = self.emu_.predict(
            X,
            include_noise=include_noise,
            include_truncation=include_truncation,
            include_pca_sampling=include_pca_sampling,
        )

        Y_pred = np.array(Y_pred)
        full_cov = np.array(full_cov)

        # Inverse log-transform if needed
        if self.logTrafo_:
            Y_pred_exp = np.exp(Y_pred)
            # delta method: Cov_y[i,j] = exp(mu_i) * Cov_log[i,j] * exp(mu_j)
            outer_exp = Y_pred_exp[:, :, None] * Y_pred_exp[:, None, :]
            full_cov = full_cov * outer_exp
            Y_pred = Y_pred_exp

        # Add extra_std to diagonal
        if np.any(extra_std != 0.0):
            extra = np.atleast_1d(extra_std)
            for i in range(X.shape[0]):
                np.fill_diagonal(
                    full_cov[i],
                    full_cov[i].diagonal() + extra[i] ** 2,
                )

        if return_cov:
            return Y_pred, full_cov
        return Y_pred

    # -------------------------
    # Validation
    # -------------------------
    def testEmulatorErrors(self, number_test_points=1, **fit_kwargs):
        """
        Leave-one-out (or leave-n-out) emulator validation.

        Uses (nev - number_test_points) points to train the emulator and
        evaluates it on the held-out points.

        Returns
        -------
        emulator_predictions : array (number_test_points, nobs)
        emulator_predictions_err : array (number_test_points, nobs)
            Predictive standard deviations (sqrt of diagonal of covariance).
        validation_data : array (number_test_points, nobs)
        validation_data_err : array (number_test_points, nobs)
        """
        logging.info("Validating sparse GP emulator ...")
        event_idx_list = range(self.nev - number_test_points, self.nev)
        train_event_mask = [True] * self.nev
        for event_i in event_idx_list:
            train_event_mask[event_i] = False

        self.trainEmulator(train_event_mask, **fit_kwargs)
        validate_event_mask = [not i for i in train_event_mask]

        pred_mean, pred_cov = self.predict(
            self.design_points[validate_event_mask, :],
            return_cov=True,
        )
        pred_var = np.sqrt(np.array(
            [pred_cov[i].diagonal() for i in range(pred_cov.shape[0])]))

        if self.logTrafo_:
            # predictions are already back-transformed by self.predict()
            validation_data = np.exp(self.model_data[validate_event_mask, :])
            validation_data_err = (
                self.model_data_err[validate_event_mask, :]
                * validation_data
            )
        else:
            validation_data = self.model_data[validate_event_mask, :]
            validation_data_err = self.model_data_err[validate_event_mask, :]

        emulator_predictions = np.array(pred_mean).reshape(-1, self.nobs)
        emulator_predictions_err = np.array(pred_var).reshape(-1, self.nobs)
        validation_data = np.array(validation_data).reshape(-1, self.nobs)
        validation_data_err = np.array(validation_data_err).reshape(-1, self.nobs)

        return (emulator_predictions, emulator_predictions_err,
                validation_data, validation_data_err)
