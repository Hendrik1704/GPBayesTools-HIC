"""
Fast synthetic end-to-end tests for the sparse variational GP emulator.

Tests are intentionally small-scale (N=300, M=30, steps=400) so the full
suite finishes in a few minutes on CPU.

Tests covered
-------------
  T1  PCASparseGPEmulator smoke test — shapes and training_history keys
  T2  Y_err increases predictive variance
  T3  PCASparseGPEnsemble shapes, shared PCA basis, Y_err effect
  T4  Variance decomposition identity  aleatoric + epistemic == full_cov (diag)
  T5  predict_members consistency  mean(member means) == ensemble Y_pred
  T6  OOD aleatoric uncertainty higher than in-domain
  T7  Calibration: empirical 1σ / 2σ coverage within generous bounds
  T8  API contract  return_var_decomposition=False returns 2-tuple
  T9  Y_err shape mismatch raises ValueError
  T10 EmulatorSparseGP high-level wrapper end-to-end (pickle format)
"""

import os
import pickle
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np

# Resolve the project root (one level up from tests/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from src.emulator_sparseGP import (
    PCASparseGPEmulator,
    PCASparseGPEnsemble,
    EmulatorSparseGP,
)

# ── ANSI helpers ──────────────────────────────────────────────────────────────
PASS_SYM = "\033[92m✓\033[0m"
FAIL_SYM = "\033[91m✗\033[0m"

_results = []


def check(name, cond, detail=""):
    status = PASS_SYM if cond else FAIL_SYM
    label  = "PASS" if cond else "FAIL"
    _results.append((label, name, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))


# ── Shared small-scale training data ─────────────────────────────────────────
# N=300, D=2, P=3.  True function uses only one cycle per unit interval so
# M=30 inducing points (inter-spacing ≈ 0.18) can resolve the lengthscale.
_KEY  = jax.random.PRNGKey(42)
_N    = 300
_D    = 2
_P    = 3

_X_tr = jax.random.uniform(_KEY, (_N, _D))

def _true(X):
    return jnp.stack([
        jnp.sin(2 * jnp.pi * X[:, 0]),
        jnp.cos(2 * jnp.pi * X[:, 1]) * X[:, 0],
        X[:, 0] * X[:, 1],
    ], axis=-1)

_Y_tr  = _true(_X_tr) + 0.05 * jax.random.normal(_KEY, (_N, _P))
# Heteroscedastic errors — larger near x=1
_Y_err = 0.02 + 0.08 * jnp.abs(_X_tr[:, :1]) * jnp.ones((_N, _P))

# In-domain test set
_X_te = jax.random.uniform(jax.random.PRNGKey(99), (100, _D))
_Y_te = _true(_X_te)

# OOD test set — outside the training domain [0,1]²
_X_ood = jax.random.uniform(jax.random.PRNGKey(77), (60, _D), minval=1.3, maxval=1.7)

# Shared fast fit kwargs: steps is a ceiling; early_stopping cuts it earlier.
_FIT_KW = dict(
    steps=400, batch_size=64,
    kernel_lr=1e-3, variational_lr=1e-3, inducing_lr=3e-4,
    early_stopping=True, patience=50, verbose=False,
)

# ── Configuration ─────────────────────────────────────────────────────────────
N_PC       = 2      # fixed integer PCA components (fast)
M          = 30     # inducing points per emulator
N_ENSEMBLE = 3      # ensemble members (minimum for epistemic variance)

MODEL_PAR_FILE = os.path.join(PROJECT_ROOT, "modelDesign_test_3par.txt")
TRAINING_PKL   = os.path.join(PROJECT_ROOT, "tests", "synthetic_sparse_training.pickle")


# ── Analytical ground-truth model (for high-level wrapper test) ───────────────
def _hl_true_model(params, n_obs=6):
    """
    Map a 3-parameter vector (alpha, beta, gamma) ∈ [0,1]³ to n_obs outputs.

    f_k(alpha, beta, gamma) =
        (1 + alpha) * sin(pi * k/n_obs * beta)
        + gamma²   * cos(2*pi * k/n_obs)
        + 3          (shift to ensure strictly positive values)
    """
    alpha, beta, gamma = params
    k = np.arange(n_obs, dtype=float)
    frac = k / n_obs
    return (
        (1.0 + alpha) * np.sin(np.pi * frac * beta)
        + gamma**2    * np.cos(2.0 * np.pi * frac)
        + 3.0
    )


def _lhd(n_samples, n_dim, rng):
    """Simple random Latin-hypercube design in [0, 1]^n_dim."""
    result = np.zeros((n_samples, n_dim))
    for d in range(n_dim):
        perm = rng.permutation(n_samples)
        result[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
    return result


# =============================================================================
# T1 — PCASparseGPEmulator smoke test
# =============================================================================
def test_t1():
    print("=" * 56)
    print("T1 — PCASparseGPEmulator smoke test (no Y_err)")
    print("=" * 56)
    try:
        em = PCASparseGPEmulator(n_pc=N_PC, M=M, key=_KEY)
        em.fit(_X_tr, _Y_tr, **_FIT_KW)
        yp, cov = em.predict(_X_te)
        check("Y_pred shape",
              yp.shape == (_X_te.shape[0], _P))
        check("full_cov shape",
              cov.shape == (_X_te.shape[0], _P, _P))
        check("full_cov diag ≥ 0",
              bool(jnp.all(jnp.diagonal(cov, axis1=1, axis2=2) >= 0)))
        check("training_history keys present",
              all(k in em.training_history
                  for k in ("elbos", "steps", "converged", "n_steps", "jitter")))
        return em  # reuse in T2
    except Exception as e:
        check("T1 raised exception", False, str(e))
        traceback.print_exc()
        return None


# =============================================================================
# T2 — Y_err increases predictive variance
# =============================================================================
def test_t2():
    print()
    print("=" * 56)
    print("T2 — Y_err increases predictive variance")
    print("=" * 56)
    try:
        em_no  = PCASparseGPEmulator(n_pc=N_PC, M=M, key=_KEY)
        em_yes = PCASparseGPEmulator(n_pc=N_PC, M=M, key=_KEY)
        em_no .fit(_X_tr, _Y_tr,         **_FIT_KW)
        em_yes.fit(_X_tr, _Y_tr, _Y_err, **_FIT_KW)

        _, cov_no  = em_no .predict(_X_te, include_obs_noise=True)
        _, cov_yes = em_yes.predict(_X_te, include_obs_noise=True)
        var_no  = float(jnp.diagonal(cov_no,  axis1=1, axis2=2).mean())
        var_yes = float(jnp.diagonal(cov_yes, axis1=1, axis2=2).mean())
        check("Y_err increases mean predictive variance",
              var_yes > var_no,
              f"no-err={var_no:.4e}  with-err={var_yes:.4e}")
        check("trunc_cov_yn_ stored after fit",
              hasattr(em_yes, "trunc_cov_yn_") and em_yes.trunc_cov_yn_ is not None)
        return em_yes  # reuse in T7
    except Exception as e:
        check("T2 raised exception", False, str(e))
        traceback.print_exc()
        return None


# =============================================================================
# T3 — PCASparseGPEnsemble shapes, shared PCA basis, Y_err effect
# =============================================================================
def test_t3():
    print()
    print("=" * 56)
    print("T3 — PCASparseGPEnsemble shapes, shared PCA, Y_err")
    print("=" * 56)
    ens_no = ens_yes = None
    try:
        ens_no  = PCASparseGPEnsemble(n_ensemble=N_ENSEMBLE, n_pc=N_PC, M=M, base_key=_KEY)
        ens_yes = PCASparseGPEnsemble(n_ensemble=N_ENSEMBLE, n_pc=N_PC, M=M, base_key=_KEY)
        ens_no .fit(_X_tr, _Y_tr,         **{**_FIT_KW, "verbose": False})
        ens_yes.fit(_X_tr, _Y_tr, _Y_err, **{**_FIT_KW, "verbose": False})

        yp_no,  cov_no,  dec_no  = ens_no .predict(
            _X_te,
            include_obs_noise=True,
            return_var_decomposition=True,
        )
        yp_yes, cov_yes, dec_yes = ens_yes.predict(
            _X_te,
            include_obs_noise=True,
            return_var_decomposition=True,
        )

        N_te = _X_te.shape[0]
        check("Y_pred shape",    yp_yes.shape  == (N_te, _P))
        check("full_cov shape",  cov_yes.shape == (N_te, _P, _P))
        check("aleatoric shape", dec_yes["aleatoric"].shape == (N_te, _P, _P))
        check("epistemic shape", dec_yes["epistemic"].shape == (N_te, _P, _P))

        var_no  = float(jnp.diagonal(cov_no,  axis1=1, axis2=2).mean())
        var_yes = float(jnp.diagonal(cov_yes, axis1=1, axis2=2).mean())
        check("Y_err increases ensemble predictive variance",
              var_yes > var_no,
              f"no-err={var_no:.4e}  with-err={var_yes:.4e}")

        # All members must share the identical PCA weight matrix
        Ws = [m.pca.components_ for m in ens_yes.members]
        max_diffs = [float(np.max(np.abs(Ws[i] - Ws[0]))) for i in range(1, len(Ws))]
        check("All members share identical PCA basis (max diff == 0)",
              all(d == 0.0 for d in max_diffs),
              f"max_diff={max(max_diffs):.2e}")

        return ens_yes, yp_yes, cov_yes, dec_yes
    except Exception as e:
        check("T3 raised exception", False, str(e))
        traceback.print_exc()
        return ens_yes, None, None, None


# =============================================================================
# T4 — Variance decomposition identity
# =============================================================================
def test_t4(cov_yes, dec_yes):
    print()
    print("=" * 56)
    print("T4 — Variance decomposition identity")
    print("=" * 56)
    if cov_yes is None or dec_yes is None:
        check("T4 skipped (T3 failed)", False)
        return
    try:
        tot_diag = jnp.diagonal(cov_yes,              axis1=1, axis2=2)
        ale_diag = jnp.diagonal(dec_yes["aleatoric"], axis1=1, axis2=2)
        epi_diag = jnp.diagonal(dec_yes["epistemic"], axis1=1, axis2=2)
        max_err  = float(jnp.max(jnp.abs(tot_diag - (ale_diag + epi_diag))))
        check("aleatoric + epistemic == full_cov diag (tol=1e-5)",
              max_err < 1e-5, f"max_abs_err={max_err:.2e}")
        check("aleatoric diag ≥ 0", bool(jnp.all(ale_diag >= 0)))
        check("epistemic diag ≥ 0", bool(jnp.all(epi_diag >= 0)))
    except Exception as e:
        check("T4 raised exception", False, str(e))
        traceback.print_exc()


# =============================================================================
# T5 — predict_members consistency
# =============================================================================
def test_t5(ens_yes, yp_yes):
    print()
    print("=" * 56)
    print("T5 — predict_members consistency")
    print("=" * 56)
    if ens_yes is None or yp_yes is None:
        check("T5 skipped (T3 failed)", False)
        return
    try:
        member_preds = ens_yes.predict_members(_X_te)
        member_means = jnp.stack([m[0] for m in member_preds], axis=0)  # (K, N, P)
        manual_mean  = jnp.mean(member_means, axis=0)
        max_diff     = float(jnp.max(jnp.abs(manual_mean - jnp.array(yp_yes))))
        check("mean(member means) == ensemble Y_pred (tol=1e-5)",
              max_diff < 1e-5, f"max_abs_diff={max_diff:.2e}")
        check(f"predict_members returns {N_ENSEMBLE} tuples",
              len(member_preds) == N_ENSEMBLE)
    except Exception as e:
        check("T5 raised exception", False, str(e))
        traceback.print_exc()


# =============================================================================
# T6 — OOD aleatoric uncertainty higher than in-domain
# =============================================================================
def test_t6(ens_yes):
    print()
    print("=" * 56)
    print("T6 — OOD uncertainty higher than in-domain")
    print("=" * 56)
    if ens_yes is None:
        check("T6 skipped (T3 failed)", False)
        return
    try:
        _, cov_id,  dec_id  = ens_yes.predict(_X_te,  return_var_decomposition=True)
        _, cov_ood, dec_ood = ens_yes.predict(_X_ood, return_var_decomposition=True)

        ale_id  = float(jnp.diagonal(dec_id ["aleatoric"], axis1=1, axis2=2).mean())
        ale_ood = float(jnp.diagonal(dec_ood["aleatoric"], axis1=1, axis2=2).mean())
        check("GP aleatoric uncertainty higher OOD than in-domain",
              ale_ood > ale_id,
              f"in-domain={ale_id:.4e}  OOD={ale_ood:.4e}")

        tot_id  = float(jnp.diagonal(cov_id,  axis1=1, axis2=2).mean())
        tot_ood = float(jnp.diagonal(cov_ood, axis1=1, axis2=2).mean())
        check("Total predictive variance higher OOD than in-domain",
              tot_ood > tot_id,
              f"in-domain={tot_id:.4e}  OOD={tot_ood:.4e}")
    except Exception as e:
        check("T6 raised exception", False, str(e))
        traceback.print_exc()


# =============================================================================
# T7 — Calibration: 1σ / 2σ empirical coverage
# =============================================================================
def test_t7(em_yes):
    print()
    print("=" * 56)
    print("T7 — Calibration: 1σ / 2σ coverage")
    print("=" * 56)
    if em_yes is None:
        check("T7 skipped (T2 failed)", False)
        return
    try:
        X_cal = jax.random.uniform(jax.random.PRNGKey(55), (500, _D))
        Y_cal = _true(X_cal)
        yp_cal, cov_cal = em_yes.predict(X_cal)
        std_cal = jnp.sqrt(jnp.clip(
            jnp.diagonal(cov_cal, axis1=1, axis2=2), 1e-12, None))
        z = jnp.abs(yp_cal - Y_cal) / (std_cal + 1e-12)

        cov_1s = float(jnp.mean(z < 1.0))
        cov_2s = float(jnp.mean(z < 2.0))
        # Generous bounds: a 2-PC model with M=30 on N=300 is intentionally
        # small to keep the test fast; exact calibration is not expected.
        check("1σ coverage ∈ [35%, 98%]  (ideal ~68%)",
              0.35 < cov_1s < 0.98, f"{cov_1s*100:.1f}%")
        check("2σ coverage ∈ [65%, 100%] (ideal ~95%)",
              0.65 < cov_2s <= 1.00, f"{cov_2s*100:.1f}%")
        check("2σ coverage > 1σ coverage", cov_2s > cov_1s)
        print(f"    1σ={cov_1s*100:.1f}%  2σ={cov_2s*100:.1f}%")
    except Exception as e:
        check("T7 raised exception", False, str(e))
        traceback.print_exc()


# =============================================================================
# T8 — API contract: return_var_decomposition=False returns 2-tuple
# =============================================================================
def test_t8(ens_yes, yp_yes):
    print()
    print("=" * 56)
    print("T8 — API contract: return_var_decomposition=False")
    print("=" * 56)
    if ens_yes is None or yp_yes is None:
        check("T8 skipped (T3 failed)", False)
        return
    try:
        out = ens_yes.predict(_X_te, return_var_decomposition=False)
        check("Returns 2-tuple when decomposition=False", len(out) == 2)
        yp2, _ = out
        max_diff = float(jnp.max(jnp.abs(jnp.array(yp2) - jnp.array(yp_yes))))
        check("Y_pred identical regardless of decomp flag (tol=1e-6)",
              max_diff < 1e-6, f"max_abs_diff={max_diff:.2e}")
    except Exception as e:
        check("T8 raised exception", False, str(e))
        traceback.print_exc()


# =============================================================================
# T9 — Y_err shape mismatch raises ValueError
# =============================================================================
def test_t9():
    print()
    print("=" * 56)
    print("T9 — Y_err shape mismatch raises ValueError")
    print("=" * 56)
    try:
        bad_err = jnp.ones((_N, _P + 1))   # wrong last dimension
        em_bad  = PCASparseGPEmulator(n_pc=N_PC, M=10, key=_KEY)
        raised  = False
        try:
            em_bad.fit(_X_tr, _Y_tr, bad_err,
                       **{**_FIT_KW, "verbose": False, "steps": 1})
        except ValueError:
            raised = True
        check("ValueError raised for mismatched Y_err shape", raised)
    except Exception as e:
        check("T9 raised exception", False, str(e))
        traceback.print_exc()


# =============================================================================
# T10 — EmulatorSparseGP high-level wrapper (pickle format)
# =============================================================================
def test_t10():
    print()
    print("=" * 56)
    print("T10 — EmulatorSparseGP high-level wrapper (end-to-end)")
    print("=" * 56)

    N_DESIGN = 60
    N_OBS    = 6
    N_PARAMS = 3

    try:
        # 1. Generate synthetic training data
        rng    = np.random.default_rng(42)
        design = _lhd(N_DESIGN, N_PARAMS, rng)

        data_dict = {}
        for i in range(N_DESIGN):
            values  = _hl_true_model(design[i], n_obs=N_OBS)
            errors  = 0.01 * np.abs(values) * (rng.standard_normal(N_OBS) + 1e-6)
            noisy   = values + errors
            abs_err = np.abs(errors) + 1e-6
            # obs shape: (2, N_OBS) — row 0 = values, row 1 = errors
            data_dict[str(i)] = {
                "parameter": design[i],
                "obs":       np.vstack([noisy, abs_err]),
            }

        with open(TRAINING_PKL, "wb") as f:
            pickle.dump(data_dict, f)
        check("Training pickle written", os.path.exists(TRAINING_PKL))

        # 2. Create and train (single emulator, very small scale)
        emu = EmulatorSparseGP(
            training_set_path=TRAINING_PKL,
            parameter_file=MODEL_PAR_FILE,
            n_pc=2,
            M=20,
            n_ensemble=1,
            logTrafo=False,
            max_rel_uncertainty_data=0.5,
        )
        emu.trainEmulatorAutoMask(
            steps=300, batch_size=32,
            kernel_lr=1e-3, variational_lr=1e-3, inducing_lr=3e-4,
            early_stopping=True, patience=30, verbose=False,
        )
        check("Emulator trained (emu_ attribute set)", hasattr(emu, "emu_") and emu.emu_ is not None)

        # 3. Predict at a few test points
        n_test    = 5
        test_par  = rng.uniform(size=(n_test, N_PARAMS))
        pred_mean, pred_cov = emu.predict(test_par, return_cov=True)
        check("predict Y_pred shape",
              pred_mean.shape == (n_test, pred_mean.shape[1]))
        check("predict full_cov shape",
              pred_cov.shape  == (n_test, pred_mean.shape[1], pred_mean.shape[1]))
        check("predict cov diag ≥ 0",
              bool(np.all(np.diagonal(pred_cov, axis1=1, axis2=2) >= 0)))

        # 4. testEmulatorErrors
        emu_pred, emu_pred_err, vali_data, vali_data_err = \
            emu.testEmulatorErrors(
                number_test_points=2,
                steps=300, batch_size=32,
                kernel_lr=1e-3, variational_lr=1e-3, inducing_lr=3e-4,
                early_stopping=True, patience=30, verbose=False,
            )
        check("testEmulatorErrors shapes match",
              emu_pred.shape == vali_data.shape)

    except Exception as e:
        check("T10 raised exception", False, str(e))
        traceback.print_exc()
    finally:
        if os.path.exists(TRAINING_PKL):
            os.remove(TRAINING_PKL)


# =============================================================================
# main
# =============================================================================
def main():
    print("=" * 56)
    print("  SparseGP emulator — fast synthetic tests")
    print("=" * 56)
    print(f"  N={_N}, D={_D}, P={_P}, M={M}, n_pc={N_PC}, "
          f"n_ensemble={N_ENSEMBLE}, steps≤{_FIT_KW['steps']}")
    print()

    em_yes = test_t2()        # T2 returns trained em_yes for T7
    test_t1()                 # T1 standalone smoke test

    t3_result = test_t3()     # T3 returns (ens_yes, yp_yes, cov_yes, dec_yes)
    if isinstance(t3_result, tuple) and len(t3_result) == 4:
        ens_yes, yp_yes, cov_yes, dec_yes = t3_result
    else:
        ens_yes = yp_yes = cov_yes = dec_yes = None

    test_t4(cov_yes, dec_yes)
    test_t5(ens_yes, yp_yes)
    test_t6(ens_yes)
    test_t7(em_yes)
    test_t8(ens_yes, yp_yes)
    test_t9()
    test_t10()

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 56)
    n_pass = sum(1 for r in _results if r[0] == "PASS")
    n_fail = sum(1 for r in _results if r[0] == "FAIL")
    print(f"Results: {n_pass} passed, {n_fail} failed "
          f"out of {len(_results)} checks")
    if n_fail:
        print("\nFailed checks:")
        for r in _results:
            if r[0] == "FAIL":
                print(f"  ✗  {r[1]}" + (f"  [{r[2]}]" if r[2] else ""))
    print("=" * 56)


if __name__ == "__main__":
    main()
