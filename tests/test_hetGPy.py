"""
Synthetic end-to-end test for the hetGPy emulator.

Steps:
  1. Generate a Latin-Hypercube design in 3 parameters.
  2. Evaluate a known analytical model (sum-of-sines + polynomial) to
     produce 20 observables per design point, with small Gaussian noise
     to mimic statistical errors.
  3. Write the data in the pickle format expected by EmulatorHETGPy.
  4. Train the emulator on the synthetic data.
  5. Save the emulator to a .pkl file and reload it.
  6. Call predict on the reloaded emulator and compare to the true model.
"""

import logging
import os
import pickle
import sys

import dill
import numpy as np

# Resolve the project root (one level up from tests/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)
from src.emulator_hetGPy import EmulatorHETGPy

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="[%(levelname)s] %(message)s")

# ── Configuration ────────────────────────────────────────────────────
N_DESIGN = 80        # number of training design points
N_OBS = 20            # number of observables
N_PARAMS = 3          # number of model parameters
REL_NOISE = 0.01      # relative statistical noise on each observable
SEED = 42
MODEL_PAR_FILE = os.path.join(PROJECT_ROOT, "modelDesign_test_3par.txt")
TRAINING_PKL = os.path.join(PROJECT_ROOT, "tests", "synthetic_training_data.pickle")
EMULATOR_PKL = os.path.join(PROJECT_ROOT, "tests", "emulator_synthetic.pkl")


# ── Analytical ground-truth model ────────────────────────────────────
def true_model(params, n_obs=N_OBS):
    """
    Map a parameter vector (alpha, beta, gamma) in [0,1]^3
    to *n_obs* observables using a smooth, nonlinear function that
    is easy to emulate.

    f_k(alpha, beta, gamma) =
        (1 + alpha) * sin(pi * k / n_obs * beta)
        + gamma^2 * cos(2*pi * k / n_obs)
        + 0.5 * alpha * beta * k / n_obs

    Parameters
    ----------
    params : array-like, shape (3,)
    n_obs  : int

    Returns
    -------
    values : ndarray, shape (n_obs,)
    """
    alpha, beta, gamma = params
    k = np.arange(n_obs, dtype=float)
    frac = k / n_obs
    values = (
        (1.0 + alpha) * np.sin(np.pi * frac * beta)
        + gamma**2 * np.cos(2.0 * np.pi * frac)
        + 0.5 * alpha * beta * frac
    )
    # Shift so all values are strictly positive (needed for logTrafo)
    values += 3.0
    return values


# ── 1.  Generate a Latin-Hypercube design ────────────────────────────
def latin_hypercube(n_samples, n_dim, rng):
    """Simple random LHD in [0, 1]^n_dim."""
    result = np.zeros((n_samples, n_dim))
    for d in range(n_dim):
        perm = rng.permutation(n_samples)
        result[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
    return result


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("  hetGPy emulator — synthetic end-to-end test")
    print("=" * 60)

    # ── Generate design & training data ──────────────────────────────
    print(f"\n[1] Generating {N_DESIGN} design points with {N_PARAMS} "
          f"parameters and {N_OBS} observables ...")
    design = latin_hypercube(N_DESIGN, N_PARAMS, rng)

    data_dict = {}
    for i in range(N_DESIGN):
        values = true_model(design[i])
        errors = REL_NOISE * np.abs(values) * rng.standard_normal(N_OBS)
        noisy_values = values + errors
        abs_errors = np.abs(errors)
        # obs has shape (2, n_obs): row 0 = values, row 1 = stat errors
        obs = np.vstack([noisy_values, abs_errors])
        data_dict[str(i)] = {
            "parameter": design[i],
            "obs": obs,
        }

    with open(TRAINING_PKL, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"    Training data saved to {TRAINING_PKL}")

    # ── 2.  Create & train emulator ──────────────────────────────────
    print(f"\n[2] Creating EmulatorHETGPy and training ...")
    emu = EmulatorHETGPy(
        training_set_path=TRAINING_PKL,
        parameter_file=MODEL_PAR_FILE,
        logTrafo=False,
        max_rel_uncertainty_data=0.5,
    )
    emu.trainEmulatorAutoMask()
    print(f"    Training complete — {emu.npc} principal components retained.")

    # ── 3.  Save emulator ────────────────────────────────────────────
    print(f"\n[3] Saving emulator to {EMULATOR_PKL} ...")
    with open(EMULATOR_PKL, "wb") as f:
        dill.dump(emu, f)
    print(f"    Saved ({os.path.getsize(EMULATOR_PKL) / 1024:.1f} KB).")

    # ── 4.  Reload emulator ──────────────────────────────────────────
    print(f"\n[4] Reloading emulator from {EMULATOR_PKL} ...")
    with open(EMULATOR_PKL, "rb") as f:
        emu_loaded = dill.load(f)
    print("    Loaded successfully (GP models re-trained automatically).")

    # ── 5.  Predict and compare ──────────────────────────────────────
    print(f"\n[5] Predicting at 10 new random test points ...")
    n_test = 10
    test_params = rng.uniform(size=(n_test, N_PARAMS))
    pred_mean, pred_cov = emu_loaded.predict(test_params, return_cov=True)
    # pred_mean shape: (n_test, n_obs)
    # pred_cov  shape: (n_test, n_obs, n_obs)

    # Compute the true values at the test points
    true_vals = np.array([true_model(p) for p in test_params])  # (n_test, n_obs)

    # Relative errors
    rel_err = np.abs(pred_mean - true_vals) / np.abs(true_vals)
    mean_rel_err = rel_err.mean()
    max_rel_err = rel_err.max()

    print(f"    Mean relative error : {mean_rel_err:.4f}")
    print(f"    Max  relative error : {max_rel_err:.4f}")

    # Check that predicted uncertainties are sensible
    pred_std = np.array([np.sqrt(np.diag(pred_cov[k])) for k in range(n_test)])
    pull = (pred_mean - true_vals) / (pred_std + 1e-30)
    rms_pull = np.sqrt(np.mean(pull**2))
    print(f"    RMS pull            : {rms_pull:.2f}  (ideal ≈ 1)")

    # ── 6.  Also run the built-in validation ─────────────────────────
    print(f"\n[6] Running built-in testEmulatorErrors (leave-last-5-out) ...")
    emu_val = EmulatorHETGPy(
        training_set_path=TRAINING_PKL,
        parameter_file=MODEL_PAR_FILE,
        logTrafo=False,
        max_rel_uncertainty_data=0.5,
    )
    emu_pred, emu_pred_err, vali_data, vali_data_err = \
        emu_val.testEmulatorErrors(number_test_points=5)
    val_rel_err = np.abs(emu_pred - vali_data) / (np.abs(vali_data) + 1e-30)
    print(f"    Validation mean relative error: {val_rel_err.mean():.4f}")
    print(f"    Validation max  relative error: {val_rel_err.max():.4f}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if mean_rel_err < 0.05:
        print("  PASS — emulator reproduces the synthetic model well.")
    else:
        print("  WARNING — mean relative error > 5%. Consider increasing "
              "N_DESIGN or checking the implementation.")
    print("=" * 60)

    # Cleanup temporary files
    os.remove(TRAINING_PKL)
    os.remove(EMULATOR_PKL)


if __name__ == "__main__":
    main()
