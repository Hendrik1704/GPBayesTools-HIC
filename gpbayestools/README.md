# User guide

This guide describes the modules of `gpbayestools` and how they are used together. The complete
list of arguments of each class and function is in its docstring, e.g. `help(EmulatorBAND)`. A
complete worked example is in [`examples/full_workflow`](../examples/full_workflow/README.md).

| Module | Content |
|---|---|
| `gpbayestools.design` | `Design`: Latin-hypercube parameter designs |
| `gpbayestools.emulator_sklearn` | `EmulatorSklearn`: GPs of scikit-learn |
| `gpbayestools.emulator_band` | `EmulatorBAND`: PCGP and PCSK of surmise (BAND collaboration) |
| `gpbayestools.emulator_hetgp` | `EmulatorHetGP`: heteroscedastic GPs of hetGPy |
| `gpbayestools.emulator_sparse_gp` | `EmulatorSparseGP`: sparse variational GPs (JAX), with the core classes in `gpbayestools.svgp` |
| `gpbayestools.emulator_base` | `EmulatorBase`: common base class of the emulators |
| `gpbayestools.bayesian_analysis` | `BayesianAnalysis`: likelihood and the samplers emcee, PTLMC and pocoMC |
| `gpbayestools` | `load_emulator`, `parse_model_parameter_file` |

## Workflow

1. Write a parameter file with the parameter ranges and generate a design (`Design`).
2. Run the model at the design points and collect the output in a training data file.
3. Train an emulator on the training data, validate it and save it.
4. Sample the posterior for the experimental data with `BayesianAnalysis`.

```python
import dill

from gpbayestools.bayesian_analysis import BayesianAnalysis
from gpbayestools.emulator_band import EmulatorBAND

# 3. train and validate an emulator, and save it
emu = EmulatorBAND("training_data.pkl", "parameters.txt", method="PCGP", seed=1)
emu.train_emulator_auto_mask()
pred, pred_err, data, data_err = emu.test_emulator_errors(
    n_test_points=50, random_points=True, seed=1
)
with open("emulator.dill", "wb") as f:
    dill.dump(emu, f)

# 4. sample the posterior with pocoMC
analysis = BayesianAnalysis(
    "exp_data.pkl", "parameters.txt", mcmc_path="mcmc/chain.pkl"
)
analysis.load_emulators(["emulator.dill"])
analysis.run_pocomc(seed=1)
samples = analysis.chain  # also written to mcmc/chain_pocomc.pkl
```

Several emulators, e.g. for different sets of observables, are combined by passing several files
to `load_emulators`; their observables are concatenated in this order.

## Input files

**Parameter file.** One line `name: label, min, max` per parameter; text after `#` is a comment.
The label is used in plots and must not contain commas.

```
# name: label, min, max
Qs0: $Q_{s0}$ [GeV], 0.1, 0.8
lambda: $\lambda$, 0.1, 0.5
```

**Training data.** A pickle file with a dictionary
`{event_id: {"parameter": array (n_parameters,), "obs": array (2, n_obs)}}`. The first row of
`"obs"` holds the values of the observables, the second row their uncertainties. The design
points are sorted by the integer value of `event_id`. The parameters must be in the order of the
parameter file.

**Experimental data.** A pickle file with a dictionary with exactly one entry,
`{name: {"obs": array (2, n_obs)}}`, with the values and the uncertainties in the same order as the
observables of the emulators. The covariance is diagonal with the squared uncertainties.

## Emulators

All emulators take the paths of the training data and of the parameter file, all other arguments
are keyword-only. They have the same interface:

- `train_emulator(event_mask)` trains on the training points selected by a boolean mask,
  `train_emulator_auto_mask()` on all points. `EmulatorSparseGP` passes further keyword arguments
  to the training (e.g. `steps`, `early_stopping`, see `PCASparseGPEmulator.fit`).
- `predict(X, return_cov=True, include_noise=False)` returns the mean of shape (n, n_obs) and the
  covariance of shape (n, n_obs, n_obs) at the parameter points `X`.
- `sample_y(X, n_samples, seed)` draws samples of the observables from the emulator uncertainty.
- `test_emulator_errors(n_test_points)` trains without the last (or random) training points and
  returns the predictions and errors and the training data at these points.
  `test_emulator_errors_with_training_points` predicts at the training points instead. The trained
  emulator is not changed.

| Class | Method | Main arguments | Use |
|---|---|---|---|
| `EmulatorSklearn` | GP (scikit-learn) for each principal component (PC) of the standardized observables | `npc` (number of PCs or fraction of the variance), `n_restarts`, `alpha`, `perform_no_pca` | general purpose |
| `EmulatorBAND` | surmise: `method="PCGP"`, `"PCSK"`, `"PCGPwM"`, `"PCGPwImpute"` | `method` (surmise chooses the PCs) | PCSK uses the uncertainties of the training data as noise of the simulations; PCGPwM and PCGPwImpute allow missing observables (NaN) |
| `EmulatorHetGP` | heteroscedastic GP (hetGPy) for each PC | `npc` | noise that varies over the parameter space |
| `EmulatorSparseGP` | sparse variational GP (JAX) for each PC, optionally an ensemble | `npc`, `n_inducing`, `n_ensemble`, `bootstrap`, `init_strategy` | large designs (thousands of points) |

Common arguments of all emulators:

- `log_trafo`: train on the logarithm of the observables (see below).
- `exp_and_cov_diagonal`: with `log_trafo`, return the predictions in the original scale.
- `max_rel_uncertainty_data`: discard training points with a larger relative uncertainty of any
  observable.
- `errors_are_noise`: whether the uncertainties of the training data are statistical noise of the
  simulations (default) or, with `False`, e.g. an assigned model uncertainty of noise-free
  simulations (see below).
- `seed`: reproducible training (`EmulatorSklearn`, `EmulatorBAND`, `EmulatorSparseGP`).

Training points with non-finite values or uncertainties are discarded (except for PCGPwM and
PCGPwImpute, which treat them as missing).

### Emulator uncertainty

The `predict` function of all emulators returns by default the uncertainty of the emulated
model function, which is used in the MCMC, since the experimental data are compared with the
expectation value of the model. With `include_noise=True`, the noise that the GPs fitted to the
(statistically noisy) training data is included as well, i.e. the uncertainty of a new noisy
simulation. The validation functions `test_emulator_errors` and
`test_emulator_errors_with_training_points` compare with such simulations and include the noise.
For `EmulatorSparseGP`, `include_noise=True` also includes the statistical errors of the training
data propagated through the PCA (`include_obs_noise`, which follows `include_noise` by default).
The PCSK emulator of `EmulatorBAND` models the noise with the statistical errors of the training
data, so `include_noise=True` adds their mean variance.

The covariance also contains the variance of the principal components that are not emulated
(truncation). With noisy training data, these components contain the statistical noise of the
training data. By default, `EmulatorSklearn`, `EmulatorHetGP` and `EmulatorSparseGP` remove this
noise, estimated from the statistical errors of the training data, from the truncation
covariance; with `include_noise=True` the full truncation covariance is used. `EmulatorBAND` uses
the truncation variance of surmise, which is zero for PCSK.

If the simulations are noise-free and the uncertainties of the training data are not statistical
errors (e.g. an assigned model uncertainty), use `errors_are_noise=False`: the full truncation
covariance is then used, and `EmulatorSparseGP` is trained without the uncertainties as
observation noise. Otherwise the emulator uncertainty is underestimated. `EmulatorBAND` ignores
this option.

### Emulators trained on the log of the observables

All emulators have a `log_trafo` option to train them on the logarithm of the observables.
This requires positive observables, and the relative statistical errors of the training data
are used as errors in log space. For observables that span orders of magnitude, this usually
makes the emulators more accurate.
What the `predict` function returns depends on the `exp_and_cov_diagonal` option:

- `exp_and_cov_diagonal=False` (default): the mean and the covariance are returned in log space.
  The experimental data used with the emulator in the MCMC (`BayesianAnalysis`) are used as they
  are given, so they have to be log-transformed by the user as well: `log(y)` for the values and
  the relative errors `sigma/y` for the errors.
- `exp_and_cov_diagonal=True`: the predictions are transformed back to the original scale,
  i.e. `exp(mean)` and the covariance `(sigma * exp(mean))^2`, and the experimental data
  are used in the original scale.
  The covariance is diagonal for `EmulatorSklearn`, `EmulatorBAND` and `EmulatorHetGP`, while
  `EmulatorSparseGP` keeps the correlations between the observables.

If emulators with different settings are combined in one `BayesianAnalysis`, the experimental
data of each emulator must be given in the scale of its predictions.

### Saving and loading

Trained emulators are saved with `dill.dump` and loaded with `gpbayestools.load_emulator` (or
`BayesianAnalysis.load_emulators`). Emulators and chains saved with versions < 3.0.0 cannot be
loaded; retrain the emulators with the current version.

## Bayesian analysis

`BayesianAnalysis(exp_data_path, parameter_file, mcmc_path="./mcmc/chain.pkl")` evaluates the
Gaussian likelihood of the experimental data with the covariance of the data plus the emulator
covariance, with a uniform prior in the parameter ranges of the parameter file.

| Method | Sampler | Output file |
|---|---|---|
| `run_pocomc(n_effective, n_active, n_total, seed, pool, prior, ...)` | [pocoMC](https://github.com/minaskar/pocomc), preconditioned Monte Carlo (recommended) | `chain_pocomc.pkl`: samples, log-likelihood, log prior, log evidence |
| `run_emcee(n_steps, n_burn_steps, n_walkers, n_thin, seed)` | [emcee](https://github.com/dfm/emcee), affine-invariant ensemble; an existing chain is continued | `chain_emcee.pkl` |
| `run_ptlmc(n_steps, n_walkers, n_temps, max_temp, seed)` | PTLMC of [surmise](https://github.com/bandframework/surmise), parallel tempering Langevin Monte Carlo | `chain_ptlmc.pkl` |

The chain files are written next to `mcmc_path` (`BayesianAnalysis.chain_path(sampler)`), and the
last chain is in `analysis.chain`. `compute_log_likelihood_for_chain(sampler)` computes the
log-likelihood of all samples of a chain. `run_pocomc` can evaluate the likelihood in parallel
with `pool` (a number of processes or a pool object, e.g. of mpi4py). The pocoMC evidence `logz`
is not normalized (see the docstring of `run_pocomc`), so only compare it between runs with the
same experimental data.

## Recommended settings for a large heavy-ion analysis

The following settings are a starting point for a typical heavy-ion analysis with about 23 model
parameters, where the training data are event averages of the observables over about 1000 events
per design point. They are not tuned for a specific model; check them with the validation metrics
and a closure test.

**Training data.** The uncertainty in the second row of `"obs"` must be the statistical
uncertainty of the event average, i.e. the standard deviation over the events of the centrality
bin divided by $\sqrt{N_\mathrm{events}}$ (or a jackknife error for cumulants and ratios), not
the standard deviation over the events itself. The emulators treat it as the uncertainty of the
stored value (PCSK, `errors_are_noise`, `max_rel_uncertainty_data`). The analysis scripts of
[iEBE-MUSIC](https://github.com/chunshen1987/iEBE-MUSIC/tree/dev/analysisKit) compute the
uncertainties in this way.

**Design.** About 1000 design points of a maximum projection Latin hypercube (`Design`, method
`"maxpro"`), plus separate validation points or a part of the design kept for the validation.

**Emulators.** One emulator per group of observables (e.g. identified particle yields, mean
transverse momenta, flow coefficients), since the groups need different settings:

| Observables | `log_trafo` | `max_rel_uncertainty_data` |
|---|---|---|
| multiplicities, e.g. $dN/dy$, $dN/d\eta$ | `True` | 0.1 |
| mean transverse momenta $\langle p_T \rangle$ | `False` | 0.1 |
| flow coefficients $v_n$ | `False` | 0.2 (larger for noisier observables) |

Use `EmulatorBAND(method="PCSK", seed=...)`: PCSK models the statistical uncertainties of the event
averages as noise of the simulations, and the training is fast. Keep `errors_are_noise=True` (the
default), since the event averages have statistical noise. Only positive observables can be
log-transformed; flow coefficients and correlators, which can be close to zero or negative, are
emulated in the original scale.

```python
groups = [  # (training data, log_trafo, max_rel_uncertainty_data)
    ("PbPb5020_pid_dNdy.pkl", True, 0.1),
    ("PbPb5020_pid_meanpT.pkl", False, 0.1),
    ("PbPb5020_charged_vn.pkl", False, 0.2),
]
for i, (training_file, log_trafo, max_rel) in enumerate(groups):
    emu = EmulatorBAND(
        training_file,
        "parameters.txt",
        method="PCSK",
        log_trafo=log_trafo,
        max_rel_uncertainty_data=max_rel,
        seed=1,
    )
    emu.train_emulator_auto_mask()
    with open(f"emulator_{i}.dill", "wb") as f:
        dill.dump(emu, f)
```

Validate each emulator before the inference, e.g. with
`test_emulator_errors(n_test_points=100, random_points=True, seed=1)`, and compute the RMS relative
error and the honesty (see [`examples/full_workflow`](../examples/full_workflow/README.md)).

**Experimental data.** One file with the observables of all groups concatenated in the order of
the emulators passed to `load_emulators`. The data of log-transformed groups must be
log-transformed as well: $\log y$ for the values and the relative uncertainties $\sigma/y$ for the
uncertainties.

**pocoMC.** For $d$ parameters:

| Argument | Value | For $d = 23$ |
|---|---|---|
| `n_effective` | $500\,d$ (fewer, e.g. $300\,d$, can be enough; check that the posterior does not change) | 11500 |
| `n_active` | $0.3 \times$ `n_effective` | 3450 |
| `n_prior` | $2\,(\lfloor$`n_effective`/`n_active`$\rfloor)\,$`n_active`, the default of pocoMC (the default of `run_pocomc` is 2000) | 20700 |
| `n_total` | 150000 if the samples are used e.g. to train a machine learning model, fewer (e.g. 10000–20000) for histograms of the posterior | |
| `n_evidence` | 7500 (0 if the evidence is not needed) | |
| `pool` | the number of available cores; `n_active` should be a multiple of it | |

```python
n_dim = 23
n_effective = 500 * n_dim
n_active = int(0.3 * n_effective)
analysis = BayesianAnalysis(
    "exp_data.pkl", "parameters.txt", mcmc_path="mcmc/chain.pkl"
)
analysis.load_emulators([f"emulator_{i}.dill" for i in range(len(groups))])
analysis.run_pocomc(
    n_effective=n_effective,
    n_active=n_active,
    n_prior=2 * (n_effective // n_active) * n_active,
    n_total=150000,
    n_evidence=7500,
    pool=48,
    seed=42,
)
```

With a pool, set the environment variable `OMP_NUM_THREADS=1` before numpy is imported (e.g. in
the job script), so that the processes do not compete for the cores, and
`RDMAV_FORK_SAFE=1` if the processes fail to start (e.g. on clusters with InfiniBand). Before
the analysis of the measured data, run a closure test with the model output at a validation
point as pseudo-data (see [`examples/full_workflow`](../examples/full_workflow/README.md)).

## Latin-hypercube designs

`Design(parameter_file, n_points=500, seed=None, method="maxpro")` generates a maximum
projection (`"maxpro"`, R package MaxPro) or maximin (`"maximin"`, R package lhs) Latin-hypercube
design in the parameter ranges; R with the package must be installed. `design.write_files(basedir)`
writes one input file per design point, and `np.asarray(design)` is the design array. An example is
[`examples/generate_LHD_Bayes.py`](../examples/generate_LHD_Bayes.py) with the parameter file
[`examples/modelDesign_example.txt`](../examples/modelDesign_example.txt). Designs are cached in
`cache/lhs/` in the directory given by the environment variable `WORKDIR` (default: the current
directory).

## Posterior cluster sampling

The script [`examples/generate_posterior_clusters.py`](../examples/generate_posterior_clusters.py)
sorts the samples of a pocoMC chain file (`chain_pocomc.pkl`) by their log-likelihood and clusters
the most likely samples with k-means. The cluster centers are written to `cluster_centers.txt` in
the current directory (one parameter set per column) and can be used as parameter sets for model
runs:

```
python generate_posterior_clusters.py <chain_file> <number_of_most_likely_samples> <number_of_clusters>
```

## Logging

The modules report their progress with the `logging` module (loggers `gpbayestools.<module>`).
To see the messages, configure logging in your script or notebook, e.g.

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Further examples

The notebooks in [`examples/`](../examples) show the training and validation of emulators
(`EmulatorTraining.ipynb`, `EmulatorValidation.ipynb`), a Bayesian analysis
(`RunBayesianAnalysis.ipynb`, `PlotMCMC.ipynb`), a closure test (`ClosureTest.ipynb`) and a
sensitivity analysis (`SensitivityAnalysis.ipynb`). Their paths and data files have to be adapted
to your input files.
