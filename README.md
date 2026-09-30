# GPBayesTools-HIC

Gaussian Process Bayesian Toolkit with Monte Carlo Sampler Integration for Heavy Ion Collisions

This toolkit implements a wrapper for Gaussian Process (GP) emulators and Monte Carlo (MC) samplers used in 
high-energy heavy-ion simulations.

The following wrappers for GP emulators are currently included:
- Scikit Learn GP emulator wrapper
- PCGP and PCSK wrapper for the GPs implemented in the [surmise](https://github.com/bandframework/surmise) package of the [BAND](https://bandframework.github.io/) Collaboration
- Heteroskedastic GP wrapper for the [hetgpy](https://hetgpy.readthedocs.io) package
- Sparse variational GP emulator (single emulator or ensemble) implemented with [JAX](https://github.com/jax-ml/jax)

The following wrappers for MC sampling are included:
- MCMC wrapper for the [emcee](https://github.com/dfm/emcee) package
- [PTLMC](https://github.com/bandframework/surmise) from the surmise package (Parallel Tempering Langevin Monte Carlo)
- [pocoMC](https://github.com/minaskar/pocomc) Preconditioned Monte Carlo method for accelerated Bayesian inference

We recommend to use the `pocoMC` sampler.

## Example: full Bayesian workflow

[`examples/full_workflow`](examples/full_workflow/README.md) goes through a complete Bayesian study
with the fit of HERA deep-inelastic scattering data with two parameters from
[JHEP 04 (2026) 185](https://doi.org/10.1007/JHEP04(2026)185): parameter design, training data,
training and validation of all emulators, and closure tests with pocoMC. It compares the
accuracy and the calibration of the emulators and the posteriors they give.

## Emulator uncertainty

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
training data. By default, `EmulatorSklearn`, `EmulatorHetGP` and `EmulatorSparseGP` remove this noise,
estimated from the statistical errors of the training data, from the truncation covariance; with
`include_noise=True` the full truncation covariance is used. `EmulatorBAND` uses the truncation
variance of surmise, which is zero for PCSK.

## Emulators trained on the log of the observables

All emulators have a `log_trafo` option to train them on the logarithm of the observables.
This requires positive observables, and the relative statistical errors of the training data
are used as errors in log space.
What the `predict` function returns depends on the `exp_and_cov_diagonal` option:

- `exp_and_cov_diagonal=False` (default): the mean and the covariance are returned in log space.
  The experimental data used with the emulator in the MCMC (`BayesianAnalysis`) are used as they are given,
  so they have to be log-transformed by the user as well: `log(y)` for the values and the
  relative errors `sigma/y` for the errors.
- `exp_and_cov_diagonal=True`: the predictions are transformed back to the original scale,
  i.e. `exp(mean)` and the covariance `(sigma * exp(mean))^2`, and the experimental data
  are used in the original scale.
  The covariance is diagonal for `EmulatorSklearn`, `EmulatorBAND` and `EmulatorHetGP`, while
  `EmulatorSparseGP` keeps the correlations between the observables.

If emulators with different settings are combined in one `BayesianAnalysis`, the experimental data of each
emulator must be given in the scale of its predictions.

## Latin Hypercube Sampling

There is also a script to generate Latin Hypercube Design parameter files.
An example how to use it is given in the `examples` directory in the `generate_LHD_Bayes.py` script.
This requires a file specifying the parameter ranges, see for example `examples/modelDesign_example.txt`.

## Posterior Cluster Sampling

The `generate_posterior_clusters.py` script in the `examples` directory sorts the samples of a
pocoMC chain file (`chain_pocomc.pkl`) by their log-likelihood and clusters the most likely
samples with k-means. The cluster centers are written to `cluster_centers.txt` in the current
directory (one parameter set per column) and can be used as parameter sets for model runs.

## Installation

The package can be installed with pip from the root directory of the repository:

```
pip install .
```

pocoMC depends on PyTorch, which pip installs with CUDA support by default (several GB). Without
a GPU, install the CPU version of PyTorch first:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use `pip install -e ".[dev]"` for an editable installation with the dependencies for the tests
and the code formatting, and `pip install ".[examples]"` for the notebooks in `examples/`. The modules are then imported from `gpbayestools`, e.g.
`from gpbayestools.emulator_band import EmulatorBAND`.
Emulators and chains saved with versions < 3.0.0 cannot be loaded; retrain the emulators with
the current version.

## Logging

The modules report their progress with the `logging` module (loggers `gpbayestools.<module>`).
To see the messages, configure logging in your script or notebook, e.g.

```
import logging
logging.basicConfig(level=logging.INFO)
```

## Requirements

The dependencies are listed in `pyproject.toml` and, for use without installation, in the
`requirements.txt` file.
The `design.py` module additionally requires R with the MaxPro or lhs package.

## Tests

The tests in the `tests` directory can be run with

```
python -m pytest tests
```

:exclamation: The jupyter notebooks are just meant as examples for how to use the emulators and samplers and analyze the output.
Paths and data files need the proper input formats.