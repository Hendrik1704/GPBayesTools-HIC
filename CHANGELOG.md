# Changelog for GPBayesTools-HIC

## v3.0.0
Date: not released yet

New features:
- `Design` takes a `method` argument to choose between maximum projection Latin-hypercube designs (`'maxpro'`, R package MaxPro, default as before) and maximin designs (`'maximin'`, R package lhs).
- `run_mcmc` (emcee) and `run_MCMC_PTLMC` take an optional `seed` for reproducible chains.
- All emulators have `sample_y` to draw samples of the observables from the emulator uncertainty, independently at each parameter point. Before, it was only available for `Emulator`, where it sampled the GPs jointly at all points.
- `Emulator` and `EmulatorSparseGP` take an optional `seed` for reproducible training. For `EmulatorSparseGP`, the KMeans and Sobol initialisations of the inducing points are seeded as well, so that ensemble members start from different inducing points.
- The validation functions `testEmulatorErrors` and `testEmulatorErrorsWithTrainingPoints` can choose random test points (`random_points=True`, reproducible with `seed`) instead of the last points of the training data.
- The emulators share the base class `EmulatorBase` (`gpbayestools/emulator_base.py`), which implements loading and filtering the training data and the validation functions for all of them. It checks that the number of parameters in the parameter file matches the training data.

New emulators:
- Add the `EmulatorHETGPy` emulator, a wrapper for the heteroskedastic GPs of the [hetgpy](https://hetgpy.readthedocs.io) package combined with a PCA of the outputs.
- Add the `EmulatorSparseGP` emulator, a sparse variational GP emulator (SVGP) with PCA of the outputs, implemented with JAX. It can be trained as a single emulator or as an ensemble.

Changes that are not backward compatible:
- The package is renamed from `src` to `gpbayestools`, e.g. `from gpbayestools.emulator_BAND import EmulatorBAND`. It can be installed with `pip install .` (`pyproject.toml`, distribution name `gpbayestools-hic`). Emulators saved with older versions refer to the module names `src.*` and can be loaded with `gpbayestools.load_emulator`, which `Chain.loadEmulator` uses.
- Update the surmise package requirement from version 0.3.0 to version 1.0.0. This includes a major update of the PCSK emulator (surmise 0.4.0). Training now requires a global random number generator, which `EmulatorBAND` sets via `surmise.set_RNG` before each training. Use the new optional `seed` argument of `EmulatorBAND` for reproducible training. Emulators trained and saved with surmise 0.4.0 can still be loaded and give identical predictions.
- Remove the `parameterTrafoPCA` option of `Emulator` and `EmulatorBAND` (PCA transformation of the $\zeta/s(T)$, $\eta/s(\mu_B)$ and $\langle y_{\rm loss}\rangle(y_{\rm init})$ parameters). It was specific to the parametrization of one analysis. Use version v2.0.1 or older to reproduce results obtained with it.
- Remove the constant prior term of the unused `extra_std` parameter from the log-likelihood and log-posterior. The log-likelihood values and the pocoMC evidence (`logl`, `logz`) are shifted by +73.68 compared to older versions. Posterior samples are not affected.
- Raise an error in the MCMC likelihood if the covariance matrix is not positive definite, instead of returning NaN.
- The predicted covariance of all emulators is by default the uncertainty of the emulated model function, without the noise fitted to the training data. `Emulator` (WhiteKernel), `EmulatorBAND` (nugget of the surmise GPs) and `EmulatorHETGPy` (nugs) included this noise before, which counted the statistical noise of the simulations as emulator uncertainty in the MCMC. The noise can be included with `predict(..., include_noise=True)`, which the validation functions use, since they compare with noisy simulations. `Emulator.sample_y` has the same option.
- The GPs of `Emulator` are trained with `alpha=1e-8` (new constructor argument) instead of 0.1, and the lower bound of the fitted WhiteKernel noise is 1e-6 instead of 1e-2. With `alpha=0.1`, a fixed 10% of the variance of each PC was treated as noise, which overestimated the emulator uncertainty. Use `alpha=0.1` to reproduce older results.
- By default, `Emulator`, `EmulatorHETGPy` and `EmulatorSparseGP` also remove the statistical noise of the training data, estimated from their errors, from the covariance of the discarded principal components (truncation covariance). With noisy training data, this covariance mostly consists of that noise. With `include_noise=True`, the full truncation covariance is used as before.
- Each sampler writes its chain to its own file derived from `mcmc_path`, e.g. `./mcmc/chain_emcee.pkl`, `./mcmc/chain_pocoMC.pkl` and `./mcmc/chain_PTLMC.pkl` (`Chain.chain_path(sampler)`), instead of all samplers overwriting `./mcmc/chain.pkl`. `compute_log_likelihood_for_chain` takes the sampler of the chain as first argument (default: the last sampler run) and writes the result next to the chain file by default.
- `max_rel_uncertainty_data` is `None` (no filtering of the training data) by default in all emulators. Before, it was 0.1 in `Emulator`, `EmulatorBAND` and `EmulatorHETGPy`.
- Remove `Emulator.print_learning_curve()`, which did not use the settings of the trained emulator, and the unused `Emulator.getAvgTrainingDataRelError()`.
- `log_posterior` is the sum of `log_prior` and `log_likelihood`, i.e. it includes the constant log(1/prior volume). Points on the boundaries of the parameter ranges are inside the prior, as for pocoMC.
- The number of principal components is given by `npc` in all emulators except `EmulatorBAND`, where surmise chooses it: an int for the number of PCs or a float in (0, 1) for the fraction of the explained variance. `EmulatorSparseGP` used `n_pc`, and `EmulatorHETGPy` always used 99% of the variance (still the default).
- Remove `predict_test_emu_errors` from `EmulatorBAND` and `EmulatorHETGPy`, `predict` gives the same results.
- Remove the `extra_std` option from the `predict` functions of all emulators and from the MCMC. It was always 0 in the MCMC and ignored or treated differently by the emulators.
- `EmulatorSparseGP` handles `logTrafo` like the other emulators: by default, `predict` returns the mean and covariance in log space. The new option `exp_and_cov_diagonal=True` returns the predictions in the original scale, keeping the correlations between the observables. Previously, the predictions were always transformed back. Emulators saved with older versions keep the old behavior.
- The validation functions `testEmulatorErrors` and `testEmulatorErrorsWithTrainingPoints` no longer change the trained emulator. Previously, the emulator was left trained on the reduced training set. The argument `nTestPoints` of `Emulator` is renamed to `number_test_points` as in the other emulators, and `EmulatorSparseGP` also has `testEmulatorErrorsWithTrainingPoints`.
- All emulators raise a `ValueError` for observables <= 0 with `logTrafo=True`. Previously `log(|x|)` was used, so the sign was lost, and zeros became `log(1e-30)`.
- `Chain.loadEmulator` replaces previously loaded emulators instead of appending to them, and checks that the numbers of observables of the emulators add up to the number of experimental data points. The experimental data file must contain exactly one data set.
- The sparse GP emulator computes in 64-bit floats. Importing `gpbayestools.emulator_sparseGP` enables `jax_enable_x64` for the whole Python process.
- With mini-batches, the SVGP training returns the parameters with the best exponential moving average of the ELBO instead of the best single-batch ELBO, which selected the parameters of the luckiest batch.

Bug fixes:
- Fix `Emulator.predict` with `return_cov=True`, which failed with NumPy 2 (`np.array(..., copy=False)`). The same applies to the log prior, likelihood and posterior in `mcmc.py` for list inputs.
- `Emulator.outputPCAvsParam()` no longer overwrites the training data with standardized values.
- Fix the covariance of `Emulator` with `perform_no_PCA=True`, which was returned in standardized units instead of observable units.
- Fix `Emulator.sample_y`: the PCs are sampled independently and reproducibly with `random_state`, and the samples are returned in physical space if `exp_and_cov_diagonal` is set.
- Fix the `PCGPwM` option of `EmulatorBAND`, which previously trained a `PCGPwImpute` emulator.
- Raise a `ValueError` in `EmulatorBAND` when an unknown emulator method is requested. Previously the error was never raised.
- `compute_log_likelihood_for_chain` now also works for chains from pocoMC.
- Saved `EmulatorHETGPy` emulators now contain the trained GP models, so that loading gives exactly the same predictions. Previously the models were refitted from some of their hyperparameters after loading. Emulators saved with the older format can still be loaded, but their predictions can differ from the trained emulator.
- The output standardization and PCA of `EmulatorHETGPy` are fitted to the training points only. Previously, they included the points held out in the validation.
- Add the covariance of the PCs discarded by the output PCA to the `EmulatorHETGPy` covariance. This increases the predicted emulator uncertainty.
- The SVGP training now keeps the parameters that belong to the best ELBO, and the NaN recovery restarts from parameters with a finite ELBO.
- Add the variance of the PCs discarded by surmise to the `EmulatorBAND` covariance. surmise's `covx()` does not contain it, which underestimated the emulator uncertainty, strongly for PCGP with `logTrafo=True`.
- `EmulatorBAND.predict` with `exp_and_cov_diagonal=True` works for a single 1D parameter vector.
- `Emulator` limits `npc` to the number of available PCs instead of failing when fewer observables or training points than `npc` are given.
- `Emulator.outputPCAvsParam()` no longer refits the scaler and PCA of the trained emulator, which changed later predictions.
- Raise a `ValueError` in `Emulator` for unknown kernel types.
- Training points with NaN or infinite observables are discarded when loading the training data in all emulators. Previously they passed the relative-error filter.
- The SVGP training caps observation errors that are more than 1e5 times larger than the spread of the training data. They overflowed in float32 and made the training fail. `nan_patience` now counts consecutive NaN steps only.
- `EmulatorSparseGP` accepts the `verbose_members` training argument also for a single emulator.
- MCMC with emcee: fix crashes for `nsteps < 10`, give a clear error for `nburnsteps < 2`, and check an existing chain before continuing it (pocoMC chains or a different number of walkers raise a `ValueError`; `nwalkers` is taken from the chain if not given). The burn-in restart only uses distinct points with finite probability.
- PTLMC draws at least one starting point per chain.
- `compute_log_likelihood_for_chain` creates its output directory before the computation.
- Empty lines in parameter files are skipped, and keys are stripped.
- Observables that are exactly zero are ignored in the relative-error filter of the training data, instead of discarding the whole training point.
- `generate_posterior_clusters.py` no longer overwrites its input file when the chain file name does not contain `.pkl`.
- Unpickling or copying an untrained `EmulatorHETGPy` no longer trains it.
- The PTLMC sampler is updated from the surmise 0.2.1 code to surmise 1.0.0, keeping the modifications of this package. This fixes NaN perturbations of the starting points when the inverse Hessian of the optimizer is not positive definite, and the random numbers are drawn from a generator seeded with `seed`, without changing numpy's global random state.
- PTLMC no longer swaps chains with the same temperature, which shuffled the temperature-1 walkers in every iteration, so that the saved walker traces are continuous. The sampled distribution is unchanged.
- A continued emcee chain starts from the last walker positions of the previous run (saved as `last_position` in the chain file) instead of the last thinned sample.
- Refitting a `PCASparseGPEmulator` uses the requested number or fraction of PCs again instead of the number found in the previous fit.
- Importing the package no longer creates a `./cache` directory; it is only created for the Latin-hypercube designs.
- Fix the imports in the `EmulatorValidation` notebook and several docstrings of the sparse GP emulator (`patience`, `include_noise`).
- The default seed of `Design` is an integer from the current time (stored in `Design.seed`). The float timestamp was truncated by R, so the printed seed was not the one used.
- Fix the chain loading and the error bars in the `ClosureTest` notebook, and the log-likelihood output in the `RunBayesianAnalysis` notebook.

Documentation:
- Document in the README and the docstrings that emulators trained with `logTrafo=True` return predictions in log space by default, so the experimental data must be log-transformed by the user.

Development:
- GitHub workflows run the tests (Python 3.11 and 3.12) and format the code with ruff on pushes and pull requests to `main` and `devel`. On pull requests, the formatting is only checked.
- The code is formatted with ruff.

Tests:
- Add tests in `tests/` for all emulators, the emulator base class and the MCMC module, which can be run with `python -m pytest tests`. The MCMC tests compare the samples of emcee, PTLMC and pocoMC with an analytically known posterior.

[Link to diff from previous version](https://github.com/Hendrik1704/GPBayesTools-HIC/compare/v2.0.1...v3.0.0)

## v2.0.1
Date: 2025-12-03

- Make the number of beta steps in the pocoMC sampler adjustable via the `n_max_steps` argument. Previously it was fixed to `ndim`, now it is `n_max_steps*ndim`.

[Link to diff from previous version](https://github.com/Hendrik1704/GPBayesTools-HIC/compare/v2.0.0...v2.0.1)

## v2.0.0
Date: 2025-09-17

- Fix the standard behavior of the pocoMC sampler to resample the samples (i.e., make them have equal weights). The resampled points can be used just like you would do with MCMC samples. In older versions the 'weights' from the chain have to be used to generate the posterior corner plots. Thanks to @hejajama for pointing this out.
- Update to surmise 0.3.0. This update in our `predict` function is not backward compatible, since the handling of the covariance matrices has changed. `fpredcov = gp.covx().transpose((1, 0, 2))` has to be used when using older versions of surmise (<=0.2.1) or emulators trained with that version. The new version (0.3.0) returns the covariance matrices in the expected shape `(theta, ndim, ndim)`, so `fpredcov = gp.covx()` is sufficient.
- Increase the default `n_steps` for the pocoMC sampler to `2*ndim` (twice the number of dimensions). This should improve the exploration of the posterior distribution. The default value was `ndim` in previous versions.

[Link to diff from previous version](https://github.com/Hendrik1704/GPBayesTools-HIC/compare/v1.2.1...v2.0.0)

## v1.2.1
Date: 2025-08-06

- Fix a bug in the example script `generate_posterior_clusters.py` where the number of samples argument was not handled correctly when set to 'None'.

[Link to diff from previous version](https://github.com/Hendrik1704/GPBayesTools-HIC/compare/v1.2.0...v1.2.1)

## v1.2.0
Date: 2025-07-14

- Delete unused dependencies and add `requirements.txt`
- Running the scripts parsing arguments from the terminal is no longer supported
- Move all examples to the `examples` directory
- Add a Latin Hypercube Sampler script (`R` with `lhs` required)
- Remove PTMCMC since the code does not parallelize properly and ptemcee is no longer maintained
- Fix range of pocoMC uniform prior distributions. This does not cause problems with previous results, since the `log_likelihood` evaluates to `-np.inf` for values outside the prior range. Thanks @wenbin1501110084 for pointing this out.
- Implement option to switch off PCA transformation in the scikit GP emulator wrapper
- Implement option for 'Matern' kernel in the scikit GP emulator wrapper ('RBF' is the default)
- Implement option to only use the diagonal of the covariances in all GP emulator wrappers (only when logTrafo is set to True)
- Add a script to sample posterior clusters from the posterior chain file after a Bayesian inference run.
- Add possibility to use pocoMC with a custom prior distribution class

[Link to diff from previous version](https://github.com/Hendrik1704/GPBayesTools-HIC/compare/v1.1.0...v1.2.0)

## v1.1.0
Date: 2024-07-24

- pocoMC sampler added
- Optional parameter for error handling of the training points in the GP emulators

[Link to diff from previous version](https://github.com/Hendrik1704/GPBayesTools-HIC/compare/v1.0.0...v1.1.0)

## v1.0.0
Date: 2024-05-13

**[First public version of GPBayesTools-HIC ](https://github.com/Hendrik1704/GPBayesTools-HIC/releases/tag/v1.0.0)**