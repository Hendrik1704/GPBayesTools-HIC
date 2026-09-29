# Changelog for GPBayesTools-HIC

## v3.0.0
Date: not released yet

New features:
- `Design` takes a `method` argument to choose between maximum projection Latin-hypercube designs (`'maxpro'`, R package MaxPro, default as before) and maximin designs (`'maximin'`, R package lhs).

New emulators:
- Add the `EmulatorHETGPy` emulator, a wrapper for the heteroskedastic GPs of the [hetgpy](https://hetgpy.readthedocs.io) package combined with a PCA of the outputs.
- Add the `EmulatorSparseGP` emulator, a sparse variational GP emulator (SVGP) with PCA of the outputs, implemented with JAX. It can be trained as a single emulator or as an ensemble.

Changes that are not backward compatible:
- Update the surmise package requirement from version 0.3.0 to version 1.0.0. This includes a major update of the PCSK emulator (surmise 0.4.0). Training now requires a global random number generator, which `EmulatorBAND` sets via `surmise.set_RNG` before each training. Use the new optional `seed` argument of `EmulatorBAND` for reproducible training. Emulators trained and saved with surmise 0.4.0 can still be loaded and give identical predictions.
- Remove the `parameterTrafoPCA` option of `Emulator` and `EmulatorBAND` (PCA transformation of the $\zeta/s(T)$, $\eta/s(\mu_B)$ and $\langle y_{\rm loss}\rangle(y_{\rm init})$ parameters). It was specific to the parametrization of one analysis. Use version v2.0.1 or older to reproduce results obtained with it.
- Remove the constant prior term of the unused `extra_std` parameter from the log-likelihood and log-posterior. The log-likelihood values and the pocoMC evidence (`logl`, `logz`) are shifted by +73.68 compared to older versions. Posterior samples are not affected.
- Raise an error in the MCMC likelihood if the covariance matrix is not positive definite, instead of returning NaN.

Bug fixes:
- Fix `Emulator.predict` with `return_cov=True`, which failed with NumPy 2 (`np.array(..., copy=False)`). The same applies to the log prior, likelihood and posterior in `mcmc.py` for list inputs.
- `Emulator.outputPCAvsParam()` and `Emulator.print_learning_curve()` no longer overwrite the training data with standardized values.
- Fix the covariance of `Emulator` with `perform_no_PCA=True`, which was returned in standardized units instead of observable units.
- Fix `Emulator.sample_y`: the PCs are sampled independently and reproducibly with `random_state`, and the samples are returned in physical space if `exp_and_cov_diagonal` is set.
- Fix the `PCGPwM` option of `EmulatorBAND`, which previously trained a `PCGPwImpute` emulator.
- Raise a `ValueError` in `EmulatorBAND` when an unknown emulator method is requested. Previously the error was never raised.
- `compute_log_likelihood_for_chain` now also works for chains from pocoMC.
- Saved `EmulatorHETGPy` emulators now contain the trained GP models, so that loading gives exactly the same predictions. Previously the models were refitted from some of their hyperparameters after loading. Emulators saved with the older format can still be loaded, but their predictions can differ from the trained emulator.
- Add the covariance of the PCs discarded by the output PCA to the `EmulatorHETGPy` covariance. This increases the predicted emulator uncertainty.
- `EmulatorSparseGP.predict` accepts a scalar `extra_std` for several parameter points.
- The SVGP training now keeps the parameters that belong to the best ELBO, and the NaN recovery restarts from parameters with a finite ELBO.
- Add the variance of the PCs discarded by surmise to the `EmulatorBAND` covariance. surmise's `covx()` does not contain it, which underestimated the emulator uncertainty, strongly for PCGP with `logTrafo=True`.
- `EmulatorBAND.predict` with `exp_and_cov_diagonal=True` works for a single 1D parameter vector.
- `Emulator` limits `npc` to the number of available PCs instead of failing when fewer observables or training points than `npc` are given.
- `Emulator.outputPCAvsParam()` and `Emulator.print_learning_curve()` no longer refit the scaler and PCA of the trained emulator, which changed later predictions.
- Raise a `ValueError` in `Emulator` for unknown kernel types.
- Training points with NaN or infinite observables are discarded when loading the training data in all emulators. Previously they passed the relative-error filter.
- The SVGP training caps observation errors that are more than 1e5 times larger than the spread of the training data. They overflowed in float32 and made the training fail. `nan_patience` now counts consecutive NaN steps only.
- `EmulatorSparseGP` accepts the `verbose_members` training argument also for a single emulator.
- MCMC with emcee: fix crashes for `nsteps < 10`, give a clear error for `nburnsteps < 2`, and check an existing chain before continuing it (pocoMC chains or a different number of walkers raise a `ValueError`; `nwalkers` is taken from the chain if not given). The burn-in restart only uses distinct points with finite probability.
- PTLMC draws at least one starting point per chain.
- `compute_log_likelihood_for_chain` creates its output directory before the computation.
- Empty lines in parameter files are skipped, and keys are stripped.
- Fix the chain loading and the error bars in the `ClosureTest` notebook, and the log-likelihood output in the `RunBayesianAnalysis` notebook.

Tests:
- Add tests for the hetGP and sparse GP emulators in `tests/`, which can be run with pytest.

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