# Changelog for GPBayesTools-HIC

## v3.0.0
Date: not released yet

New features:
- `Design` takes a `method` argument to choose between maximum projection Latin-hypercube designs (`'maxpro'`, R package MaxPro, default as before) and maximin designs (`'maximin'`, R package lhs).
- `run_emcee` (emcee) and `run_ptlmc` take an optional `seed` for reproducible chains.
- All emulators have `sample_y` to draw samples of the observables from the emulator uncertainty, independently at each parameter point. Before, it was only available for `EmulatorSklearn`, where it sampled the GPs jointly at all points. The samples are reproducible with the `seed` argument and in the scale of the predictions (log-normal samples with `exp_and_cov_diagonal=True`).
- `EmulatorSklearn` and `EmulatorSparseGP` take an optional `seed` for reproducible training. For `EmulatorSparseGP`, the KMeans and Sobol initialisations of the inducing points are seeded as well, so that ensemble members start from different inducing points.
- The validation functions `test_emulator_errors` and `test_emulator_errors_with_training_points` can choose random test points (`random_points=True`, reproducible with `seed`) instead of the last points of the training data.
- The emulators share the base class `EmulatorBase` (`gpbayestools/emulator_base.py`), which implements loading and filtering the training data and the validation functions for all of them. It checks that the number of parameters in the parameter file matches the training data.
- `EmulatorBAND` with `PCGPwM` or `PCGPwImpute` keeps training points with missing observables: non-finite observables (or observables with non-finite errors) are passed to surmise as missing (NaN), and only points without any finite observable are discarded. Before, these points were discarded, so both methods behaved like PCGP. surmise no longer removes points and observables with at least 80% missing values.
- New constructor argument `errors_are_noise` of all emulators (default `True`). Set it to `False` if the simulations are noise-free and the uncertainties of the training data are e.g. an assigned model uncertainty: the covariance of the discarded principal components is then used in full for predictions of the model function instead of removing the uncertainties from it as noise, and `EmulatorSparseGP` trains without them as observation noise. `EmulatorBAND` accepts it but ignores it.

New emulators:
- Add the `EmulatorHetGP` emulator, a wrapper for the heteroskedastic GPs of the [hetgpy](https://hetgpy.readthedocs.io) package combined with a PCA of the outputs.
- Add the `EmulatorSparseGP` emulator, a sparse variational GP emulator (SVGP) with PCA of the outputs, implemented with JAX. It can be trained as a single emulator or as an ensemble.

Changes that are not backward compatible:
- Version 3.0.0 is a clean break: emulators saved with older versions cannot be loaded (retrain them with this version), and emcee chain files of older versions cannot be continued. The old module, class, method and argument names listed below are not available as aliases.
- Consistent names: the modules are `emulator_sklearn` (was `emulator`), `emulator_band` (was `emulator_BAND`), `emulator_hetgp` (was `emulator_hetGPy`), `emulator_sparse_gp` (was `emulator_sparseGP`) and `bayesian_analysis` (was `mcmc`); the classes `EmulatorSklearn` (was `Emulator`), `EmulatorBAND`, `EmulatorHetGP` (was `EmulatorHETGPy`), `EmulatorSparseGP` and `BayesianAnalysis` (was `Chain`). Methods and arguments use snake_case: `train_emulator` (was `trainEmulator`), `train_emulator_auto_mask` (was `trainEmulatorAutoMask`), `test_emulator_errors` (was `testEmulatorErrors`), `test_emulator_errors_with_training_points` (was `testEmulatorErrorsWithTrainingPoints`), `output_pca_vs_param` (was `outputPCAvsParam`), `load_emulators` (was `loadEmulator`), `run_emcee` (was `run_mcmc`), `run_ptlmc` (was `run_MCMC_PTLMC`), `run_pocomc` (was `run_pocoMC`), `log_trafo` (was `logTrafo`), `n_restarts` (was `nrestarts`), `perform_no_pca` (was `perform_no_PCA`), `n_inducing` (was `M` of `EmulatorSparseGP`), and for the samplers `n_steps` (was `nsteps`), `n_burn_steps` (was `nburnsteps`), `n_walkers` (was `nwalkers`), `n_thin` (was `nthin`), `n_temps` (was `ntemps`), `max_temp` (was `maxtemp`) and `n_start_parameters` (was `nstartparameters`). The sampler names of the chain files are lowercase (`chain_emcee.pkl`, `chain_pocomc.pkl`, `chain_ptlmc.pkl`). The attributes follow the scikit-learn convention: constructor arguments under their own name, results of the training with a trailing underscore (e.g. `npc_`, `emu_`).
- The constructor arguments of `BayesianAnalysis` are `exp_data_path` (was `expdata_path`, default `./exp_data.pkl` instead of `./exp_data.dat`) and `parameter_file` (was `model_parafile`), and its attributes `exp_data`, `exp_data_cov` (were `expdata`, `expdata_cov`), `emulators` (was `emuList`), `labels` (was `label`), `param_min`, `param_max` (were `min`, `max`) and `prior_volume` (was `prior_volume_`).
- `Design` takes `parameter_file` (was `parfile`), and its attributes are `design_type` (was `type`) and `param_min`, `param_max` (were `min`, `max`). The design functions are `generate_maxpro_lhs` and `generate_maximin_lhs` (was `generate_lhs`).
- `EmulatorSklearn.sample_y` (now `EmulatorBase.sample_y` for all emulators) takes `seed` instead of `random_state` and samples the points independently, see the new features.
- The package is renamed from `src` to `gpbayestools`, e.g. `from gpbayestools.emulator_band import EmulatorBAND`. It can be installed with `pip install .` (`pyproject.toml`, distribution name `gpbayestools-hic`).
- Update the surmise package requirement from version 0.3.0 to version 1.0.0. This includes a major update of the PCSK emulator (surmise 0.4.0). Training now requires a global random number generator, which `EmulatorBAND` sets via `surmise.set_RNG` before each training. Use the new optional `seed` argument of `EmulatorBAND` for reproducible training: each training with the same seed and training points gives the same emulator.
- Remove the `parameterTrafoPCA` option of `EmulatorSklearn` and `EmulatorBAND` (PCA transformation of the $\zeta/s(T)$, $\eta/s(\mu_B)$ and $\langle y_{\rm loss}\rangle(y_{\rm init})$ parameters). It was specific to the parametrization of one analysis. Use version v2.0.1 or older to reproduce results obtained with it.
- Remove the constant prior term of the unused `extra_std` parameter from the log-likelihood and log-posterior. The log-likelihood values and the pocoMC evidence (`logl`, `logz`) are shifted by +73.68 compared to older versions. Posterior samples are not affected. As before, the log-likelihood is not normalized, so `logl` and `logz` are larger than the normalized values by n/2·log(2π) for n experimental data points; compare `logz` only between runs with the same experimental data.
- Raise an error in the MCMC likelihood if the covariance matrix is not positive definite, instead of returning NaN.
- The predicted covariance of all emulators is by default the uncertainty of the emulated model function, without the noise fitted to the training data. `EmulatorSklearn` (WhiteKernel), `EmulatorBAND` (nugget of the surmise GPs) and `EmulatorHetGP` (nugs) included this noise before, which counted the statistical noise of the simulations as emulator uncertainty in the MCMC. The noise can be included with `predict(..., include_noise=True)`, which the validation functions use, since they compare with noisy simulations. `sample_y` has the same option.
- The GPs of `EmulatorSklearn` are trained with `alpha=1e-8` (new constructor argument) instead of 0.1, and the lower bound of the fitted WhiteKernel noise is 1e-6 instead of 1e-2. With `alpha=0.1`, a fixed 10% of the variance of each PC was treated as noise, which overestimated the emulator uncertainty. Use `alpha=0.1` to reproduce older results.
- By default, `EmulatorSklearn`, `EmulatorHetGP` and `EmulatorSparseGP` also remove the statistical noise of the training data, estimated from their errors, from the covariance of the discarded principal components (truncation covariance). With noisy training data, this covariance mostly consists of that noise. With `include_noise=True`, the full truncation covariance is used as before.
- `include_obs_noise` of `EmulatorSparseGP.predict` is `None` by default and then follows `include_noise`, so `include_noise=True` gives the uncertainty of a new noisy simulation as in the other emulators, which the validation functions use. The core classes `PCASparseGPEmulator` and `PCASparseGPEnsemble` keep `include_obs_noise=False`.
- Each sampler writes its chain to its own file derived from `mcmc_path`, e.g. `./mcmc/chain_emcee.pkl`, `./mcmc/chain_pocomc.pkl` and `./mcmc/chain_ptlmc.pkl` (`BayesianAnalysis.chain_path(sampler)`), instead of all samplers overwriting `./mcmc/chain.pkl`. `compute_log_likelihood_for_chain` takes the sampler of the chain as first argument (default: the last sampler run) and writes the result next to the chain file by default.
- `max_rel_uncertainty_data` is `None` (no filtering of the training data) by default in all emulators. Before, it was 0.1 in `EmulatorSklearn`, `EmulatorBAND` and `EmulatorHetGP`.
- Remove `EmulatorSklearn.print_learning_curve()`, which did not use the settings of the trained emulator, and the unused `EmulatorSklearn.getAvgTrainingDataRelError()`.
- Clearer log messages: the loading of the training data ends with one summary of the loaded and discarded points, discarding points with non-finite observables is a warning, and the validation functions log the number and choice of the test points. The emulators log one message at the start and at the end of the training, with the numbers of training points, PCs and GPs. The sparse GP emulator no longer logs separator lines, logs diagnostics such as the truncation covariance at the DEBUG level, and always logs the jitter increases and learning rate reductions of the NaN recovery, also with `verbose=False`. `BayesianAnalysis` logs what it loaded (model parameters, experimental data points, emulators and their observables), one start message per sampler with its settings, the pocoMC evidence in one line, the progress of `log_likelihood_point_by_point` about 10 times, and a warning when the PTLMC or pocoMC chain file is overwritten. PTLMC logs its progress about 10 times instead of every 100 steps, and `Design` logs the generated design, the seed chosen without a given seed, and the number of written files.
- `log_posterior` is the sum of `log_prior` and `log_likelihood`, i.e. it includes the constant log(1/prior volume). Points on the boundaries of the parameter ranges are inside the prior, as for pocoMC.
- The number of principal components is given by `npc` in all emulators except `EmulatorBAND`, where surmise chooses it: an int for the number of PCs or a float in (0, 1) for the fraction of the explained variance. `EmulatorSparseGP` used `n_pc`, and `EmulatorHetGP` always used 99% of the variance (still the default).
- Remove `predict_test_emu_errors` from `EmulatorBAND` and `EmulatorHetGP`, `predict` gives the same results.
- Remove the `extra_std` option from the `predict` functions of all emulators and from the MCMC. It was always 0 in the MCMC and ignored or treated differently by the emulators.
- `EmulatorSparseGP` handles `log_trafo` like the other emulators: by default, `predict` returns the mean and covariance in log space. The new option `exp_and_cov_diagonal=True` returns the predictions in the original scale, keeping the correlations between the observables. Previously, the predictions were always transformed back.
- The validation functions `test_emulator_errors` and `test_emulator_errors_with_training_points` no longer change the trained emulator. Previously, the emulator was left trained on the reduced training set. The argument for the number of test points is `n_test_points` in all emulators (was `nTestPoints` in `EmulatorSklearn` and `number_test_points` in the other emulators), and `EmulatorSparseGP` also has `test_emulator_errors_with_training_points`.
- All emulators raise a `ValueError` for observables <= 0 with `log_trafo=True`. Previously `log(|x|)` was used, so the sign was lost, and zeros became `log(1e-30)`.
- `BayesianAnalysis.load_emulators` replaces previously loaded emulators instead of appending to them, and checks that the numbers of observables of the emulators add up to the number of experimental data points. The experimental data file must contain exactly one data set.
- The sparse GP emulator computes in 64-bit floats. Importing `gpbayestools.emulator_sparse_gp` enables `jax_enable_x64` for the whole Python process.
- With mini-batches, the SVGP training returns the parameters with the best exponential moving average of the ELBO instead of the best single-batch ELBO, which selected the parameters of the luckiest batch.
- The spread between the members of the sparse GP ensemble (`PCASparseGPEnsemble`) is the covariance of the equal-weight mixture (divided by K instead of K-1), as the law of total variance in the docstring, which gives a slightly smaller ensemble uncertainty for few members. The keys of the variance decomposition are `within_members` and `between_members` (were `aleatoric` and `epistemic`; the GP posterior variance is not aleatoric).
- The seed of `run_pocomc` is called `seed` like for the other samplers (was `random_state`) and is `None` by default (was 42). pocoMC sets it as the seed of numpy's global random number generator, so the default 42 made later code that uses this generator, e.g. `run_emcee` without a seed, deterministic.
- The emulators have no default paths for the training data and the parameter file (the defaults `"."` and `"ABCD.txt"` could not work), and all other constructor arguments are keyword-only, since their order differed between the emulators, e.g. `EmulatorSklearn(training_file, parameter_file, npc=10)`. In the same way, `BayesianAnalysis(exp_data_path, parameter_file, mcmc_path=...)` requires the paths of the experimental data and the parameter file (the defaults `./exp_data.pkl` and `./model.dat` are removed), and the arguments of `Design` except `parameter_file` are keyword-only.
- The length scales of the RBF kernel of `EmulatorSklearn` have the same bounds as the Matern kernel, 1e-3 to 1e5 times the parameter ranges (were 0.1 to 100 times the ranges, which the fits often reached). This changes the trained RBF emulators.
- The fitted attributes of the sparse GP classes `PCASparseGPEmulator` and `PCASparseGPEnsemble` have a trailing underscore like in the other emulators (`pca_`, `params_`, `jitter_`, `Xm_`, `Xs_`, `Ym_`, `Ys_`, `pc_mean_`, `pc_std_`, `n_train_` (was `N_train`), `training_history_`, `members_`, `training_histories_`). `n_pc` is the argument and `n_pc_` the fitted number of PCs, also for the ensemble; before, `n_pc` of `PCASparseGPEmulator` was replaced by the fitted number.
- The sparse GP uses independent random subkeys of its key for the inducing-point initialization, the seeds of numpy/scikit-learn, the mini-batches and the bootstrap samples of the ensemble members, which were drawn from the same key before. The trained emulators differ from older versions for the same seed.
- The `seed` of `run_emcee` no longer changes numpy's global random number generator: its state is restored after the emcee sampler is created, as for the other samplers.
- `run_emcee` uses the vectorized emcee sampler (`vectorize=True`) instead of passing `BayesianAnalysis` as a dummy pool, and the `BayesianAnalysis.map` method is removed. The chains are the same.
- `PCASparseGPEmulator.fit` returns the fitted emulator like `PCASparseGPEnsemble.fit` (was the training history, which is stored in `training_history_`).
- Counts use the `n_` prefix: the emulator attributes `n_ev`, `n_obs` and `n_parameters` (were `nev`, `nobs`, `nparameters`), `BayesianAnalysis.n_dim` and `n_obs` (were `ndim`, `nobs`), and `Design(..., n_points=...)`, `Design.n_dim` and `generate_maxpro_lhs`/`generate_maximin_lhs(n_points, n_dim, seed)` (were `npoints`, `ndim`).

Bug fixes:
- Fix `EmulatorSklearn.predict` with `return_cov=True`, which failed with NumPy 2 (`np.array(..., copy=False)`). The same applies to the log prior, likelihood and posterior of `BayesianAnalysis` for list inputs.
- `EmulatorSklearn.output_pca_vs_param()` no longer overwrites the training data with standardized values.
- Fix the covariance of `EmulatorSklearn` with `perform_no_pca=True`, which was returned in standardized units instead of observable units.
- Fix the `PCGPwM` option of `EmulatorBAND`, which previously trained a `PCGPwImpute` emulator.
- Raise a `ValueError` in `EmulatorBAND` when an unknown emulator method is requested. Previously the error was never raised.
- `compute_log_likelihood_for_chain` now also works for chains from pocoMC.
- Saved `EmulatorHetGP` emulators now contain the trained GP models, so that loading gives exactly the same predictions. Previously the models were refitted from some of their hyperparameters after loading.
- The output standardization and PCA of `EmulatorHetGP` are fitted to the training points only. Previously, they included the points held out in the validation.
- Add the covariance of the PCs discarded by the output PCA to the `EmulatorHetGP` covariance. This increases the predicted emulator uncertainty.
- The SVGP training now keeps the parameters that belong to the best ELBO, and the NaN recovery restarts from parameters with a finite ELBO.
- Add the variance of the PCs discarded by surmise to the `EmulatorBAND` covariance. surmise's `covx()` does not contain it, which underestimated the emulator uncertainty, strongly for PCGP with `log_trafo=True`.
- `EmulatorBAND.predict` with `exp_and_cov_diagonal=True` works for a single 1D parameter vector.
- `EmulatorSklearn` limits `npc` to the number of available PCs instead of failing when fewer observables or training points than `npc` are given.
- `EmulatorSklearn.output_pca_vs_param()` no longer refits the scaler and PCA of the trained emulator, which changed later predictions.
- Raise a `ValueError` in `EmulatorSklearn` for unknown kernel types.
- Training points with NaN or infinite observables or statistical errors are discarded when loading the training data in all emulators (an infinite error carries no information, a NaN error is unknown), except for the surmise methods for missing observables (see the new features). Previously non-finite observables passed the relative-error filter and non-finite errors were used.
- The SVGP training caps observation errors that are more than 1e5 times larger than the spread of the training data. They overflowed in the float32 computations of earlier versions and made the training fail. `nan_patience` now counts consecutive NaN steps only.
- `EmulatorSparseGP` accepts the `verbose_members` training argument also for a single emulator.
- MCMC with emcee: fix crashes for `n_steps < 10` and check an existing chain before continuing it (pocoMC chains or a different number of walkers raise a `ValueError`; `n_walkers` is taken from the chain if not given). The burn-in restart only uses distinct points with finite probability.
- PTLMC draws at least one starting point per chain.
- `compute_log_likelihood_for_chain` creates its output directory before the computation.
- Empty lines in parameter files are skipped, and keys are stripped.
- Observables that are exactly zero are ignored in the relative-error filter of the training data, instead of discarding the whole training point.
- `generate_posterior_clusters.py` no longer overwrites its input file when the chain file name does not contain `.pkl`.
- Unpickling or copying an untrained `EmulatorHetGP` no longer trains it.
- The PTLMC sampler is updated from the surmise 0.2.1 code to surmise 1.0.0, keeping the modifications of this package. This fixes NaN perturbations of the starting points when the inverse Hessian of the optimizer is not positive definite, and the random numbers are drawn from a generator seeded with `seed`, without changing numpy's global random state.
- A continued emcee chain starts from the last walker positions of the previous run (saved as `last_position` in the chain file) instead of the last thinned sample.
- Importing the package no longer creates a `./cache` directory; it is only created for the Latin-hypercube designs.
- The default seed of `Design` is an integer from the current time (stored in `Design.seed`). The float timestamp was truncated by R, so the printed seed was not the one used.
- `EmulatorSklearn.predict` accepts a single parameter point as a 1D array or a list, like the other emulators, and computes only the variances of the GPs at the points instead of their full covariance between all points. This makes the prediction for many points (e.g. all MCMC walkers) much faster.
- `EmulatorHetGP` computes the output PCA with the exact (full) SVD like `EmulatorSklearn`. The default of scikit-learn could choose a randomized, approximate solver for more than 500 observables, which made the PCs change between trainings.
- `EmulatorHetGP` warns if a hetGP model gives non-finite predictions at the training points (failed fit), and logs the messages that hetgpy prints to stdout at the DEBUG level.
- The validation functions check that `n_test_points` leaves at least 2 training points, and `test_emulator_errors` that there is at least one test point. Before, `n_test_points=0` failed after the training, and a single training point gave NaN errors.
- `EmulatorSklearn.npc_` is the number of trained GPs also with `perform_no_pca=True` (the number of observables) instead of the requested `npc`, and the fitted attributes `npc_`, `scaler_` and `pca_` are only set by the training.
- Training points that are discarded by `max_rel_uncertainty_data` no longer raise the `log_trafo` error for values <= 0.
- The sparse GP emulator predicts with the jitter with which the returned best parameters were trained. After a NaN recovery, which increases the jitter, it used the increased jitter, which changed the predictions. `training_history_["jitter"]` is the jitter of the returned parameters, the new `"jitter_final"` the jitter at the end of the training. A warning is logged if no training step had a finite ELBO.
- The sparse GP emulator caps an int `npc` at the number of available PCs with a warning like the other emulators instead of failing in the PCA, and uses the exact (full) SVD for the PCA. The options `M`, `init_strategy` and `print_every` are checked before the emulator is modified. A warning is logged if a bootstrap sample has fewer unique points than inducing points, and if `bootstrap=True` is ignored because `n_ensemble=1`.
- The `pool` argument of `run_pocomc` is used: with a pool, the likelihood is evaluated point by point in the pool instead of vectorized, which pocoMC does without using the pool. A pool created from an integer is closed after the run, and the analysis with its emulators is sent to its processes once instead of with every task. The warnings about overwritten chain files of pocoMC and PTLMC are logged before the run, and the docstring says that the seed of pocoMC seeds numpy's global random number generator.
- `run_emcee` checks `n_steps`, `n_thin` and `n_burn_steps` before the sampling instead of failing after the burn-in or the whole run (`n_thin=0` failed after the run without writing the chain). The thinning is stored in the chain file and must match when a chain is continued; `n_thin=None` (new default) uses 10 for a new chain and the stored thinning otherwise. A warning is logged if `n_steps` is not a multiple of `n_thin`.
- `parse_model_parameter_file` raises a `ValueError` with the file and line number for malformed lines, non-numeric ranges, min >= max and parameters that are defined twice. Before, a duplicate parameter silently replaced the first one, and min == max gave an infinite log prior.
- The perturbation of the optimized PTLMC starting points uses the inverse Hessian as covariance (the surmise code used a rotated covariance) and stops after a few step reductions, so that it cannot loop forever when the log posterior at the optimum is not finite.
- `Design.write_files` accepts the base directory as a string. `Design` raises a clear error for MaxPro designs with one parameter (R crashed) and includes the error message of R when R fails. The docstring says that a validation design needs a different seed than the main design.
- `BayesianAnalysis` raises a clear error when the likelihood is evaluated without loaded emulators and for non-finite experimental data values or errors (NaN errors were set to 0 before), and warns when `compute_log_likelihood_for_chain` overwrites its output file. `log_likelihood_point_by_point` uses `log_likelihood` for each point instead of a copy of its code.
- The initial walker positions of `run_emcee` are drawn before the emcee sampler is created. Before, emcee copied the state of numpy's global random number generator first, so that its first random numbers were the ones of the initial positions. The chains differ from earlier versions for the same seed. Continuing a chain with a different number of parameters raises a clear error.
- Training an `EmulatorBAND` keeps the warning filters of the process, which surmise resets with `warnings.resetwarnings()`.
- The emulators check that the mask of `train_emulator` is a boolean array of shape (nev,). An integer array of 0 and 1 was used as indices of the training points by `EmulatorSklearn`, `EmulatorBAND` and `EmulatorSparseGP`. `EmulatorBAND.predict` accepts a single point as a 1D array or a list.
- For the PCSK emulator of `EmulatorBAND`, `include_noise=True` includes the noise of the simulations, the mean variance of the statistical errors of the training data. PCSK models the noise with these errors, which surmise does not include in the predictive variance, so the uncertainty of new noisy simulations (and the errors of the validation functions) was underestimated by a large factor.
- A failed `fit` of `PCASparseGPEmulator` or `PCASparseGPEnsemble` restores the previous state. Before, a refit that failed, e.g. because of an invalid `Y_err` or a NaN loss, left a partly updated emulator, which could predict wrong values without an error (e.g. a PCA with more components than trained GPs, or the untrained initial parameters). `n_pc`, `steps`, `batch_size` and the shape of `Y_err` are checked before the fit.
- In `PCASparseGPEmulator.predict`, the observation noise of the training data in the discarded PCA directions (truncation covariance) is included with `include_obs_noise`, like the noise in the retained directions, instead of with `include_noise`, which only controls the fitted nugget. `EmulatorSparseGP` is not affected, since `include_obs_noise` follows `include_noise` by default. The mean and covariance are always numpy arrays.
- The chain files and the log-likelihood file of `BayesianAnalysis` are written via a temporary file, so that an interrupted write does not destroy an existing emcee chain, which contains all previous runs. New files get the default permissions of the process and existing files keep theirs.
- `run_pocomc(pool=1)` runs without a pool like `pool=None` instead of failing in pocoMC.
- `parse_model_parameter_file` also rejects empty parameter names and infinite ranges, and its docstring says that labels must not contain commas.
- `compute_log_likelihood_for_chain` evaluates the log-likelihood in batches of 1000 points instead of point by point, which is much faster, since the emulators predict many points at once.
- The members of a bootstrap ensemble of the sparse GP use the noise-free part of the truncation covariance of all training data, like the truncation covariance itself, instead of one computed from the errors of their bootstrap sample.

Documentation:
- Document in the README and the docstrings that emulators trained with `log_trafo=True` return predictions in log space by default, so the experimental data must be log-transformed by the user.
- The `PlotMCMC` notebook reads the chain files of the samplers (`chain_<sampler>.pkl`), also handles the two-dimensional pocoMC samples and uses raw strings for the LaTeX labels. The `SensitivityAnalysis` notebook works with NumPy 2.
- Fix the imports in the `EmulatorValidation` notebook and several docstrings of the sparse GP emulator (`patience`, `include_noise`).
- Fix the chain loading and the error bars in the `ClosureTest` notebook, and the log-likelihood output in the `RunBayesianAnalysis` notebook.
- New example `examples/full_workflow` of a complete Bayesian study with the HERA DIS fit of JHEP 04 (2026) 185: parameter design, training data, training and validation of all emulators (RMS error and honesty, with and without log transformation) and closure tests with pocoMC, with a README that shows the figures. The main README links to it.

Development:
- The core sparse GP code (`PCASparseGPEmulator`, `PCASparseGPEnsemble`) is in `gpbayestools/svgp.py` (still importable from `emulator_sparse_gp`), and the PTLMC sampler from surmise in `gpbayestools/ptlmc.py`.
- Consistent code style: module loggers (`logging.getLogger(__name__)`) instead of the root logger and `print`, importing the package no longer configures the logging of the program, NumPy-style docstrings for all public classes and functions, and lint rules (ruff) that are checked in the workflow.
- GitHub workflows run the tests (Python 3.11 and 3.12) and format the code with ruff on pushes and pull requests to `main` and `devel`. On pull requests, the formatting is only checked.
- The code is formatted with ruff.
- The minimum versions of numpy (2.0) and scipy (1.14) are the ones required by jax, and the new `examples` extra installs matplotlib and jupyter for the notebooks. The README explains how to install the CPU version of PyTorch.

Tests:
- Add tests in `tests/` for all emulators, the emulator base class and the MCMC module, which can be run with `python -m pytest tests`. The MCMC tests compare the samples of emcee, PTLMC and pocoMC with an analytically known posterior.
- Add tests of `Design`, with the R call replaced by a random design.
- The test that the validation includes the noise runs for all emulators, and the guards for surmise < 1.0.0, which is not supported, are removed.

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