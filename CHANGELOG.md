## v2.0.0
Date: 2025-09-17

- Fix the standard behavior of the pocoMC sampler to resample the samples (i.e., make them have equal weights). The resampled points can be used just like you would do with MCMC samples. In older versions the 'weights' from the chain have to be used to generate the posterior corner plots. Thanks to @hejajama for pointing this out.
- Update to surmise 0.3.0. This update in our `predict` function is not backward compatible, since the handling of the covariance matrices has changed. `fpredcov = gp.covx().transpose((1, 0, 2))` has to be used when using older versions of surmise (<=0.2.1) or emulators trained with that version. The new version (0.3.0) returns the covariance matrices in the expected shape `(theta, ndim, ndim)`, so `fpredcov = gp.covx()` is sufficient.

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