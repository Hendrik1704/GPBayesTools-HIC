## v1.2.0
Date: 2024-XX-XX

- Delete unused dependencies and add `requirements.txt`
- Running the scripts parsing arguments from the terminal is no longer supported
- Move all examples to the `examples` directory
- Add a Latin Hypercube Sampler script (`R` with `lhs` required)
- Remove PTMCMC since the code does not parallelize properly and ptemcee is no longer maintained
- Fix range of pocoMC uniform prior distributions. This does not cause problems with previous results, since the `log_likelihood` evaluates to `-np.inf` for values outside the prior range. Thanks @wenbin1501110084 for pointing this out.
- Implement option to switch off PCA transformation in the scikit GP emulator wrapper
- Implement option for 'Matern' kernel in the scikit GP emulator wrapper ('RBF' is the default)
- Implement option to only use the diagonal of the covariances in all GP emulator wrappers (only when logTrafo is set to True)

[Link to diff from previous version](https://github.com/Hendrik1704/GPBayesTools-HIC/compare/v1.1.0...v1.2.0)

## v1.1.0
Date: 2024-07-24

- pocoMC sampler added
- Optional parameter for error handling of the training points in the GP emulators

[Link to diff from previous version](https://github.com/Hendrik1704/GPBayesTools-HIC/compare/v1.0.0...v1.1.0)

## v1.0.0
Date: 2024-05-13

**[First public version of GPBayesTools-HIC ](https://github.com/Hendrik1704/GPBayesTools-HIC/releases/tag/v1.0.0)**