## v1.2.0
Date: 2024-XX-XX

- Delete unused dependencies and add `requirements.txt`
- Running the scripts parsing arguments from the terminal is no longer supported
- Move all examples to the `examples` directory
- Fix range of pocoMC uniform prior distributions. This does not cause problems with previous results, since the `log_likelihood` evaluates to `-np.inf` for values outside the prior range.

## v1.1.0
Date: 2024-07-24

- pocoMC sampler added
- Optional parameter for error handling of the training points in the GP emulators

[Link to diff from previous version](https://github.com/Hendrik1704/GPBayesTools-HIC/compare/v1.0.0...v1.1.0)

## v1.0.0
Date: 2024-05-13

**[First public version of GPBayesTools-HIC ](https://github.com/Hendrik1704/GPBayesTools-HIC/releases/tag/v1.0.0)**