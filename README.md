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
- MCMC wrapper for the [emcee](https://github.com/topics/emcee) package
- [PTLMC](https://github.com/bandframework/surmise) from the surmise package (Parallel Tempering Langevin Monte Carlo)
- [pocoMC](https://github.com/minaskar/pocomc) Preconditioned Monte Carlo method for accelerated Bayesian inference

We recommend to use the `pocoMC` sampler.

## Emulators trained on the log of the observables

All emulators have a `logTrafo` option to train them on the logarithm of the observables.
This requires non-negative observables, and the relative statistical errors of the training data
are used as errors in log space.
What the `predict` function returns depends on the `exp_and_cov_diagonal` option:

- `exp_and_cov_diagonal=False` (default): the mean and the covariance are returned in log space.
  The experimental data used with the emulator in the MCMC (`Chain`) are used as they are given,
  so they have to be log-transformed by the user as well: `log(y)` for the values and the
  relative errors `sigma/y` for the errors.
- `exp_and_cov_diagonal=True`: the predictions are transformed back to the original scale,
  i.e. `exp(mean)` and the covariance `(sigma * exp(mean))^2`, and the experimental data
  are used in the original scale.
  The covariance is diagonal for `Emulator`, `EmulatorBAND` and `EmulatorHETGPy`, while
  `EmulatorSparseGP` keeps the correlations between the observables.

If emulators with different settings are combined in one `Chain`, the experimental data of each
emulator must be given in the scale of its predictions.

## Latin Hypercube Sampling

There is also a script to generate Latin Hypercube Design parameter files.
An example how to use it is given in the `examples` directory in the `generate_LHD_Bayes.py` script.
This requires a file specifying the parameter ranges, see for example `examples/modelDesign_example.txt`.

## Posterior Cluster Sampling

The `generate_posterior_clusters.py` script in the `examples` directory can be used to sample parameter 
clusters from the posterior chain file after a Bayesian inference run and propagate model
uncertainties to the observables.
The final `cluster_centers.txt` file contains the sampled parameter clusters as separate columns.

## Requirements

Check the `requirements.txt` file for the dependencies of this code.

:exclamation: The jupyter notebooks are just meant as examples for how to use the emulators and samplers and analyze the output.
Paths and data files need the proper input formats.