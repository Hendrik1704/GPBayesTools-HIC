# GPBayesTools-HIC

Gaussian Process Bayesian Toolkit with Monte Carlo Sampler Integration for Heavy Ion Collisions

This toolkit implements a wrapper for Gaussian Process (GP) emulators and Monte Carlo (MC) samplers used in 
high-energy heavy-ion simulations.

The following wrappers for GP emulators are currently included:
- Scikit Learn GP emulator wrapper
- PCGP and PCSK wrapper for the GPs implemented in the [surmise](https://github.com/bandframework/surmise) package of the [BAND](https://bandframework.github.io/) Collaboration

The following wrappers for MC sampling are included:
- MCMC wrapper for the [emcee](https://github.com/topics/emcee) package
- [PTMCMC](https://github.com/willvousden/ptemcee) (Parallel Tempering Markov Chain Monte Carlo) wrapper
    - Not recommend to use this one for larger runs. There are problems with the parallelization.
- [PTLMC](https://github.com/bandframework/surmise) from the surmise package (Parallel Tempering Langevin Monte Carlo)
- [pocoMC](https://github.com/minaskar/pocomc) Preconditioned Monte Carlo method for accelerated Bayesian inference


:exclamation: The jupyter notebooks are just meant as examples for how to use the emulators and samplers and analyze the output.
Paths and data files need the proper input formats.