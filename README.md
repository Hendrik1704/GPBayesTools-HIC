# GPBayesTools-HIC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11186661.svg)](https://doi.org/10.5281/zenodo.11186661)

Gaussian Process Bayesian Toolkit with Monte Carlo Sampler Integration for Heavy Ion Collisions

This toolkit provides wrappers for Gaussian process (GP) emulators and Monte Carlo (MC) samplers
for the Bayesian calibration of models of high-energy heavy-ion collisions and related physics.
It covers the whole workflow: the parameter design, the training and validation of emulators of
the model output, and the sampling of the posterior distribution of the model parameters.

The following GP emulators are included:
- Scikit Learn GP emulator wrapper
- PCGP and PCSK wrapper for the GPs implemented in the [surmise](https://github.com/bandframework/surmise) package of the [BAND](https://bandframework.github.io/) Collaboration
- Heteroskedastic GP wrapper for the [hetgpy](https://hetgpy.readthedocs.io) package
- Sparse variational GP emulator (single emulator or ensemble) implemented with [JAX](https://github.com/jax-ml/jax)

The following MC samplers are included:
- MCMC wrapper for the [emcee](https://github.com/dfm/emcee) package
- [PTLMC](https://github.com/bandframework/surmise) from the surmise package (Parallel Tempering Langevin Monte Carlo)
- [pocoMC](https://github.com/minaskar/pocomc) Preconditioned Monte Carlo method for accelerated Bayesian inference

We recommend to use the `pocoMC` sampler.

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
and the code formatting, and `pip install ".[examples]"` for the notebooks in `examples/`. The
modules are then imported from `gpbayestools`, e.g. `from gpbayestools.emulator_band import EmulatorBAND`.
The dependencies are listed in `pyproject.toml` and, for use without installation, in
`requirements.txt`. The generation of parameter designs additionally requires R with the MaxPro
or lhs package.

Version 3.0.0 is not backward compatible: emulators and chains saved with older versions cannot
be loaded, see the [CHANGELOG](CHANGELOG.md).

## Documentation

The [user guide](gpbayestools/README.md) describes the modules, the formats of the input files,
the emulators and their options, and the Bayesian analysis. The complete list of arguments of each
class and function is in its docstring.

## Example: full Bayesian workflow

[`examples/full_workflow`](examples/full_workflow/README.md) goes through a complete Bayesian study
with the fit of HERA deep-inelastic scattering data with two parameters from
[JHEP 04 (2026) 185](https://doi.org/10.1007/JHEP04(2026)185): parameter design, training data,
training and validation of all emulators, and closure tests with pocoMC. It compares the
accuracy and the calibration of the emulators and the posteriors they give. Further example
notebooks and scripts are in [`examples/`](examples); their paths and data files have to be
adapted to your input files.

## Tests

The tests in the `tests` directory can be run with

```
python -m pytest tests
```

## Citation

If you use this package, please cite the version you used via its Zenodo entry,
[10.5281/zenodo.11186661](https://doi.org/10.5281/zenodo.11186661), which links to all versions.
The validation metrics of the emulators are defined in
[H. Roch, S. A. Jahan and C. Shen, arXiv:2405.12019](https://arxiv.org/abs/2405.12019).
