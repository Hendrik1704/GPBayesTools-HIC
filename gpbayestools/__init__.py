"""Project initialization and common objects."""

import copy
import functools
import os
from pathlib import Path

import dill
import numpy as np


workdir = Path(os.getenv("WORKDIR", "."))

# created when it is needed (see design.py)
cachedir = workdir / "cache"


def keep_trained_state(method):
    """
    Decorator for emulator validation methods, which train the emulator on a
    subset of the training data. The attributes of the emulator are restored
    after the call, so that the trained emulator is not changed. Training must
    therefore replace attributes instead of modifying them in place. Random
    number generators are copied, so that their state is restored as well.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        state = {
            key: copy.deepcopy(val) if isinstance(val, np.random.Generator) else val
            for key, val in self.__dict__.items()
        }
        try:
            return method(self, *args, **kwargs)
        finally:
            self.__dict__.clear()
            self.__dict__.update(state)

    return wrapper


# module and class names of versions < 3.0.0
_LEGACY_MODULES = {
    "emulator": "emulator_sklearn",
    "emulator_BAND": "emulator_band",
    "emulator_hetGPy": "emulator_hetgp",
    "emulator_sparseGP": "emulator_sparse_gp",
    "mcmc": "bayesian_analysis",
}
_LEGACY_CLASSES = {
    ("emulator_sklearn", "Emulator"): "EmulatorSklearn",
    ("emulator_hetgp", "EmulatorHETGPy"): "EmulatorHetGP",
    ("bayesian_analysis", "Chain"): "BayesianAnalysis",
}


class _LegacyUnpickler(dill.Unpickler):
    """Unpickler that maps the module and class names of versions < 3.0.0, in
    which the package was called src and the modules and classes had other
    names, to the current names."""

    def find_class(self, module, name):
        if module == "src" or module.startswith("src."):
            module = __name__ + module[len("src") :]
        if module.startswith(__name__ + "."):
            submodule = module[len(__name__) + 1 :]
            submodule = _LEGACY_MODULES.get(submodule, submodule)
            module = __name__ + "." + submodule
            name = _LEGACY_CLASSES.get((submodule, name), name)
        return super().find_class(module, name)


def load_emulator(path):
    """
    Load an emulator saved with dill. Emulators saved with versions < 3.0.0,
    in which the package, the modules and some classes had other names, can
    be loaded as well.
    """
    with open(path, "rb") as f:
        return _LegacyUnpickler(f).load()


def parse_model_parameter_file(parfile):
    pardict = {}
    with open(parfile, "r") as f:
        for line in f:
            par = line.split("#")[0].strip()
            if par == "":
                # skip empty and comment lines
                continue
            key, par = par.split(":", 1)
            val = [ival.strip() for ival in par.split(",")]
            for i in range(1, 3):
                val[i] = float(val[i])
            pardict.update({key.strip(): val})
    return pardict
