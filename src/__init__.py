""" Project initialization and common objects. """

import copy
import functools
import logging
import os
from pathlib import Path
import sys

import numpy as np


logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

workdir = Path(os.getenv('WORKDIR', '.'))

cachedir = workdir / 'cache'
cachedir.mkdir(parents=True, exist_ok=True)


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
        state = {key: copy.deepcopy(val)
                 if isinstance(val, np.random.Generator) else val
                 for key, val in self.__dict__.items()}
        try:
            return method(self, *args, **kwargs)
        finally:
            self.__dict__.clear()
            self.__dict__.update(state)
    return wrapper


def parse_model_parameter_file(parfile):
    pardict = {}
    with open(parfile, 'r') as f:
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