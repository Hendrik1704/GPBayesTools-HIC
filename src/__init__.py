""" Project initialization and common objects. """

import logging
import os
from pathlib import Path
import sys


logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

workdir = Path(os.getenv('WORKDIR', '.'))

cachedir = workdir / 'cache'
cachedir.mkdir(parents=True, exist_ok=True)


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