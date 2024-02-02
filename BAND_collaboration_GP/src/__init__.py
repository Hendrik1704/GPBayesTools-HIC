""" Project initialization and common objects. """

import logging
import os
from pathlib import Path
import re
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
    f = open(parfile, 'r')
    for line in f:
        par = line.split("#")[0]
        if par != "":
            par = par.split(":")
            key = par[0]
            val = [ival.strip() for ival in par[1].split(",")]
            for i in range(1, 3):
                val[i] = float(val[i])
            pardict.update({key: val})
    return pardict