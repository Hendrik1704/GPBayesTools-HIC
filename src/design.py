"""
Generates Latin-hypercube parameter designs.

When run as a script, writes input files for use with the physics model
Run ``python -m src.design --help`` for usage information.

.. warning::

    This module uses the R `lhs package
    <https://cran.r-project.org/package=lhs>`_ to generate maximin
    Latin-hypercube samples.  As far as I know, there is no equivalent library
    for Python (I am aware of `pyDOE <https://pythonhosted.org/pyDOE>`_, but
    that uses a much more rudimentary algorithm for maximin sampling).

    This means that R must be installed with the lhs package (run
    ``install.packages('lhs')`` in an R session).

"""

import logging
from pathlib import Path
import subprocess
from datetime import datetime

import numpy as np

from . import cachedir, parse_model_parameter_file


def generate_lhs(npoints, ndim, seed):
    """
    Generate a maximin Latin-hypercube sample (LHS) with the given number of
    points, dimensions, and random seed.

    """
    logging.debug(
        'generating maximin LHS: '
        'npoints = %d, ndim = %d, seed = %d',
        npoints, ndim, seed
    )

    cachefile = (
        cachedir / 'lhs' /
        'npoints{}_ndim{}_seed{}.npy'.format(npoints, ndim, seed)
    )

    if cachefile.exists():
        logging.debug('loading from cache')
        lhs = np.load(cachefile)
    else:
        logging.debug('not found in cache, generating using R')
        '''
        proc = subprocess.run(
            ['R', '--slave'],
            input="""
            library('lhs')
            set.seed({})
            write.table(maximinLHS({}, {}), col.names=FALSE, row.names=FALSE)
            """.format(seed, npoints, ndim).encode(),
            stdout=subprocess.PIPE,
            check=True
        )
        '''
        # Maximum Projection Latin Hypercube sampling
        proc = subprocess.run(
            ['R', '--slave'],
            input="""
            library(MaxPro)
            set.seed({})
            write.table(MaxProRunOrder(MaxProLHD({}, {})$Design)$Design, col.names=FALSE, row.names=FALSE)
            """.format(seed, npoints, ndim).encode(),
            stdout=subprocess.PIPE,
            check=True
        )
        lhs = np.array(
            [l.split() for l in proc.stdout.decode().splitlines()],
            dtype=float
        )

        cachefile.parent.mkdir(exist_ok=True)
        np.save(cachefile, lhs)

    return lhs


class Design:
    """
    Latin-hypercube model design.

    Creates a design for the given parameter set
    with the given number of points.
    Creates the main (training) design if `validation` is false (default);
    creates the validation design if `validation` is true.
    If `seed` is not given, a default random seed is used
    (different defaults for the main and validation designs).

    Public attributes:

    - ``type``: 'main' or 'validation'
    - ``pardict``: a dictionary contains all the parameters and their bounds
    - ``min``, ``max``: numpy arrays of parameter min and max
    - ``ndim``: number of parameters (i.e. dimensions)
    - ``points``: list of design point names (formatted numbers)
    - ``array``: the actual design array

    The class also implicitly converts to a numpy array.

    """
    def __init__(self, parfile, npoints=500, validation=False, seed=None):
        self.pardict = parse_model_parameter_file(parfile)
        self.type = 'validation' if validation else 'main'

        self.ndim = len(self.pardict.keys())

        # use padded numbers for design point names
        fmt = 'parameter_{:0' + str(len(str(npoints - 1))) + 'd}'
        self.points = [fmt.format(i) for i in range(npoints)]

        # set default seeds
        if seed is None:
            #seed = 751783496 if validation else 450829120
            seed = datetime.now().timestamp()
            print("seed = {}".format(seed))

        self.min = []
        self.max = []
        for par, val in self.pardict.items():
            self.min.append(val[1])
            self.max.append(val[2])
        self.min = np.array(self.min)
        self.max = np.array(self.max)

        # generate the Latin-Hypercube samples
        #self.array = self.min + (self.max - self.min)*generate_lhs(
        #    npoints=npoints, ndim=self.ndim, seed=seed
        #)
        array_tmp = generate_lhs(npoints, self.ndim, seed)
        self.array = self.min + (self.max - self.min)*array_tmp[:,1:]

    def __array__(self):
        return self.array

    def write_files(self, basedir):
        """
        Write input files for each design point as a python dictionary
        to `basedir`.

        """
        outdir = basedir / self.type
        outdir.mkdir(parents=True, exist_ok=True)

        for point, row in zip(self.points, self.array):
            filepath = outdir / point
            with filepath.open('w') as f:
                idx = 0
                for ikey in self.pardict.keys():
                    f.write("{} {}\n".format(ikey, row[idx]))
                    idx += 1
                logging.debug('wrote %s', filepath)

