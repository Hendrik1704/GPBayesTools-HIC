"""
Generates Latin-hypercube parameter designs.

The :class:`Design` class generates a design for the parameters in a
parameter file and writes the input files for the physics model, see
``examples/generate_LHD_Bayes.py``.

.. warning::

    This module uses R to generate the Latin-hypercube samples, either with
    the `MaxPro package <https://cran.r-project.org/package=MaxPro>`_
    (maximum projection designs) or with the
    `lhs package <https://cran.r-project.org/package=lhs>`_ (maximin
    designs).  As far as I know, there are no equivalent libraries for
    Python.

    This means that R must be installed with the package of the chosen method
    (run ``install.packages('MaxPro')`` or ``install.packages('lhs')`` in an
    R session).

"""

import logging
import subprocess
from datetime import datetime

import numpy as np

from . import cachedir, parse_model_parameter_file

logger = logging.getLogger(__name__)


def _generate_with_r(method, r_code, npoints, ndim, seed):
    """
    Run `r_code` in R and return the design it writes to stdout as an array.
    The design is cached in cachedir/lhs/<method>/.

    """
    cachefile = (
        cachedir / "lhs" / method / f"npoints{npoints}_ndim{ndim}_seed{seed}.npy"
    )

    if cachefile.exists():
        logger.debug("loading from cache")
        return np.load(cachefile)

    logger.debug("not found in cache, generating using R")
    proc = subprocess.run(
        ["R", "--slave"], input=r_code.encode(), stdout=subprocess.PIPE, check=True
    )
    lhs = np.array(
        [line.split() for line in proc.stdout.decode().splitlines()], dtype=float
    )

    cachefile.parent.mkdir(parents=True, exist_ok=True)
    np.save(cachefile, lhs)
    return lhs


def generate_maximin_lhs(npoints, ndim, seed):
    """
    Generate a maximin Latin-hypercube sample (LHS) in [0, 1]^ndim with the
    given number of points, dimensions, and random seed, using the R package
    lhs.

    """
    logger.debug(
        "generating maximin LHS: npoints = %d, ndim = %d, seed = %d",
        npoints,
        ndim,
        seed,
    )
    return _generate_with_r(
        "maximin",
        f"""
        library('lhs')
        set.seed({seed})
        write.table(maximinLHS({npoints}, {ndim}), col.names=FALSE, row.names=FALSE)
        """,
        npoints,
        ndim,
        seed,
    )


def generate_maxpro_lhs(npoints, ndim, seed):
    """
    Generate a maximum projection Latin-hypercube sample (LHS) in [0, 1]^ndim
    with the given number of points, dimensions, and random seed, using the R
    package MaxPro.

    """
    logger.debug(
        "generating maximum projection LHS: npoints = %d, ndim = %d, seed = %d",
        npoints,
        ndim,
        seed,
    )
    lhs = _generate_with_r(
        "maxpro",
        f"""
        library(MaxPro)
        set.seed({seed})
        write.table(MaxProRunOrder(MaxProLHD({npoints}, {ndim})$Design)$Design, col.names=FALSE, row.names=FALSE)
        """,
        npoints,
        ndim,
        seed,
    )
    # the first column of MaxProRunOrder is the run order
    return lhs[:, 1:]


# available design methods for the Design class
design_generators = {
    "maxpro": generate_maxpro_lhs,
    "maximin": generate_maximin_lhs,
}


class Design:
    """
    Latin-hypercube model design.

    Creates a design for the given parameter set
    with the given number of points.
    Creates the main (training) design if `validation` is false (default);
    creates the validation design if `validation` is true.
    If `seed` is not given, a random seed is generated from the current time.
    It is printed and stored in ``seed`` to be able to reproduce the design.
    `method` selects the Latin-hypercube design: 'maxpro' (maximum
    projection design, R package MaxPro, default) or 'maximin' (maximin
    design, R package lhs).

    Public attributes:

    - ``type``: 'main' or 'validation'
    - ``pardict``: a dictionary contains all the parameters and their bounds
    - ``min``, ``max``: numpy arrays of parameter min and max
    - ``ndim``: number of parameters (i.e. dimensions)
    - ``points``: list of design point names (formatted numbers)
    - ``array``: the actual design array
    - ``seed``: the random seed used to generate the design

    The class also implicitly converts to a numpy array.

    """

    def __init__(
        self, parfile, npoints=500, validation=False, seed=None, method="maxpro"
    ):
        if method not in design_generators:
            raise ValueError(
                f"Unknown design method '{method}', use one of {list(design_generators)}"
            )
        self.pardict = parse_model_parameter_file(parfile)
        self.type = "validation" if validation else "main"

        self.ndim = len(self.pardict.keys())

        # use padded numbers for design point names
        fmt = "parameter_{:0" + str(len(str(npoints - 1))) + "d}"
        self.points = [fmt.format(i) for i in range(npoints)]

        # set default seeds
        if seed is None:
            # R's set.seed() requires an integer, positive 32-bit seeds are
            # used here
            seed = int(datetime.now().timestamp() * 1000) % (2**31 - 1)
            logger.info(f"seed = {seed}")
        self.seed = seed

        self.min = []
        self.max = []
        for val in self.pardict.values():
            self.min.append(val[1])
            self.max.append(val[2])
        self.min = np.array(self.min)
        self.max = np.array(self.max)

        # generate the Latin-Hypercube samples
        self.array = self.min + (self.max - self.min) * design_generators[method](
            npoints, self.ndim, seed
        )

    def __array__(self):
        return self.array

    def write_files(self, basedir):
        """
        Write input files for each design point as a python dictionary
        to `basedir`.

        """
        outdir = basedir / self.type
        outdir.mkdir(parents=True, exist_ok=True)

        for point, row in zip(self.points, self.array, strict=True):
            filepath = outdir / point
            with filepath.open("w") as f:
                idx = 0
                for ikey in self.pardict.keys():
                    f.write(f"{ikey} {row[idx]}\n")
                    idx += 1
                logger.debug("wrote %s", filepath)
