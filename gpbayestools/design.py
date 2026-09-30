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
    designs). scipy (``scipy.stats.qmc.LatinHypercube``) has optimized
    Latin hypercubes, but no maximum projection designs.

    This means that R must be installed with the package of the chosen method
    (run ``install.packages('MaxPro')`` or ``install.packages('lhs')`` in an
    R session).

"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from . import cachedir, parse_model_parameter_file

logger = logging.getLogger(__name__)


def _generate_with_r(method, r_code, npoints, ndim, seed):
    """
    Run `r_code` in R and return the design it writes to stdout as an array.

    The design is cached in cachedir/lhs/<method>/ and loaded from there if
    it exists.
    """
    cachefile = (
        cachedir / "lhs" / method / f"npoints{npoints}_ndim{ndim}_seed{seed}.npy"
    )

    if cachefile.exists():
        logger.debug(f"Loading the design from the cache {cachefile}")
        return np.load(cachefile)

    logger.debug(f"Design not in the cache {cachefile}, generating it with R ...")
    proc = subprocess.run(
        ["R", "--slave"], input=r_code.encode(), capture_output=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"R failed to generate the {method} design (exit code "
            f"{proc.returncode}):\n{proc.stderr.decode().strip()}"
        )
    lhs = np.array(
        [line.split() for line in proc.stdout.decode().splitlines()], dtype=float
    )
    # the MaxPro output contains the run order as the first column
    expected_shape = (npoints, ndim + 1 if method == "maxpro" else ndim)
    if lhs.shape != expected_shape:
        raise RuntimeError(
            f"R returned a {method} design of shape {lhs.shape}, expected "
            f"{expected_shape}"
        )

    cachefile.parent.mkdir(parents=True, exist_ok=True)
    np.save(cachefile, lhs)
    return lhs


def generate_maximin_lhs(npoints, ndim, seed):
    """
    Generate a maximin Latin-hypercube sample (LHS) in [0, 1]^ndim.

    The sample is generated with the R package lhs and cached.

    Parameters
    ----------
    npoints : int
        Number of design points.
    ndim : int
        Number of dimensions (parameters).
    seed : int
        Random seed passed to R's ``set.seed()``.

    Returns
    -------
    ndarray of shape (npoints, ndim)
        The design points in [0, 1]^ndim.
    """
    logger.debug(
        "Generating a maximin LHS: npoints = %d, ndim = %d, seed = %d",
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
    Generate a maximum projection Latin-hypercube sample (LHS) in [0, 1]^ndim.

    The sample is generated with the R package MaxPro and cached.

    Parameters
    ----------
    npoints : int
        Number of design points.
    ndim : int
        Number of dimensions (parameters).
    seed : int
        Random seed passed to R's ``set.seed()``.

    Returns
    -------
    ndarray of shape (npoints, ndim)
        The design points in [0, 1]^ndim.
    """
    logger.debug(
        "Generating a maximum projection LHS: npoints = %d, ndim = %d, seed = %d",
        npoints,
        ndim,
        seed,
    )
    lhs = _generate_with_r(
        "maxpro",
        f"""
        library(MaxPro)
        set.seed({seed})
        write.table(MaxProRunOrder(MaxProLHD({npoints}, {ndim})$Design)$Design,"""
        """ col.names=FALSE, row.names=FALSE)
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

    Creates a design with the given number of points for the parameters in
    the parameter file. The class also implicitly converts to a numpy array.

    Parameters
    ----------
    parameter_file : str or path-like
        Path to the model parameter file.
    npoints : int, default=500
        Number of design points.
    validation : bool, default=False
        If True, the design is a validation design, which only changes the
        output directory of `write_files` (``design_type``). A validation
        design needs a different `seed` than the main design, with the same
        seed both designs are identical.
    seed : int or None, default=None
        Random seed. If None, a seed is generated from the current time. It
        is logged and stored in ``seed`` to be able to reproduce the design.
    method : {"maxpro", "maximin"}, default="maxpro"
        Latin-hypercube design: 'maxpro' (maximum projection design, R
        package MaxPro) or 'maximin' (maximin design, R package lhs).

    Attributes
    ----------
    design_type : str
        'main' or 'validation'.
    pardict : dict
        All parameters and their bounds, as returned by
        ``parse_model_parameter_file``.
    param_min, param_max : ndarray
        Minimum and maximum values of the parameters.
    ndim : int
        Number of parameters (i.e. dimensions).
    points : list of str
        Design point names ``parameter_000``, ``parameter_001``, ... (used as
        file names by `write_files`).
    array : ndarray of shape (npoints, ndim)
        The actual design array.
    seed : int
        The random seed used to generate the design.

    Raises
    ------
    ValueError
        If `method` is unknown, or if `method` is 'maxpro' and there is only
        one parameter (MaxPro requires at least two).
    RuntimeError
        If R fails to generate the design.
    """

    def __init__(
        self, parameter_file, npoints=500, validation=False, seed=None, method="maxpro"
    ):
        if method not in design_generators:
            raise ValueError(
                f"Unknown design method '{method}', "
                f"use one of {list(design_generators)}"
            )
        self.pardict = parse_model_parameter_file(parameter_file)
        self.design_type = "validation" if validation else "main"

        self.ndim = len(self.pardict.keys())
        if method == "maxpro" and self.ndim < 2:
            raise ValueError(
                "MaxPro designs require at least two parameters, use method='maximin'"
            )

        # use padded numbers for design point names
        fmt = "parameter_{:0" + str(len(str(npoints - 1))) + "d}"
        self.points = [fmt.format(i) for i in range(npoints)]

        # set default seeds
        if seed is None:
            # R's set.seed() requires an integer, positive 32-bit seeds are
            # used here
            seed = int(datetime.now().timestamp() * 1000) % (2**31 - 1)
            logger.info(f"No seed given, using the seed {seed}")
        self.seed = seed

        self.param_min = []
        self.param_max = []
        for val in self.pardict.values():
            self.param_min.append(val[1])
            self.param_max.append(val[2])
        self.param_min = np.array(self.param_min)
        self.param_max = np.array(self.param_max)

        # generate the Latin-Hypercube samples
        self.array = self.param_min + (
            self.param_max - self.param_min
        ) * design_generators[method](npoints, self.ndim, seed)
        logger.info(
            f"Generated a {method} design with {npoints} points for {self.ndim} "
            f"parameters (seed {seed})"
        )

    def __array__(self, dtype=None, copy=None):
        """Return the design array (numpy array interface)."""
        if copy:
            return np.array(self.array, dtype=dtype)
        return np.asarray(self.array, dtype=dtype)

    def write_files(self, basedir):
        """
        Write an input file for each design point.

        The files are written to ``basedir / design_type``, one file per
        design point named after the point, with one line ``name value`` per
        parameter.

        Parameters
        ----------
        basedir : str or path-like
            Base directory of the input files. It is created if it does not
            exist.
        """
        outdir = Path(basedir) / self.design_type
        outdir.mkdir(parents=True, exist_ok=True)

        for point, row in zip(self.points, self.array, strict=True):
            filepath = outdir / point
            with filepath.open("w") as f:
                idx = 0
                for ikey in self.pardict.keys():
                    f.write(f"{ikey} {row[idx]}\n")
                    idx += 1
                logger.debug("Wrote %s", filepath)
        logger.info(f"Wrote {len(self.points)} design files to {outdir}")
