"""
Generate a Latin-hypercube design for the parameters in
``modelDesign_example.txt`` and write the input files to ``./designs``.
"""

import logging
import sys
from os import path

# Add the parent directory to sys.path
sys.path.insert(0, path.abspath("../"))
from pathlib import Path

from gpbayestools.design import Design

# show the messages of gpbayestools, e.g. the seed of the design
logging.basicConfig(level=logging.INFO)

# Create a LHD with 100 points
# method: 'maxpro' (maximum projection design, R package MaxPro) or
#         'maximin' (maximin design, R package lhs)
design = Design(
    "./modelDesign_example.txt", n_points=100, validation=False, seed=42, method="maxpro"
)
design.write_files(Path("./designs"))
