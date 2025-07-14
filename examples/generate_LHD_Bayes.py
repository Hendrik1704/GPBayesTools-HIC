import sys
from os import path
# Add the parent directory to sys.path
sys.path.insert(0, path.abspath('../'))
from src.design import Design
from pathlib import Path

# Create a LHD with 1000 points
design = Design('./modelDesign_example.txt', npoints=100, validation=False, seed=42)
design.write_files(Path('./designs'))