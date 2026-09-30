"""Tests of the Latin-hypercube design; R is replaced by a random design."""

import numpy as np
import pytest

import gpbayestools.design as design
from gpbayestools.design import Design


@pytest.fixture(autouse=True)
def fake_r(monkeypatch):
    """Replace the R call by a random design with the same columns."""

    def generate(method, r_code, n_points, n_dim, seed):
        points = np.random.default_rng(seed).uniform(size=(n_points, n_dim))
        if method == "maxpro":
            # MaxProRunOrder adds the run order as the first column
            return np.column_stack([np.arange(1, n_points + 1), points])
        return points

    monkeypatch.setattr(design, "_generate_with_r", generate)


@pytest.fixture
def param_file(tmp_path):
    path = tmp_path / "par.txt"
    path.write_text("a: $a$, 0, 1\nb: $b$, -2, 2\n")
    return path


@pytest.mark.parametrize("method", ["maxpro", "maximin"])
def test_design_in_parameter_ranges(param_file, method):
    d = Design(param_file, n_points=20, seed=1, method=method)
    assert np.asarray(d).shape == (20, 2)
    assert np.all(d.array >= [0, -2]) and np.all(d.array <= [1, 2])
    assert d.seed == 1
    np.testing.assert_array_equal(
        d.array, Design(param_file, n_points=20, seed=1, method=method).array
    )


def test_write_files(param_file, tmp_path):
    d = Design(param_file, n_points=3, seed=1, validation=True)
    # the base directory can be a string
    d.write_files(str(tmp_path / "out"))
    files = sorted((tmp_path / "out" / "validation").iterdir())
    assert [f.name for f in files] == ["parameter_0", "parameter_1", "parameter_2"]
    name, value = files[1].read_text().splitlines()[1].split()
    assert name == "b" and float(value) == d.array[1, 1]


def test_invalid_options(param_file, tmp_path):
    with pytest.raises(ValueError):
        Design(param_file, method="random")
    one_parameter = tmp_path / "one.txt"
    one_parameter.write_text("a: $a$, 0, 1\n")
    with pytest.raises(ValueError):
        Design(one_parameter)
    assert Design(one_parameter, n_points=5, seed=1, method="maximin").n_dim == 1


def test_invalid_design_arguments(param_file):
    # the arguments are inserted into the R code, so only integers are accepted
    for kwargs in (
        {"seed": "1); system('ls')"},
        {"seed": 1.5},
        {"n_points": "10"},
        {"seed": True},
    ):
        with pytest.raises(TypeError):
            Design(param_file, **{"n_points": 5, "seed": 1, **kwargs})
    for kwargs in ({"n_points": 0}, {"seed": 2**31}):
        with pytest.raises(ValueError):
            Design(param_file, **{"n_points": 5, "seed": 1, **kwargs})
    with pytest.raises(TypeError):
        design.generate_maximin_lhs(10, 2, "1")
    # numpy integers are integers
    assert Design(param_file, n_points=np.int64(5), seed=np.int32(3)).seed == 3
