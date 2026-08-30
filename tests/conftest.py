"""
Shared fixtures for the test suite.

`main_ref` loads tests/reference_main.py -- a frozen copy of the original,
hand-validated single-file implementation -- as a plain module, so its
functions can be used as the "ground truth" oracle that the
magnetohydrodynamics package (and the migrated main.py) are checked against.
See tests/reference_main.py's docstring.
"""
import importlib.util
from pathlib import Path

import pytest

import magnetohydrodynamics as mhd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def main_ref():
    return _load_module("main_ref", REPO_ROOT / "tests" / "reference_main.py")


@pytest.fixture(scope="session")
def main_module():
    """The real, migrated main.py -- uses the magnetohydrodynamics package."""
    return _load_module("main_module", REPO_ROOT / "main.py")


@pytest.fixture
def gas_type(main_ref) -> mhd.GasType:
    """Argon, matching main.py's module-level m_p/gamma constants."""
    return mhd.GasType(name="Argon", molar_mass=39.948e-3, heat_capacity_ratio=main_ref.gamma)


@pytest.fixture
def seed_type(main_ref) -> mhd.SeedType:
    """Caesium seed, matching main.py's module-level eps_eV/sigma_ep constants."""
    return mhd.SeedType(
        name="Caesium",
        ionization_potential=main_ref.eps_eV,
        electron_neutral_cross_section=main_ref.sigma_ep,
        degeneracy_ratio=1.0,
    )


@pytest.fixture
def transport_model(seed_type, gas_type) -> mhd.MHDTransportModel:
    return mhd.MHDTransportModel(seed_type=seed_type, gas_type=gas_type)


@pytest.fixture
def ionization_model(seed_type) -> mhd.LocalThermodynamicEquilibrium:
    return mhd.LocalThermodynamicEquilibrium(seed_type=seed_type)


@pytest.fixture
def hall_solver(gas_type, seed_type, transport_model, ionization_model) -> mhd.HallSolver:
    return mhd.HallSolver(
        gas_type=gas_type,
        seed_type=seed_type,
        transport_model=transport_model,
        ionization_model=ionization_model,
    )


@pytest.fixture
def channel_params() -> dict:
    """The exact operating point used in main.py's main()."""
    load_resistivity = 1.5 / 2 * 50 / 2 / 2 / 2 / 2 / 20  # Ohm * m
    channel_area = 48e-3 * 48e-3  # m^2
    channel_length = 0.2  # m
    return dict(
        num_slices=200,
        L=channel_length,
        A=channel_area,
        u0=150.115,
        p0=101.01e3,
        Tp0=2000.0,
        B0=0.5,
        seed_frac0=6.18e-3,
        R_L=load_resistivity * channel_length / channel_area,
    )
