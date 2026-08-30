import numpy as np
import pytest


@pytest.mark.parametrize("Te", [3000.0, 5000.0, 8000.0, 12000.0])
@pytest.mark.parametrize("ns", [1e18, 1e20, 1e22])
def test_electron_density_matches_reference_scalar(main_ref, ionization_model, Te, ns):
    expected = main_ref.get_electron_density(Te, ns)
    actual = ionization_model.get_electron_density(Te, ns)
    assert actual == pytest.approx(float(expected), rel=1e-12)


def test_electron_density_matches_reference_array(main_ref, ionization_model):
    Te = np.linspace(2000.0, 15000.0, 50)
    ns = np.linspace(1e17, 1e23, 50)

    expected = main_ref.get_electron_density(Te, ns)
    actual = ionization_model.get_electron_density(Te, ns)

    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_electron_density_never_exceeds_seed_density(ionization_model):
    ns = 1e20
    ne = ionization_model.get_electron_density(50000.0, ns)
    assert 0.0 <= ne <= ns
