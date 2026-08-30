import numpy as np
import pytest


@pytest.mark.parametrize("Te", [3000.0, 5000.0, 8783.0, 20000.0])
@pytest.mark.parametrize("n_p", [1e22, 1e24])
def test_frequencies_match_reference(main_ref, transport_model, Te, n_p):
    expected_nu_E, expected_nu_M = main_ref.get_frequencies(n_p, main_ref.sigma_ep, Te)

    actual_nu_M = transport_model.get_momentum_transfer_frequency(Te, n_p)
    actual_nu_E = transport_model.get_energy_transfer_frequency(Te, n_p)

    # nu_M doesn't depend on particle_mass, so it matches to full precision;
    # nu_E does (via m_e/m_p), so it inherits the ~1e-10 rel. divergence between
    # main.py's hardcoded amu literal and GasType's molar_mass/N_A derivation.
    assert actual_nu_M == pytest.approx(expected_nu_M, rel=1e-12)
    assert actual_nu_E == pytest.approx(expected_nu_E, rel=1e-9)


def test_frequencies_match_reference_array(main_ref, transport_model):
    Te = np.linspace(2000.0, 20000.0, 25)
    n_p = np.linspace(1e21, 1e25, 25)

    expected_nu_E, expected_nu_M = main_ref.get_frequencies(n_p, main_ref.sigma_ep, Te)

    np.testing.assert_allclose(transport_model.get_momentum_transfer_frequency(Te, n_p), expected_nu_M, rtol=1e-12)
    np.testing.assert_allclose(transport_model.get_energy_transfer_frequency(Te, n_p), expected_nu_E, rtol=1e-9)


def test_energy_transfer_is_momentum_transfer_scaled_by_mass_ratio(main_ref, transport_model, gas_type):
    """nu_E = 2 (m_e / m_p) nu_M -- Friedberg 3.4."""
    nu_M = transport_model.get_momentum_transfer_frequency(6000.0, 1e23)
    nu_E = transport_model.get_energy_transfer_frequency(6000.0, 1e23)
    expected_ratio = 2.0 * main_ref.m_e / gas_type.particle_mass
    assert nu_E / nu_M == pytest.approx(expected_ratio, rel=1e-12)
