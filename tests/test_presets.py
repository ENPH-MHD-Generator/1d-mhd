"""
presets.py had no test coverage before this file (verified by exploration) -- these
tests cover only the new additions (the alternate seed-species presets), not the
pre-existing default_gas_type()/default_seed_type(), which is out of scope here.
"""
from magnetohydrodynamics.presets import (
    all_seed_types,
    default_seed_type,
    potassium_seed_type,
    sodium_seed_type,
)


def test_potassium_seed_type_fields():
    seed_type = potassium_seed_type()
    assert seed_type.name == "Potassium"
    assert seed_type.ionization_potential == 4.341
    assert seed_type.degeneracy_ratio == 1.0


def test_sodium_seed_type_fields():
    seed_type = sodium_seed_type()
    assert seed_type.name == "Sodium"
    assert seed_type.ionization_potential == 5.139
    assert seed_type.degeneracy_ratio == 1.0


def test_new_seed_types_reuse_caesiums_cross_section():
    """Deliberate: this model only accounts for electron-primary-gas collisions, so the
    cross-section is physically an electron-argon quantity, not seed-species-specific --
    see potassium_seed_type()'s docstring."""
    caesium = default_seed_type()
    assert potassium_seed_type().electron_neutral_cross_section == caesium.electron_neutral_cross_section
    assert sodium_seed_type().electron_neutral_cross_section == caesium.electron_neutral_cross_section


def test_all_seed_types_includes_every_preset():
    seed_types = all_seed_types()
    names = {seed_type.name for seed_type in seed_types}
    assert names == {"Caesium", "Potassium", "Sodium"}
    assert len(seed_types) == 3


def test_all_seed_types_caesium_entry_matches_default():
    caesium_from_list = next(s for s in all_seed_types() if s.name == "Caesium")
    assert caesium_from_list == default_seed_type()
