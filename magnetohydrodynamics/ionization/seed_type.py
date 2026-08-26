from dataclasses import dataclass


@dataclass(frozen=True)
class SeedType:
    """Seed species data used in an ionization model."""
    name: str
    ionization_potential: float             # [eps] = [eV]
    electron_neutral_cross_section: float   # [sigma_eps] = [m^2]
    degeneracy_ratio: float = 1.0           # [g] = [ ]
