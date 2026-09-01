"""
Symbolic verification of Derivation.md's "Variable-Area (Tapered) Channel" section --
the dT_p/dx, dv_p/dx elimination for a channel whose cross-sectional area A(x) varies,
generalizing the existing zero-divergence closed forms.

This exists because the analogous elimination for the *zero-divergence* case already
bit this project once (see Derivation.md's "Correction" callout box): an earlier
hand-derivation of that simpler case had a sign error in exactly this kind of algebraic
collection step, caught only by a numerical energy-conservation test after the fact
(tests/test_energy_accounting.py). The tapered case adds a genuinely new term (the A'/A
correction) to collect correctly, so this time the elimination itself is checked
symbolically -- solving the two linear equations directly (sympy.solve), not by
differentiating an assumed closed form -- which is closer to where the historical bug
actually was (an algebraic elimination error, not a differentiation one).

No code under magnetohydrodynamics/ is imported here -- this is a pure check that the
closed-form expressions written into Derivation.md (and implemented in
magnetohydrodynamics/solver/tapered_hall_solver.py) are the actual solution of the two
governing equations, independent of any Python implementation of them.
"""
import sympy as sp


def _closed_form_solution():
    """Solve Eq. A (momentum, after the ideal-gas-law substitution) and Eq. B
    (stagnation energy) for (dv_p/dx, dT_p/dx) directly, as two linear equations in
    those two unknowns -- mirrors exactly how Derivation.md presents the elimination.
    Treats J_y*B_z, S_Omega, Phi(x), and A'(x)/A(x) as independent local quantities
    (not functions of dv/dT), matching how the zero-divergence version of this same
    elimination treats them."""
    dv, dT = sp.symbols("dv dT")
    Phi, m_p, v, T, k, gamma, JyBz, S_Omega, A_over_A = sp.symbols(
        "Phi m_p v T k gamma JyBz S_Omega A_over_A"
    )
    c_p = gamma * k / ((gamma - 1) * m_p)

    eq_a = sp.Eq(
        Phi * (m_p * v**2 - k * T) * dv + k * Phi * v * dT,
        JyBz * v**2 + k * Phi * v * T * A_over_A,
    )
    eq_b = sp.Eq(m_p * Phi * v * dv + m_p * Phi * c_p * dT, JyBz * v + S_Omega)

    solution = sp.solve([eq_a, eq_b], [dv, dT])
    symbols = dict(Phi=Phi, m_p=m_p, v=v, T=T, k=k, gamma=gamma, JyBz=JyBz, S_Omega=S_Omega, A_over_A=A_over_A)
    return solution[dv], solution[dT], symbols


def test_solving_eq_a_and_eq_b_matches_derivation_mds_closed_form():
    """Confirms Derivation.md's published dv_p/dx, dT_p/dx (the tapered-channel
    section) are really what you get from solving Eq. A/Eq. B, not just a
    plausible-looking guess."""
    dv_solution, dT_solution, s = _closed_form_solution()
    denom = s["m_p"] * s["v"] ** 2 - s["gamma"] * s["k"] * s["T"]

    dv_candidate = (
        s["v"] * (s["JyBz"] * s["v"] - s["S_Omega"] * (s["gamma"] - 1)) / (s["Phi"] * denom)
        + s["gamma"] * s["k"] * s["T"] * s["v"] / denom * s["A_over_A"]
    )
    dT_candidate = (s["gamma"] - 1) / s["k"] * (
        s["S_Omega"] * s["m_p"] * s["v"] ** 2 - s["k"] * s["T"] * (s["JyBz"] * s["v"] + s["S_Omega"])
    ) / (s["Phi"] * denom) - (s["gamma"] - 1) * s["m_p"] * s["v"] ** 2 * s["T"] / denom * s["A_over_A"]

    assert sp.simplify(dv_solution - dv_candidate) == 0
    assert sp.simplify(dT_solution - dT_candidate) == 0


def test_zero_divergence_limit_recovers_the_existing_constant_area_formulas():
    """Setting A'(x)/A(x) = 0 (constant area) must reduce Derivation.md's tapered
    dv_p/dx, dT_p/dx *exactly* to the pre-existing zero-divergence formulas -- the
    required special case, and the cheapest possible regression guard against the new
    term being wrong in a way that only shows up away from the limit."""
    dv_solution, dT_solution, s = _closed_form_solution()
    denom = s["m_p"] * s["v"] ** 2 - s["gamma"] * s["k"] * s["T"]

    dv_zero_divergence = s["v"] * (s["JyBz"] * s["v"] - s["S_Omega"] * (s["gamma"] - 1)) / (s["Phi"] * denom)
    dT_zero_divergence = (s["gamma"] - 1) / s["k"] * (
        s["S_Omega"] * s["m_p"] * s["v"] ** 2 - s["k"] * s["T"] * (s["JyBz"] * s["v"] + s["S_Omega"])
    ) / (s["Phi"] * denom)

    assert sp.simplify(dv_solution.subs(s["A_over_A"], 0) - dv_zero_divergence) == 0
    assert sp.simplify(dT_solution.subs(s["A_over_A"], 0) - dT_zero_divergence) == 0
