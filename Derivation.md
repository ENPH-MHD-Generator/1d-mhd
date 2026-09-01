# Temperature Estimation

The goal is to estimate the axial electron temperature as a function of the controllable variables. They are,
1. Flow velocity, $u$,
2. Inlet pressure, $p_0$,
3. Inlet temperature of the (primary), $T_{p0}$,
4. Primary gas to seed ratio, $\frac{n_{s0}}{n_{p0}}$,
5. Seed ionization fraction, $\frac{n_{e0}}{n_{s0}}$,
6. External (vacuum) magnetic field, $B_0$,
7. Gas type,
8. Channel geometry (inlet and outlet size, length),
9. Load resistance, 

where subscripts with a $0$ indicate initial values at the inlet (except for $B_0$ which is everywhere).


We'll assume a few things,
1. Initially, we'll solve using constant velocity equations (namely the pressure ratio)
2. We are ignoring wall effects (so this analysis is for the core plasma)
3. This is a one-dimensional simulation in the axial (direction of plasma flow) where we will be slicing axially and assuming that each slice is in a local equilibrium
4. This is a steady-state simulation (in that it is independent of time, not of a steady-state system).
5. The collisionality is well-described by only primary gas and electron collisions (especially valid for argon).

In terms of high-level decisions, we'll be using an argon primary gas seeded with caesium and a linear Hall generator with a square cross-section.
The two references used are,
1. _Magnetohydrodynamic electrical power generation_, Messerle 1995
2. _The Velikhov-ionisation instability revisited_, Freidberg 2025


## Definitions

We'll move onto defining many relations and variables,

$$ \frac{n_s}{n_{s0}} = \frac{n_p}{n_{p0}} \tag{Friedberg 3.6} $$

$$ \nu_M = n_p Q \bigg(\frac{2kT_e}{m_e}\bigg)^{\frac{1}{2}} \tag{Friedberg 3.8} $$

$$ \eta = \frac{m_e \nu_M}{e^2 n_e} $$

$$ M = \bigg( \frac{3}{5}\frac{m_p v_p^2}{kT_p} \bigg)^{\frac{1}{2}} $$

$$ \nu_E = 2 \frac{m_e}{m_p} \nu_M \tag{Friedberg 3.4} $$

where $\nu_M$ is the electron momentum exchange frequency, $Q$ is the primary gas-electron collision cross-section, $k$ 
is the Boltzmann constant, $T_e$ is the electron temperature, $m_e$ is the electron mass, $\eta$ is the resistivity, 
$e$ is the electron charge, $M$ is the Mach number,$m_p$ is the primary gas mass, $v_p$ is the primary gas speed, 
$T_p$ is the primary gas temperature, and $\nu_E$ is the temperature equilibration frequency.

We also define the all-important Hall parameter,

$$ \beta = \frac{\omega}{\nu_M} = \frac{B_0}{en_e \eta}. $$

where $\omega$ is the cyclotron frequency of the electrons.

## Electromagnetic Equations

Beginning with Ohm's law,

$$ \textbf{E} + \textbf{v}_p \times \textbf{B}_0 = \eta \textbf{J} + \frac{\textbf{J} \times \textbf{B}_0}{e n_e} $$

which in component form is,

$$ E_x + v_{py} B_{0z} - v_{pz} B_{0y} = \eta J_x + \frac{ J_y B_z - J_z B_y }{e n_e} $$
$$ E_y + v_{pz} B_{0x} - v_{px} B_{0z} = \eta J_y + \frac{ J_z B_x - J_x B_z }{e n_e} $$
$$ E_z + v_{px} B_{0y} - v_{py} B_{0x} = \eta J_z + \frac{ J_x B_y - J_y B_x }{e n_e} $$

where we can assume $\textbf{B}$ is uniform and only in the $z$-direction, the flow is along the $x$-direction so $v_{py}=v_{pz}=0$, the electrodes are shorted in the $y$-direction in this setup so $E_y=0$, and no current or voltage flows in the $z$-direction so $E_z=J_z=0$. Ohm's law becoems (Friedberg 3.7),

**NOTE: v_py and v_pz are not zero in a divergent channel!**

$$ E_x  = \eta J_x + \frac{ J_y B_0 }{e n_e}, $$
$$ - v_{p} B_{0} = \eta J_y - \frac{J_x B_0 }{e n_e}, $$
$$ E_z = 0. $$

We can then also define the load resistivity $\eta_L$,

$$ \eta_L = -\frac{E_x}{J_x}, \tag{Friedberg 3.9} $$

and it can be made dimensionless,

$$ Z = \frac{\eta_L}{\eta} = -\frac{E_x}{\eta J_x} \iff E_x = -\eta Z J_x. \tag{Friedberg 3.10} $$

We can then write (**PROVE THIS!**),

$$ J_x = \bigg[  \frac{\beta^2}{\beta^2 + 1 + Z} \bigg] e n_e v_p, $$
$$ J_y =- \bigg[  \frac{\beta (1+Z)}{\beta^2 + 1 + Z} \bigg] e n_e v_p, $$
$$ \frac{E_x}{\eta} =- \bigg[  \frac{\beta^2 Z}{\beta^2 + 1 + Z} \bigg] e n_e v_p, \tag{Friedberg 3.11} $$

and

$$ S_\Omega = \eta J^2 = \eta (J_x^2 + J_y^2) = m_e n_e \nu_M v_p^2 \frac{\beta^2 [ \beta^2 + (1 + Z)^2]}{(\beta^2 + 1 + Z)^2} $$

where the $\eta J^2$ term should be recognized as Ohmic heating. 
We can also write the load power, 

$$ S_L = -\textbf{E} \cdot \textbf{J} = m_e n_e \nu_M \frac{\beta^4 Z}{(\beta^2 + 1 + Z)^2}$$


## Energy Balance

Beginning with the electron energy balance,

$$ \frac{3}{2} \nu_E n_e (kT_e - kT_p) = \eta J^2 \tag{Friedberg 3.2}, $$
$$ T_e - T_p = \frac{2}{3} \frac{\eta J^2}{k\nu_E n_e}, $$
$$ \frac{T_e}{T_p} - 1 = \frac{2}{3} \frac{\eta J^2}{k\nu_E n_e T_p}, $$
$$ \frac{T_e}{T_p} - 1 = \frac{2}{3} \frac{m_e n_e \nu_M v_p^2 }{k\nu_E n_e T_p}\frac{\beta^2 [ \beta^2 + (1 + Z)^2]}{(\beta^2 + 1 + Z)^2}, $$
$$ \frac{T_e}{T_p} - 1 = \frac{1}{3} \frac{ v_p^2 m_p }{k T_p}\frac{\beta^2 [ \beta^2 + (1 + Z)^2]}{(\beta^2 + 1 + Z)^2}, $$
$$ \frac{T_e}{T_p} - 1 = \frac{5 M^2}{9} \frac{\beta^2 [ \beta^2 + (1 + Z)^2]}{(\beta^2 + 1 + Z)^2}. \tag{Friedberg 3.14} $$

We can also define,

$$ \Delta T = \frac{T_e}{T_p} - 1. $$

We also require the Saha equation,

$$ \frac{n_e^2}{n_s - n_e} = \bigg ( \frac{2 \pi m_e k T_e}{h^2} \bigg)^{\frac{3}{2}} \exp{ \frac{\varepsilon}{k T_e} }$$

where $\varepsilon$ is the ionization potential of the seed gas, and $h$ is the Planck constant.

## Momentum

Conservation of mass gives us,

$$ \nabla \cdot (n_p \textbf{v}_p) = 0, $$

and conservation of momentum,

$$ m_p n_p \textbf{v}_p \cdot \nabla \textbf{v}_p = \textbf{J} \times \textbf{B}_0 - \nabla p_p, $$

If we have a channel with zero divergence (same outlet and inlet size), then flow velocity must only vary in the axial ($x$) direction
so that  $\nabla \textbf{v}_p = \frac{\partial v_p}{\partial x} \hat{x}$.
Writing conservation of momentum in component form under those constraints (and the previous constants on the electromagnetic variables),

$$ m_p n_p v_p \frac{\partial v_p}{\partial x} = J_y B_z - \frac{\partial p_p}{\partial x}, $$
$$ 0 = - J_x B_z - \frac{\partial p_p}{\partial y}, $$
$$ 0 = 0, $$

Also conservation of mass,

$$ (\frac{\partial}{\partial x}\hat{x} + \frac{\partial}{\partial x}\hat{y} + \frac{\partial}{\partial x}\hat{z}) \cdot (n_p v_p \hat{x}) = 0, $$
$$ \frac{\partial}{\partial x} (n_p v_p) = 0, $$
$$ v_p \frac{\partial n_p}{\partial x} + n_p \frac{\partial v_p}{\partial x} = 0. $$

but a cleaner solution is to recognize that conservation of mass in one dimension implies the continuity equation,

$$ n_p(x) v_p(x) = \Phi. $$ 

where $\Phi$ is the number flux, which is constant in $x$.

We'll also need to consider the primary gas temperature and speed down the channel.

> **Correction:** an earlier version of this section prescribed $\partial T_p/\partial x$ and $\partial v_p/\partial x$
> from two *independently*-motivated equations -- $m_p n_p v_p c_p \partial T_p/\partial x = S_\Omega - S_L$, and the
> momentum equation below with $\partial p_p/\partial x$ eliminated (which also had a sign error in that elimination).
> Neither mistake is part of the literature-derived electromagnetic/Hall physics above; both were introduced in this
> project's own axial-march derivation. Together they broke energy conservation: numerically, the stagnation
> enthalpy of the primary gas did not decrease by the load power extracted, and could even *increase* along the
> channel. The derivation below fixes both, solving for $\partial T_p/\partial x$ and $\partial v_p/\partial x$
> simultaneously from one consistent energy statement. See `tests/test_energy_accounting.py`.

Of the power drawn from the plasma, only the **load power** $S_L$ (dissipated in the external circuit) actually
leaves the primary gas stream. The Ohmic dissipation $S_\Omega$ (dissipated resistively *within* the plasma) reappears
as heat in that same gas. Both are drawn from the flow's kinetic energy by the Lorentz retarding force $J_y B_z$:
multiplying the momentum equation by $v_p$ and using Ohm's law (Friedberg 3.11) gives the identity
$-J_y B_z v_p = S_\Omega + S_L$ (the mechanical power the field removes from the flow). So the correct energy
equation for the primary gas, written in terms of the stagnation enthalpy $h_0 = c_p T_p + \frac12 v_p^2$, is

$$ \rho v_p \frac{\partial h_0}{\partial x} = \underbrace{J_y B_z v_p}_{\text{Lorentz work}} + \underbrace{S_\Omega}_{\text{Joule heat redeposited}} = -S_L, $$

i.e. stagnation enthalpy decreases by exactly the power delivered to the load -- nothing more, nothing less.

If we then assume that the primary gas acts as an ideal gas,

$$ p_p = n_p k T_p, $$
$$ p_p = \frac{\Phi}{v_p} k T_p, $$
$$ \frac{\partial p_p}{\partial x} = \frac{\partial}{\partial x}\bigg( \frac{\Phi}{v_p} k T_p \bigg), $$
$$ \frac{\partial p_p}{\partial x} =  \Phi k \bigg( \frac{1}{v_p} \frac{\partial T_p}{\partial x} - \frac{T_p}{v_p^2}\frac{\partial v_p}{\partial x} \bigg), $$

substituting into momentum conservation (and correctly collecting the $\partial v_p/\partial x$ terms this time),

$$ \Phi m_p \frac{\partial v_p}{\partial x} = J_y B_z - \Phi k \bigg( \frac{1}{v_p} \frac{\partial T_p}{\partial x} - \frac{T_p}{v_p^2}\frac{\partial v_p}{\partial x} \bigg), $$
$$ \frac{\partial v_p}{\partial x} \bigg( \Phi m_p - \Phi k\frac{T_p}{v_p^2}\bigg) = J_y B_z - \Phi k \frac{1}{v_p} \frac{\partial T_p}{\partial x}, $$

and solving this *together with* the stagnation-energy equation above gives

$$ \frac{\partial T_p}{\partial x} = \frac{\gamma-1}{k}\cdot\frac{S_\Omega m_p v_p^2 - k T_p (J_yB_zv_p + S_\Omega)}{\Phi(m_pv_p^2 - \gamma k T_p)}, $$

$$ \frac{\partial v_p}{\partial x} = \frac{v_p\big(J_yB_zv_p - S_\Omega(\gamma-1)\big)}{\Phi(m_pv_p^2 - \gamma k T_p)}. $$

The shared denominator $m_p v_p^2 - \gamma k T_p = m_p v_p^2(1 - 1/M^2)$ (with $M$ the ordinary, $\gamma$-based Mach
number) is the classic Rayleigh-flow choking singularity at $M=1$: heat addition to subsonic flow drives it toward
sonic, and the closed-form march above isn't valid past that point. `HallSolver.march` stops early (and marks
`Channel.choked = True`) if the flow reaches $M=1$ from either side (a supersonic inlet decelerates *down* to $M=1$
instead, via the same singularity), rather than stepping through it.

**On verification:** the electromagnetic/Ohm's-law closure above (Friedberg 3.2-3.14: $Z$, $\beta$, $\eta$, $\nu_M$,
$Q$, $\eta_L$) is symbolically cross-checked with `sympy` in `recreating_freidberg.ipynb`. The $\partial T_p/\partial
x$, $\partial v_p/\partial x$ elimination above is *not* part of that notebook -- it's checked only numerically, in
`tests/test_energy_accounting.py`, by confirming the stagnation enthalpy actually drops by exactly the load power
extracted (this is what caught the sign error described in the Correction box above). The tapered-channel version of
this same elimination below *is* additionally checked symbolically, in `tests/test_tapered_derivation_sympy.py` --
see that section for why this particular derivation step warranted it.


## Variable-Area (Tapered) Channel

Everything above assumes "a channel with zero divergence (same outlet and inlet size)" (the Momentum section's own
words). This section generalizes to a channel whose cross-sectional area $A(x)$ varies with $x$ -- the standard fix,
in the real MHD-generator literature this whole derivation is built from (Sutton & Sherman; Rosa), for exactly the
problem this repo's own stability/performance exploration ran into: a supersonic inlet's heavy electron overheating
(driving it toward near-total seed ionization, and so toward *stability*) also drives the flow to the $M=1$ choking
singularity within a fraction of a channel length, in a constant-area duct. Diverging the channel buys back distance
from that singularity, the same way a supersonic wind-tunnel diffuser does.

This is implemented as a **separate solver**, `magnetohydrodynamics.solver.tapered_hall_solver.TaperedHallSolver`
-- the constant-area `HallSolver`/`Channel` above are untouched, still validated against the frozen
`tests/reference_main.py` oracle exactly as before. `TaperedHallSolver` reuses `HallSolver`'s local closure (Ohm's
law + Saha + electron energy balance, all of Friedberg 3.2-3.14 above, entirely unaffected by area since it's given
local density/speed/field values regardless of how they arose) by composition, and only replaces the axial-march
mass/momentum/energy ODEs below.

**Two genuinely different things are happening here, worth keeping separate:**

1. Mass, momentum, and stagnation energy with $A(x)$ (below) are the *rigorous* quasi-1D generalization of the
   equations above -- not a new approximation on top of the existing one. The same "properties uniform across each
   cross-section" assumption is being made either way; allowing that cross-section's area to vary axially doesn't
   weaken it further.
2. The electromagnetic closure above assumed $v_{py} = v_{pz} = 0$ ("the flow is along the $x$-direction") -- flagged
   even in the zero-divergence section as **not exactly true in a divergent channel**: following a diverging wall
   means the flow picks up a small $v_{py} \approx v_{px}\tan\theta$ ($\theta$ the local half-angle). This *is* a new
   approximation, kept here for tractability (re-deriving Ohm's law with nonzero $v_{py}/v_{pz}$ is future work, not
   attempted in this pass) -- valid for "gentle" divergence, the same regime real nozzle/diffuser design already
   restricts itself to (rule of thumb: half-angle below roughly 15°, not independently sourced from either reference
   here). `TaperedHallSolver` enforces this with a runtime check against a configurable `max_half_angle_deg`
   (default 15°), raising `DivergenceAngleWarning` (or a hard error in `strict` mode) rather than silently trusting
   it.

### Taper geometry

The first (and, for now, only) supported profile is a **linear taper**: one wall pair of the square channel held
fixed, the other diverging at a constant half-angle $\theta$, so the area itself is *exactly* linear in $x$ (not a
self-similar square taper, where both wall pairs diverge together and area would be quadratic in $x$):

$$ A(x) = A_0 + \frac{dA}{dx}\,x, \qquad \frac{dA}{dx} = 2\sqrt{A_0}\,\tan\theta. $$

The channel is therefore only literally square at the inlet ($x=0$); it becomes a rectangular duct downstream. This
keeps $dA/dx$ a single precomputed constant (not itself a function of $x$), which is what actually makes "linear
taper" correct as a description and keeps the ODEs below no more complex than they need to be for a first version.

### Mass, momentum, and stagnation energy with $A(x)$

Let $\Psi = n_p(x)\,v_p(x)\,A(x)$ be the total particle rate through the duct -- constant in $x$, same as $\Phi$
above was for the zero-divergence case. Define the *local* particle flux

$$ \Phi(x) = \frac{\Psi}{A(x)}, $$

which reduces to the constant $\Phi$ above exactly when $A$ is constant. Then $n_p(x) = \Phi(x)/v_p(x)$, same form
as before. Dividing the control-volume momentum and stagnation-energy balances through by $A(x)$ (mass flow *rate*
$\dot m = m_p \Psi$ is still constant, so $\dot m / A(x) = m_p \Phi(x)$):

$$ m_p \Phi(x) \frac{\partial v_p}{\partial x} = J_y B_z - \frac{\partial p_p}{\partial x}, $$
$$ m_p \Phi(x) \frac{\partial h_0}{\partial x} = J_y B_z v_p + S_\Omega. $$

The stagnation-energy equation picks up **no explicit $dA/dx$ term** -- it's identical in form to the zero-divergence
case, just with $\Phi(x)$ in place of a constant. All the new physics enters through the momentum equation's
ideal-gas-law substitution: $p_p(x) = k\Phi(x)T_p(x)/v_p(x)$, and since $\Phi(x) = \Psi/A(x)$,

$$ \frac{\partial \Phi}{\partial x} = -\Phi(x)\cdot\frac{A'(x)}{A(x)}, $$

which, substituted in and with $\partial v_p/\partial x$, $\partial T_p/\partial x$ collected (as before, this
collection step is exactly where an earlier, zero-divergence version of this kind of elimination introduced a
sign error -- see the Correction box above -- so it is the step most worth not trusting by hand alone here either):

$$ \Phi\big(m_pv_p^2 - kT_p\big)\frac{\partial v_p}{\partial x} + k\Phi v_p\frac{\partial T_p}{\partial x}
   = J_yB_zv_p^2 + k\Phi v_p T_p \cdot\frac{A'}{A}, \tag{Eq. A} $$
$$ m_p\Phi v_p\frac{\partial v_p}{\partial x} + m_p\Phi c_p\frac{\partial T_p}{\partial x} = J_yB_zv_p + S_\Omega. \tag{Eq. B} $$

Solving this $2\times2$ linear system for $(\partial v_p/\partial x, \partial T_p/\partial x)$
(with $c_p = \gamma k/((\gamma-1)m_p)$), the determinant is $\Phi^2 k(m_pv_p^2 - \gamma k T_p)/(\gamma-1)$ -- **the
same Rayleigh-flow factor as the zero-divergence case, with no $A'/A$ dependence** -- giving

$$ \frac{\partial T_p}{\partial x} = \frac{\gamma-1}{k}\cdot\frac{S_\Omega m_p v_p^2 - k T_p (J_yB_zv_p + S_\Omega)}{\Phi(x)\big(m_pv_p^2 - \gamma k T_p\big)}
   \;-\; (\gamma-1)\frac{m_pv_p^2 T_p}{m_pv_p^2 - \gamma k T_p}\cdot\frac{A'(x)}{A(x)}, $$

$$ \frac{\partial v_p}{\partial x} = \frac{v_p\big(J_yB_zv_p - S_\Omega(\gamma-1)\big)}{\Phi(x)\big(m_pv_p^2 - \gamma k T_p\big)}
   \;+\; \frac{\gamma k T_p v_p}{m_pv_p^2 - \gamma k T_p}\cdot\frac{A'(x)}{A(x)}. $$

Setting $A'(x)=0$ (so $\Phi(x)$ is the constant $\Phi$ again) recovers the zero-divergence $\partial T_p/\partial x$,
$\partial v_p/\partial x$ above *exactly*, term for term -- the required special case. The $A'/A$ correction term is,
in both equations, independent of $\Phi$/the mass flow rate entirely (it cancels out of the algebra), and has the
physically-expected sign on either side of $M=1$: since the shared denominator $m_pv_p^2 - \gamma kT_p$ is negative
for subsonic flow and positive for supersonic flow, a diverging channel ($A'>0$) makes $\partial v_p/\partial x$
*more negative* when subsonic (further deceleration -- the classical subsonic-diffuser result) but *more positive*
when supersonic (further acceleration, away from $M=1$) -- exactly the mechanism this section exists to use. This
hand-derivation is symbolically verified (both in general, and in the $A'=0$ limit against the equations above) in
`tests/test_tapered_derivation_sympy.py`, encoding Eq. A/Eq. B directly and solving rather than differentiating --
closer to where the historical bug actually was (an algebraic elimination error, not a differentiation one).

Because the shared denominator is untouched by the area terms, `TaperedHallSolver`'s choking/bisection logic (finding
the largest step that doesn't cross $M=1$, from either side) is structurally identical to `HallSolver.march`'s --
only the $\partial T_p/\partial x$, $\partial v_p/\partial x$ evaluated at each step differ.

One more consequence of $A$ varying: `HallSolver.march` converts a single lumped external load resistance $R_{load}$
[$\Omega$] into the load resistivity $\eta_L$ the per-slice closure above actually needs via $\eta_L = R_{load}A/L$
(code only -- not stated as an equation elsewhere in this document). With $A(x)$, this becomes a per-slice quantity,

$$ \eta_L(x) = R_{load}\cdot\frac{A(x)}{L}, $$

varying station to station instead of being one number for the whole channel, but otherwise fed into the same,
unmodified per-slice Ohm's-law closure above.


## Process

At long last,

1. Initialize inlet with pressure $p_{p0}$, velocity $v_{p0}$, and primary gas temperature $T_{p0}$
2. Guess at $\Delta T$
3. Find $T_e$
4. Use the Saha equation to find $n_e$.
5. Then, find $\nu_M$, $\nu_E$, $\eta$, $\beta$, $Z$ and $M$.
6. Use the $\Delta T$ relation to get a better estimate, and repeat steps 2-6 until the residual vanishes.
7. Find $T_p(x)$, $v_p(x)$, and $p_p(x)$ for the next slice using the differential equations for them.
8. Repeat steps 2-8 for all slices.
