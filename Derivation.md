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
Do
where we can assume $\textbf{B}_0$ is uniform and only in the $z$-direction, the flow is along the $x$-direction so $v_{py}=v_{pz}=0$, the electrodes 
are shorted in the $y$-direction in this setup so $E_y=0$, and no current or voltage flows in the $z$-direction so $E_z=J_z=0$. Ohm's law becoems (Friedberg 3.7),

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

We'll also need to consider the primary gas temperature in the constant area case,

$$ m_p n_p v_p c_p\frac{\partial T_p}{\partial x} = S_\Omega - S_L$$
$$ m_p n_p v_p c_p\frac{\partial T_p}{\partial x} = \frac{m_e n_e \nu_M}{(\beta^2 + 1 + Z)^2} \bigg( v_p^2 \beta^2 [ \beta^2 + (1 + Z)^2] - \beta^4 Z\bigg )$$
$$ \frac{\partial T_p}{\partial x} = \frac{\nu_E }{2 v_p c_p(\beta^2 + 1 + Z)^2} \bigg( v_p^2 \beta^2 [ \beta^2 + (1 + Z)^2] - \beta^4 Z\bigg )$$

If we then assume that the primary gas acts as an ideal gas,

$$ p_p = n_p k T_p, $$
$$ p_p = \frac{\Phi}{v_p} k T_p, $$
$$ \frac{\partial p_p}{\partial x} = \frac{\partial}{\partial x}\bigg( \frac{\Phi}{v_p} k T_p \bigg), $$
$$ \frac{\partial p_p}{\partial x} =  \Phi k \bigg( \frac{1}{v_p} \frac{\partial T_p}{\partial x} - \frac{T_p}{v_p^2}\frac{\partial v_p}{\partial x} \bigg), $$

substituting into momentum conservation,

$$ \Phi m_p \frac{\partial v_p}{\partial x} = J_y B_z - \Phi k \bigg( \frac{1}{v_p} \frac{\partial T_p}{\partial x} - \frac{T_p}{v_p^2}\frac{\partial v_p}{\partial x} \bigg), $$
$$ \frac{\partial v_p}{\partial x} ( \Phi m_p + \Phi k\frac{T_p}{v_p^2}) = J_y B_z - \Phi k \frac{1}{v_p} \frac{\partial T_p}{\partial x}, $$
$$ \frac{\partial v_p}{\partial x}  = \frac{ J_y B_z v_p^2 - \Phi k v_p \frac{\partial T_p}{\partial x}}{m_p \Phi v_p^2 + \Phi k T_p}. $$


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