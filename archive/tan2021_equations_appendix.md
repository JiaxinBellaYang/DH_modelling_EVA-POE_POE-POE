# Surface Recombination Model — Equation Reference

Equations below define the extended SRH surface recombination model used in
this project.  Equation numbers follow the appendix notation (A1–A12) plus
Eq. (2) for the interface trapped charge.

See `src/bulk_SRH.py` and `src/chargeanddit.py` for the implementations;
each function docstring references the corresponding equation(s).

## Implementation Notes

- **Eqs. (A8)–(A9)**: The band-edge exponential tails are approximated by
  Gaussians centred at Ev and Ec in the code (`Dig_func`).
- **Eq. (A12)**: The code parametrises the Gaussian width as
  `A = 1/(2μ²)`, so `σ(E) = σ₀·exp[−A·(E−E₀)²]`.
- **Eq. (A5)**: In `ns_zero_func_full`, the fixed charge Q is replaced by
  Q_total = Q_f + Q_it to include interface trapped charge self-consistently.
- **Eq. (A5) erratum**: The written form reads `ps + ns − pd − ps` which
  simplifies to `ns − pd`.  The correct fourth term is **nd**, giving
  `ps + ns − pd − nd` (as implemented in the code).

---

## J₀s — single-level surface recombination current (reference only)

$$
J_{0s}
= \frac{q\,n_i^{2}}
{\dfrac{p_s+p_1}{S_{n0}}+\dfrac{n_s+n_1}{S_{p0}}}.
\tag{A1}
$$

## SRH reference concentrations

$$
n_1 = N_c \exp\!\left(-\frac{E_c-E_t}{kT}\right),
\qquad
p_1 = N_v \exp\!\left(-\frac{E_t-E_v}{kT}\right).
\tag{A2}
$$

## Surface recombination velocities (single-level)

$$
S_{p0} = D_{it}\,\sigma_p\,v_{\mathrm{th},p},
\qquad
S_{n0} = D_{it}\,\sigma_n\,v_{\mathrm{th},n}.
\tag{A3}
$$

## Surface carrier concentrations

$$
n_s = n_d \exp\!\left(-\frac{\psi_s}{kT}\right),
\qquad
p_s = p_d \exp\!\left(\frac{\psi_s}{kT}\right).
\tag{A4}
$$

## Surface charge balance (Gauss's law)

$$
p_s + n_s - p_d - n_d + (N_A+N_D)\frac{\psi_s}{kT}
=
\frac{Q^2}{2q\,\varepsilon_{si}\,kT}.
\tag{A5}
$$

> **Note:** The original published form has a typo (`−p_s` instead of `−n_d`
> in the fourth term).  The corrected form above matches the code.

## Energy-resolved surface recombination rate

$$
U_s
=
\int_{E_v}^{E_c}
\frac{p_s n_s - n_i^2}
{\dfrac{p_s+p_1(E)}{S_{n0}(E)}+\dfrac{n_s+n_1(E)}{S_{p0}(E)}}\,dE.
\tag{A6}
$$

## Energy-dependent surface recombination velocities

$$
S_{p0}(E)=D_{it}(E)\,\sigma_p(E)\,v_{\mathrm{th},p},
\qquad
S_{n0}(E)=D_{it}(E)\,\sigma_n(E)\,v_{\mathrm{th},n}.
\tag{A7}
$$

## Dit band-edge tails (exponential form)

$$
D_{it,v}(E) = D_{it,0v}\,\exp\!\left(\frac{E_v-E}{E_{v,\mathrm{trap}}}\right).
\tag{A8}
$$

$$
D_{it,c}(E) = D_{it,0c}\,\exp\!\left(\frac{E-E_c}{E_{c,\mathrm{trap}}}\right).
\tag{A9}
$$

> **Implementation note:** The code approximates these exponential tails with
> Gaussians centred at Ev and Ec via `Dig_func`.

## Gaussian midgap Dit distribution

$$
D_{it,g}(E) = D_{it,0g}\,\exp\!\left[-\frac{1}{2}\left(\frac{E-E_0}{\sigma}\right)^2\right].
\tag{A10}
$$

## Total Dit

$$
D_{it,\mathrm{tot}}(E)=D_{it,v}(E)+D_{it,g}(E)+D_{it,c}(E).
\tag{A11}
$$

## Gaussian capture cross sections

$$
\sigma_{n/p}(E)
= \sigma_0\,\exp\!\left[-\frac{1}{2}\left(\frac{E-E_{0,n/p}}{\mu}\right)^2\right].
\tag{A12}
$$

> **Implementation note:** The code parametrises this as
> `σ₀·exp[−A·(E − E₀)²]` with `A = 1/(2μ²)`.

## Interface trapped charge (amphoteric model)

$$
Q_{it}
= q\int_{E_v}^{E_c}\left\{D_{it,D}(E)\,[1 - f_t(E)] - D_{it,A}(E)\,f_t(E)\right\} dE.
\tag{2}
$$

Where donor states (E < E_mid) contribute +q when empty and acceptor
states (E ≥ E_mid) contribute −q when occupied.
