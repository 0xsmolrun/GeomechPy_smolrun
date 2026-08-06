# GeomechPy Guru — Geomechanics QC with Jupyter Notebooks

## How GeomechPy powers these visualizations

GeomechPy is a lightweight, dependency-free geomechanics library: every module
under `geomechpy/` exposes plain, well-documented static functions that take
scalars or lists and return numbers — no hidden state, no plotting, no I/O. That
deliberate separation of *computation* from *presentation* is exactly what makes
the library so well-suited to Jupyter notebooks. The notebook owns the data, the
narrative, and the plots; GeomechPy owns the physics. Each function carries a
literature reference and explicit units in its docstring
(`overburden_stress`, `pore_pressure`, `elastic_properties`,
`static_elastic_properties`, `rock_strength`, `stress_calculations`,
`wellbore_stability`, and the `toolbox` rotations), so a notebook cell reads
like the equation it implements and the reviewer can trace every number back to
its source.

The `Quick_MEM` notebook shows how this composes into a full 1-D Mechanical
Earth Model. Working top-down through the standard MEM sequence — overburden,
lithology, pore pressure, dynamic then static elastic properties, rock strength,
horizontal stresses, and finally wellbore stability — each step is a single call
into GeomechPy, and its `*_array` helpers let the whole log be processed sample
by sample against a shared pandas `DataFrame`. Because the functions are pure,
the notebook can immediately turn each returned column into a **QC plot right
where it is computed**: input logs, the overburden profile, the mechanical
stratigraphy from gamma ray, pore pressure, the dynamic-versus-static elastic
crossplots, strength curves, the stress profile, and the drilling mud window.
This is the real contribution of GeomechPy to visualization — it produces clean,
unit-consistent arrays that map directly onto composite well-log tracks, so the
notebook's job is reduced to drawing them. Stresses are displayed in **psi** and
the safe drilling window is expressed as an equivalent mud weight in **ppg**
(`ppg = psi / (0.052 × TVD)`), matching how the results are actually used at the
rig. Rendered with Plotly, the tracks stay interactive (hover, zoom, pan) while
also embedding a static image so the QC is visible in any viewer, including
GitHub. The final master composite log gathers every key output onto one
depth axis for a single-glance review of the entire model.

## Tool for visualization of Kirsch near-wellbore stresses

The `Kirsch_Visualizer` notebook is a focused tool for inspecting the stress
state on the borehole wall — the part of a geomechanical model that governs
breakouts and drilling-induced tensile fractures. It is built entirely on
`geomechpy.near_wellbore_stresses.NearWellboreStressesCalculation`, which
implements the Kirsch solution (Fjaer et al., 2008) for an arbitrarily oriented
well. Given the far-field stresses, pore and mud pressure, well trajectory, and
static Poisson's ratio, `calculate_kirsch_borehole_wall_stresses` returns the
full set of wall-stress components around the circumference — radial (σrr),
tangential/hoop (σθθ), axial (σzz) and the shear terms — and
`calculate_principal_stresses_analytical` reduces the hoop, axial and shear
components to the maximum and minimum principal stresses (σ₁, σ₂) plus the
tortuosity angle. Under the hood these reuse the same `toolbox` tensor rotations
(`rotate_stress_to_shmax`, `rotate_nev_to_toh`) that the MEM workflow uses, so
the near-wellbore analysis is consistent with the far-field model.

Because those functions return angle-indexed NumPy arrays, they lend themselves
naturally to **polar visualization**: the notebook draws each component as a
closed curve around the wellbore (0° = Top-of-Hole, increasing clockwise), so
the lobes of high hoop stress — where breakouts nucleate — and the low-stress
azimuths — where tensile fractures initiate — are immediately obvious. Every
output is plotted for QC: a six-panel polar grid of σrr, σθθ, σzz, σtz, σ₁ and
σ₂ in psi, a cartesian stress-versus-azimuth view (with the tortuosity angle on
a secondary axis), and a side-by-side comparison of a vertical versus a
horizontal well that shows how the stress concentration changes with trajectory.
The result is a compact, re-runnable instrument: change the far-field stresses,
the maximum-horizontal-stress azimuth, the borehole deviation, or the mud weight
at the top of the notebook, re-run, and read the wellbore stability implications
straight off the plots — all powered by GeomechPy's near-wellbore equations.
