# GeomechPy — Example Project

Interactive examples built on top of [GeomechPy](../../geomechpy). Everything
here imports the library directly from the repository root, so nothing needs to
be installed apart from the plotting/UI dependencies.

## Contents

| File | Type | What it does |
| --- | --- | --- |
| `MEM_Calculator.py` | Streamlit app | 1D Mechanical Earth Model + wellbore-stability calculator (logs → stresses → rock properties → mud-weight window), with QC and tornado sensitivity. |
| `near_wellbore_app.py` | Streamlit app | Kirsch near-wellbore borehole-wall stress visualizer. |
| `Quick_MEM.ipynb` | Notebook | End-to-end 1D MEM walkthrough with QC plots. |
| `Kirsch_Visualizer.ipynb` | Notebook | Kirsch near-wellbore stress notebook. |
| `GeomechPy_Guru.md` | Doc | How GeomechPy drives these visualizations. |
| `requirements.txt` | — | Dependencies for both Streamlit apps. |
| `.streamlit/config.toml` | — | Light theme (keeps the app sidebar readable). |

## Running the Streamlit apps

```bash
cd example/Project
pip install -r requirements.txt

streamlit run MEM_Calculator.py       # MEM + wellbore stability calculator
streamlit run near_wellbore_app.py    # near-wellbore stress visualizer
```

Run from this directory so Streamlit picks up `.streamlit/config.toml`. Each app
adds the repository root to `sys.path` automatically, so GeomechPy is imported
straight from the source tree — no package install required.

---

## MEM_Calculator.py — Quick 1D MEM-WBS Calculator

A single-file Mechanical Earth Model builder. Upload a well log (LAS / CSV /
Excel) or load the built-in synthetic well, map the curves, and it runs the full
GeomechPy workflow:

- **Dynamic elastic properties** from DTCO / DTSM / RHOB.
- **Static conversion** (Bradford, Najibi, Fuller, Morales, or custom
  power/linear laws) with a calibration multiplier.
- **Rock strength** — UCS (Plumb, McNally, or constant), tensile strength, and
  friction angle (Lal, GR-linear, or constant).
- **Mechanical stratigraphy** — sandstone/shale flag from a single GR cutoff,
  shown as a track on every plot.
- **Stresses & wellbore stability** — overburden, pore pressure, poroelastic
  horizontal stresses, and the vertical-well **mud-weight window** (breakout →
  loss gradient), onshore or offshore.
- **QC** flagging against standard ranges and a **tornado** sensitivity analysis
  (vary input logs, or the governing parameters of one equation).
- **Oilfield / Metric** unit systems for input and display.

It bundles the calculation engine and the Streamlit UI into one module so it can
be hosted directly. The McNally UCS and GR-linear friction-angle correlations
are implemented inside the app (they are not part of this GeomechPy build);
everything else is delegated to the library.

## near_wellbore_app.py — Near-Wellbore Stresses

An interactive visualizer for the **Kirsch solution** of stresses on the
borehole wall, built on
[`geomechpy/near_wellbore_stresses.py`](../../geomechpy/near_wellbore_stresses.py).

- **Model inputs** in the sidebar — TVD, far-field stresses (Sv, SHmax, Shmin),
  pore and mud pressure, SHmax azimuth, borehole deviation/azimuth, static
  Poisson's ratio, and azimuthal sampling — with stress-regime presets.
- **psi ⇄ ppg unit toggle** — switch every plot, KPI and table between stress in
  psi and the equivalent mud weight in ppg (`= psi / (0.052 × TVD)`).
- **Azimuthal profile** and **trajectory compare** views with interpretation
  guides, plus a data table with CSV download.

Charts are responsive, render on clean white cards, carry legends, and use
spaced tick labels so axis text stays readable.
