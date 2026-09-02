# Near-Wellbore Stresses — Streamlit app

An interactive, standalone visualization tool for the **Kirsch solution** of
stresses on the borehole wall, built on GeomechPy's
[`geomechpy/near_wellbore_stresses.py`](../../geomechpy/near_wellbore_stresses.py).

![overview](https://img.shields.io/badge/streamlit-app-ff4b4b)

## Features

- **Model inputs** in the sidebar — far-field stresses (Sv, SHmax, Shmin),
  pore and mud pressure, SHmax azimuth, borehole deviation/azimuth, static
  Poisson's ratio, and azimuthal sampling — with ready-made stress-regime
  presets (normal / strike-slip / reverse faulting).
- **KPIs** — maximum/minimum hoop stress, breakout azimuth, and peak principal
  stress, updated live.
- **Polar view** — overlay any wall-stress components (σrr, σθθ, σzz, σtz, σ₁,
  σ₂) around the wellbore, with breakout/tensile azimuths marked.
- **Azimuthal profile** — every component vs azimuth in psi, with the tortuosity
  angle on a secondary axis.
- **Trajectory compare** — how the hoop / principal stress concentration changes
  with borehole deviation.
- **Data** — the full computed table with CSV download.

All stresses are in **psi**. Charts are responsive (fit any screen), render on
clean white cards so they never clash with the page background, and use spaced,
sized tick labels so axis text stays readable.

## Run

From the repository root:

```bash
pip install -r example/Project/requirements.txt
streamlit run example/Project/near_wellbore_app.py
```

The app adds the repository root to `sys.path` automatically, so GeomechPy is
imported directly from the source tree — no install of the package required.
