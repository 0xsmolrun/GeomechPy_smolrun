# Near-Wellbore Stresses — Streamlit app

An interactive, standalone visualization tool for the **Kirsch solution** of
stresses on the borehole wall, built on GeomechPy's
[`geomechpy/near_wellbore_stresses.py`](../../geomechpy/near_wellbore_stresses.py).

![overview](https://img.shields.io/badge/streamlit-app-ff4b4b)

## Features

- **Model inputs** in the sidebar — TVD, far-field stresses (Sv, SHmax, Shmin),
  pore and mud pressure, SHmax azimuth, borehole deviation/azimuth, static
  Poisson's ratio, and azimuthal sampling — with ready-made stress-regime
  presets (normal / strike-slip / reverse faulting).
- **psi ⇄ ppg unit toggle** — switch every plot, KPI and table between stress in
  psi and the equivalent mud weight in ppg (`= psi / (0.052 × TVD)`), so you can
  compare wall stresses directly against your mud weight.
- **KPIs** — maximum/minimum hoop stress, breakout azimuth, and peak principal
  stress, in the selected unit.
- **Azimuthal profile** — every component vs azimuth, with the tortuosity angle
  on a secondary axis, mud-weight/pore-pressure reference lines, and an
  interpretation guide.
- **Trajectory compare** — how the hoop / principal stress concentration changes
  with borehole deviation, with an interpretation guide.
- **Data** — the full computed table with CSV download in the selected unit.

Charts are responsive (fit any screen), render on clean white cards so they
never clash with the page background, carry legends, and use spaced, sized tick
labels so axis text stays readable.

## Run

```bash
cd example/Project
pip install -r requirements.txt
streamlit run near_wellbore_app.py
```

Running from this directory picks up `.streamlit/config.toml` (a light theme that
keeps the sidebar readable). The app adds the repository root to `sys.path`
automatically, so GeomechPy is imported directly from the source tree — no
install of the package required.
