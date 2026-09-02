"""Near-Wellbore Stresses — interactive Streamlit app.

Standalone visualization tool for the Kirsch solution of stresses on the
borehole wall, built on GeomechPy's
``geomechpy.near_wellbore_stresses.NearWellboreStressesCalculation``.

Run with:
    cd example/Project
    streamlit run near_wellbore_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --- make GeomechPy importable (repo root is two levels up) -----------------
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from geomechpy.near_wellbore_stresses import NearWellboreStressesCalculation
except ModuleNotFoundError as exc:  # pragma: no cover - defensive
    st.error(
        "Could not import GeomechPy. Run this app from the repository (the app "
        "adds the repo root to sys.path automatically):\n\n"
        "`cd example/Project && streamlit run near_wellbore_app.py`"
    )
    st.stop()
    raise exc


# ---------------------------------------------------------------------------
# Page config + theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Near-Wellbore Stresses",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour system (explicit, so charts read the same on any Streamlit theme)
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#eef2f7"
ZERO = "#d1d5db"
CARD_BG = "#ffffff"

COMPONENT_STYLE = {
    "σrr (radial)": ("sigma_rr", "#6b7280"),
    "σθθ (hoop)": ("sigma_tt", "#2563eb"),
    "σzz (axial)": ("sigma_zz", "#059669"),
    "σtz (shear)": ("sigma_tz", "#d97706"),
    "σ₁ (max principal)": ("sigma_1", "#dc2626"),
    "σ₂ (min principal)": ("sigma_2", "#7c3aed"),
}

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
      .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1400px; }

      /* Hero header */
      .nw-hero {
        background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #0ea5e9 100%);
        border-radius: 18px; padding: 1.5rem 1.75rem;
        box-shadow: 0 10px 30px rgba(37,99,235,.25); margin-bottom: 1.1rem;
      }
      .nw-hero h1, .nw-hero p { color: #ffffff !important; }
      .nw-hero h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing:-.01em; }
      .nw-hero p  { margin: .35rem 0 0; opacity: .92; font-size: .95rem; }

      /* KPI cards */
      .nw-kpi {
        background: #fff; border: 1px solid #eef2f7; border-radius: 14px;
        padding: .85rem 1rem; box-shadow: 0 2px 10px rgba(16,24,40,.05); height: 100%;
      }
      .nw-kpi .label { color: #6b7280 !important; font-size: .78rem; font-weight: 600;
                       text-transform: uppercase; letter-spacing:.04em; }
      .nw-kpi .value { color: #111827 !important; font-size: 1.35rem; font-weight: 700; margin-top:.15rem; }
      .nw-kpi .sub   { color: #9ca3af !important; font-size: .74rem; margin-top:.1rem; }

      /* Explanation callout */
      .nw-note {
        background:#f8fafc; border:1px solid #eef2f7; border-left:4px solid #2563eb;
        border-radius:10px; padding:.85rem 1.1rem; margin:.4rem 0 .2rem;
        color:#374151 !important; font-size:.9rem; line-height:1.5;
      }
      .nw-note b { color:#111827; }

      /* Chart wrappers so a plot never clashes with the page background */
      div[data-testid="stPlotlyChart"] {
        background: #fff; border: 1px solid #eef2f7; border-radius: 14px;
        padding: .35rem; box-shadow: 0 2px 10px rgba(16,24,40,.05);
      }

      /* --- Sidebar: force readable dark text on the light panel, on ANY theme --- */
      [data-testid="stSidebar"] { background: #f8fafc !important; border-right: 1px solid #eef2f7; }
      [data-testid="stSidebar"] * { color: #1f2937 !important; }

      @media (max-width: 640px) {
        .nw-hero h1 { font-size: 1.25rem; }
        .block-container { padding-left: .6rem; padding-right: .6rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Presets + defaults
# ---------------------------------------------------------------------------
PRESETS = {
    "Normal faulting (Sv > SHmax > Shmin)": dict(svert=10000.0, shmax=8500.0, shmin=6500.0),
    "Strike-slip (SHmax > Sv > Shmin)": dict(svert=8000.0, shmax=10000.0, shmin=6000.0),
    "Reverse faulting (SHmax > Shmin > Sv)": dict(svert=6000.0, shmax=11000.0, shmin=8500.0),
}
_DEFAULTS = dict(
    svert=10000.0, shmax=8500.0, shmin=6500.0, pp=4500.0, mud=5000.0,
    az=30.0, dev=0.0, bh_az=0.0, pr=0.25, n=361, tvd=8000.0,
)
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

PSI_PER_PPG_FT = 0.052  # pressure gradient of a 1 ppg fluid, psi/ft


def _apply_preset() -> None:
    name = st.session_state.get("preset")
    if name in PRESETS:
        for k, v in PRESETS[name].items():
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Compute (cached) — always in canonical psi
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute(shmin, shmax, svert, pp, mud, az, dev, bh_az, pr, n) -> pd.DataFrame:
    theta = np.linspace(0.0, 360.0, int(n))
    wall = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(
        shmin=shmin, shmax=shmax, svert=svert, pore_pressure=pp,
        shmax_azimuth=az, mud_pressure=mud, theta=theta,
        poisson_ratio_static=pr, borehole_deviation=dev, borehole_azimuth=bh_az,
    )
    prin = NearWellboreStressesCalculation.calculate_principal_stresses_analytical(
        sigma_tt=wall.sigma_tt, sigma_zz=wall.sigma_zz, sigma_tz=wall.sigma_tz,
    )
    return pd.DataFrame({
        "theta": theta,
        "sigma_rr": wall.sigma_rr,
        "sigma_tt": wall.sigma_tt,
        "sigma_zz": wall.sigma_zz,
        "sigma_tz": wall.sigma_tz,
        "sigma_1": prin.sigma_1,
        "sigma_2": prin.sigma_2,
        "tortuosity": prin.theta_tortuosity,
    })


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model inputs")

    st.selectbox("Stress regime preset", ["Custom", *PRESETS.keys()],
                 key="preset", on_change=_apply_preset,
                 help="Pick a regime to auto-fill the far-field stresses, then fine-tune below.")

    st.markdown("**Depth**")
    st.slider("True vertical depth · TVD (ft)", 1000.0, 20000.0, key="tvd", step=100.0,
              help="Used to convert pressures/stresses to an equivalent mud weight in ppg.")

    st.markdown("**Far-field stresses** (psi)")
    st.slider("Vertical stress · Sv", 2000.0, 20000.0, key="svert", step=100.0)
    st.slider("Max horizontal · SHmax", 2000.0, 20000.0, key="shmax", step=100.0)
    st.slider("Min horizontal · Shmin", 2000.0, 20000.0, key="shmin", step=100.0)

    st.markdown("**Pressures** (psi)")
    st.slider("Pore pressure · Pp", 0.0, 15000.0, key="pp", step=100.0)
    st.slider("Mud pressure · Pw", 0.0, 15000.0, key="mud", step=100.0)

    st.markdown("**Orientation** (deg)")
    st.slider("SHmax azimuth (from North)", 0.0, 360.0, key="az", step=1.0)
    st.slider("Borehole deviation (0 = vertical)", 0.0, 90.0, key="dev", step=1.0)
    st.slider("Borehole azimuth", 0.0, 360.0, key="bh_az", step=1.0)

    st.markdown("**Rock & sampling**")
    st.slider("Static Poisson's ratio", 0.05, 0.45, key="pr", step=0.01)
    st.slider("Azimuthal samples", 91, 721, key="n", step=90,
              help="Number of points around the wellbore circumference.")

    if st.session_state.shmin > st.session_state.shmax:
        st.warning("Shmin > SHmax — check your inputs (Shmin should be the minimum).")

s = st.session_state
df = compute(s.shmin, s.shmax, s.svert, s.pp, s.mud, s.az, s.dev, s.bh_az, s.pr, s.n)
theta = df["theta"].to_numpy()

# Derived (in psi)
i_max = int(df["sigma_tt"].idxmax())
i_min = int(df["sigma_tt"].idxmin())
breakout_az = float(df["theta"].iloc[i_max])
tensile_az = float(df["theta"].iloc[i_min])
max_hoop = float(df["sigma_tt"].max())
min_hoop = float(df["sigma_tt"].min())
max_s1 = float(df["sigma_1"].max())


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="nw-hero">
      <h1>🛢️ Near-Wellbore Stresses</h1>
      <p>Kirsch borehole-wall stress solution · powered by
      <b>GeomechPy</b> · SHmax az {s.az:.0f}° · deviation {s.dev:.0f}° ·
      Pw {s.mud:.0f} psi · TVD {s.tvd:.0f} ft</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Unit toggle (psi <-> ppg). Switches every plot + KPI + table -----------
uc, tc = st.columns([2, 3])
with uc:
    units = st.radio("Pressure units", ["psi", "ppg (mud weight)"],
                     horizontal=True, key="units",
                     help="ppg = psi / (0.052 × TVD). Compare stresses directly against mud weight.")
PPG = units.startswith("ppg")
USUFFIX = "ppg" if PPG else "psi"


def to_unit(v):
    """Convert a pressure/stress in psi to the selected display unit."""
    if PPG:
        return v / (PSI_PER_PPG_FT * s.tvd)
    return v


def fmt(v):
    return f"{to_unit(v):.2f} ppg" if PPG else f"{v:,.0f} psi"


with tc:
    st.markdown(
        f"<div style='padding-top:1.9rem;color:{MUTED};font-size:.85rem'>"
        f"Showing stresses in <b>{USUFFIX}</b>"
        + (f" at TVD {s.tvd:.0f} ft · mud weight Pw = {to_unit(s.mud):.2f} ppg" if PPG else "")
        + "</div>",
        unsafe_allow_html=True,
    )

# Reference pressures in current units
Pw_u = to_unit(s.mud)
Pp_u = to_unit(s.pp)


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="nw-kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Max hoop σθθ", fmt(max_hoop), "breakout-prone")
kpi(k2, "Min hoop σθθ", fmt(min_hoop), "tensile-prone")
kpi(k3, "Breakout azimuth", f"{breakout_az:.0f}°", "max σθθ around wall")
kpi(k4, "Max principal σ₁", fmt(max_s1), "peak wall stress")

st.write("")


# ---------------------------------------------------------------------------
# Figure styling helpers (no fixed width -> always fits its container)
# ---------------------------------------------------------------------------
def _base_layout(fig, height, title=""):
    fig.update_layout(
        height=height, autosize=True,
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
        margin=dict(l=64, r=28, t=52 if title else 30, b=104),
        title=dict(text=title, font=dict(size=16, color="#111827"), x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=ZERO, linecolor=ZERO,
                     tickfont=dict(size=12, color=MUTED), title_font=dict(size=13, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=ZERO, linecolor=ZERO,
                     tickfont=dict(size=12, color=MUTED), title_font=dict(size=13, color=MUTED))
    return fig


def _add_reference_lines(fig, secondary=False):
    """Mud weight (Pw) and pore pressure (Pp) reference lines, in current units."""
    kw = dict(secondary_y=False) if secondary else {}
    fig.add_trace(go.Scatter(
        x=[0, 360], y=[Pw_u, Pw_u], mode="lines", name=f"Mud weight Pw ({USUFFIX})",
        line=dict(color="#0ea5e9", width=1.6, dash="dash"),
        hovertemplate=f"Pw = {Pw_u:,.2f} {USUFFIX}<extra></extra>"), **kw)
    fig.add_trace(go.Scatter(
        x=[0, 360], y=[Pp_u, Pp_u], mode="lines", name=f"Pore pressure Pp ({USUFFIX})",
        line=dict(color="#94a3b8", width=1.4, dash="dot"),
        hovertemplate=f"Pp = {Pp_u:,.2f} {USUFFIX}<extra></extra>"), **kw)


CHART_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_azim, tab_traj, tab_data = st.tabs(
    ["📈 Azimuthal profile", "🎯 Trajectory compare", "🗂️ Data"]
)

# ---- Azimuthal profile ----------------------------------------------------
with tab_azim:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for name, (col, color) in COMPONENT_STYLE.items():
        dash = "dash" if col == "sigma_rr" else None
        fig.add_trace(go.Scatter(
            x=theta, y=to_unit(df[col]), mode="lines", name=name,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="%{x:.0f}°: %{y:,.1f} " + USUFFIX + "<extra>" + name + "</extra>",
        ), secondary_y=False)
    _add_reference_lines(fig, secondary=True)
    fig.add_trace(go.Scatter(
        x=theta, y=df["tortuosity"], mode="lines", name="tortuosity θ (deg)",
        line=dict(color="#0f766e", width=1.5, dash="dot"),
        hovertemplate="%{x:.0f}°: %{y:.1f}°<extra>tortuosity</extra>",
    ), secondary_y=True)
    for az_mark, clr in [(breakout_az, "#dc2626"), (tensile_az, "#0ea5e9")]:
        fig.add_vline(x=az_mark, line=dict(color=clr, width=1, dash="dot"))
    _base_layout(fig, height=520, title=f"Wall stress vs azimuth ({USUFFIX})")
    fig.update_xaxes(title_text="θ — azimuth from Top-of-Hole (deg)", range=[0, 360], dtick=45)
    fig.update_yaxes(title_text=f"Stress ({USUFFIX})", secondary_y=False)
    fig.update_yaxes(title_text="Tortuosity angle (deg)", secondary_y=True,
                     showgrid=False, tickfont=dict(size=12, color=MUTED))
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    st.markdown(
        f"""
        <div class="nw-note">
        <b>How to read this.</b> The x-axis is the position around the borehole wall
        (θ = 0° at the Top-of-Hole, sweeping clockwise). Each curve is a stress
        component acting on the wall, in <b>{USUFFIX}</b>.
        <ul style="margin:.4rem 0 0 .1rem">
          <li><b>σθθ (hoop stress)</b> is the key curve. Where it <b>peaks</b>
              (red dotted line, breakout azimuth ≈ {breakout_az:.0f}°) the wall is most
              compressed — <b>breakouts / borehole collapse</b> start here when σθθ exceeds
              the rock strength. Where it is <b>lowest</b> (blue dotted line ≈ {tensile_az:.0f}°)
              the wall can go into tension — <b>drilling-induced tensile fractures</b>
              initiate here.</li>
          <li><b>σ₁ / σ₂</b> are the maximum / minimum principal stresses on the wall;
              <b>σzz</b> is axial, <b>σrr</b> = Pw − Pp is the radial support from the mud.</li>
          <li>Switch the units to <b>ppg</b> and read every curve against the dashed
              <b>mud-weight line (Pw)</b>: raising mud weight lifts σrr and lowers σθθ
              (helps against collapse) but must stay below the fracture gradient to avoid losses.</li>
          <li><b>Tortuosity</b> (right axis, degrees) is the tilt of the principal-stress
              plane; it stays near 0° for a vertical well and grows as the well is deviated.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Trajectory compare ---------------------------------------------------
with tab_traj:
    c1, c2 = st.columns([1, 2])
    with c1:
        metric = st.radio("Quantity", ["σθθ (hoop)", "σ₁ (max principal)"])
    with c2:
        devs = st.multiselect("Deviations to compare (deg)", [0, 15, 30, 45, 60, 75, 90],
                              default=[0, 30, 60, 90])
    mcol = "sigma_tt" if metric.startswith("σθθ") else "sigma_1"
    palette = ["#1e3a8a", "#2563eb", "#0ea5e9", "#059669", "#d97706", "#dc2626", "#7c3aed"]
    fig = go.Figure()
    for j, dv in enumerate(sorted(devs)):
        d = compute(s.shmin, s.shmax, s.svert, s.pp, s.mud, s.az, float(dv), s.bh_az, s.pr, s.n)
        fig.add_trace(go.Scatter(
            x=d["theta"], y=to_unit(d[mcol]), mode="lines", name=f"deviation {dv}°",
            line=dict(color=palette[j % len(palette)], width=2),
            hovertemplate="%{x:.0f}°: %{y:,.1f} " + USUFFIX + "<extra>dev " + str(dv) + "°</extra>",
        ))
    _add_reference_lines(fig)
    _base_layout(fig, height=520, title=f"{metric} vs azimuth by deviation ({USUFFIX})")
    fig.update_xaxes(title_text="θ — azimuth from Top-of-Hole (deg)", range=[0, 360], dtick=45)
    fig.update_yaxes(title_text=f"Stress ({USUFFIX})")
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    st.markdown(
        f"""
        <div class="nw-note">
        <b>How to read this.</b> Each curve is the same stress component
        (<b>{metric}</b>) around the wall, but for a different <b>borehole deviation</b>
        (0° = vertical, 90° = horizontal) with the far-field stresses held fixed — so the
        chart isolates the effect of the <b>well trajectory</b>.
        <ul style="margin:.4rem 0 0 .1rem">
          <li>A curve with a <b>higher peak</b> means a stronger stress concentration on the
              wall — that trajectory is <b>more prone to breakout</b> and needs more support
              (higher mud weight or higher rock strength).</li>
          <li>A <b>flatter</b> curve means the stress is more uniform around the wall — a more
              <b>stable</b> trajectory.</li>
          <li>Compare the peaks against the dashed <b>mud-weight (Pw)</b> and rock-strength
              limits (in <b>ppg</b> mode) to choose the deviation/azimuth that keeps the well
              inside a safe operating window.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Data -----------------------------------------------------------------
with tab_data:
    st.caption(f"Computed borehole-wall stresses at every azimuth (values in {USUFFIX}).")
    out = pd.DataFrame({"theta": df["theta"], "tortuosity": df["tortuosity"]})
    for label, (col, _c) in COMPONENT_STYLE.items():
        out[col] = to_unit(df[col])
    out = out[["theta", "sigma_rr", "sigma_tt", "sigma_zz", "sigma_tz",
               "sigma_1", "sigma_2", "tortuosity"]]
    show = out.copy()
    show.columns = ["θ (deg)", f"σrr ({USUFFIX})", f"σθθ ({USUFFIX})", f"σzz ({USUFFIX})",
                    f"σtz ({USUFFIX})", f"σ₁ ({USUFFIX})", f"σ₂ ({USUFFIX})", "tortuosity (deg)"]
    st.dataframe(show.style.format("{:.2f}"), use_container_width=True, height=420)
    st.download_button(
        f"⬇️ Download CSV ({USUFFIX})", out.to_csv(index=False).encode("utf-8"),
        file_name=f"near_wellbore_stresses_{USUFFIX}.csv", mime="text/csv",
    )

st.markdown(
    f"<p style='color:{MUTED};font-size:.8rem;margin-top:1.4rem'>"
    "Model: Kirsch solution · <code>geomechpy.near_wellbore_stresses</code>. "
    "σrt and σrz are identically zero at the wall by the Kirsch boundary conditions. "
    "ppg equivalent = psi ÷ (0.052 × TVD).</p>",
    unsafe_allow_html=True,
)
