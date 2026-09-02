"""Near-Wellbore Stresses — interactive Streamlit app.

Standalone visualization tool for the Kirsch solution of stresses on the
borehole wall, built on GeomechPy's
``geomechpy.near_wellbore_stresses.NearWellboreStressesCalculation``.

Run with:
    streamlit run example/Project/near_wellbore_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
        "Could not import GeomechPy. Run this app from the repository root:\n\n"
        "`streamlit run example/Project/near_wellbore_app.py`"
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
ACCENT = "#2563eb"

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
      /* Modern, readable base */
      html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
      .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1400px; }

      /* Hero header */
      .nw-hero {
        background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #0ea5e9 100%);
        border-radius: 18px; padding: 1.5rem 1.75rem; color: #fff;
        box-shadow: 0 10px 30px rgba(37,99,235,.25); margin-bottom: 1.1rem;
      }
      .nw-hero h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing:-.01em; }
      .nw-hero p  { margin: .35rem 0 0; opacity: .92; font-size: .95rem; }

      /* KPI cards */
      .nw-kpi {
        background: #fff; border: 1px solid #eef2f7; border-radius: 14px;
        padding: .85rem 1rem; box-shadow: 0 2px 10px rgba(16,24,40,.05); height: 100%;
      }
      .nw-kpi .label { color: #6b7280; font-size: .78rem; font-weight: 600;
                       text-transform: uppercase; letter-spacing:.04em; }
      .nw-kpi .value { color: #111827; font-size: 1.35rem; font-weight: 700; margin-top:.15rem; }
      .nw-kpi .sub   { color: #9ca3af; font-size: .74rem; margin-top:.1rem; }

      /* Chart wrappers so a plot never clashes with the page background */
      div[data-testid="stPlotlyChart"] {
        background: #fff; border: 1px solid #eef2f7; border-radius: 14px;
        padding: .35rem; box-shadow: 0 2px 10px rgba(16,24,40,.05);
      }
      /* Responsive tweaks for small screens */
      @media (max-width: 640px) {
        .nw-hero h1 { font-size: 1.25rem; }
        .block-container { padding-left: .6rem; padding-right: .6rem; }
      }
      [data-testid="stSidebar"] { background: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "Normal faulting (Sv > SHmax > Shmin)": dict(svert=10000.0, shmax=8500.0, shmin=6500.0),
    "Strike-slip (SHmax > Sv > Shmin)": dict(svert=8000.0, shmax=10000.0, shmin=6000.0),
    "Reverse faulting (SHmax > Shmin > Sv)": dict(svert=6000.0, shmax=11000.0, shmin=8500.0),
}
_DEFAULTS = dict(
    svert=10000.0, shmax=8500.0, shmin=6500.0, pp=4500.0, mud=5000.0,
    az=30.0, dev=0.0, bh_az=0.0, pr=0.25, n=361,
)
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _apply_preset() -> None:
    name = st.session_state.get("preset")
    if name in PRESETS:
        for k, v in PRESETS[name].items():
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Compute (cached)
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
# Figure styling helpers (no fixed width -> always fits its container)
# ---------------------------------------------------------------------------
def _base_layout(fig: go.Figure, height: int, title: str = "") -> go.Figure:
    fig.update_layout(
        height=height, autosize=True,
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
        margin=dict(l=64, r=28, t=64 if title else 34, b=56),
        title=dict(text=title, font=dict(size=16, color="#111827"), x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hovermode="x unified",
    )
    return fig


def _style_cartesian(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=ZERO, linecolor=ZERO,
                     tickfont=dict(size=12, color=MUTED), title_font=dict(size=13, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=ZERO, linecolor=ZERO,
                     tickfont=dict(size=12, color=MUTED), title_font=dict(size=13, color=MUTED))
    return fig


CHART_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model inputs")

    st.selectbox("Stress regime preset", ["Custom", *PRESETS.keys()],
                 key="preset", on_change=_apply_preset,
                 help="Pick a regime to auto-fill the far-field stresses, then fine-tune below.")

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

# Derived quantities
i_max = int(df["sigma_tt"].idxmax())
i_min = int(df["sigma_tt"].idxmin())
breakout_az = float(df["theta"].iloc[i_max])
tensile_az = float(df["theta"].iloc[i_min])
max_hoop = float(df["sigma_tt"].max())
min_hoop = float(df["sigma_tt"].min())
max_s1 = float(df["sigma_1"].max())


# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="nw-hero">
      <h1>🛢️ Near-Wellbore Stresses</h1>
      <p>Kirsch borehole-wall stress solution · powered by
      <b>GeomechPy</b> · SHmax az {s.az:.0f}° · deviation {s.dev:.0f}° · Pw {s.mud:.0f} psi</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="nw-kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Max hoop σθθ", f"{max_hoop:,.0f} psi", "breakout-prone")
kpi(k2, "Min hoop σθθ", f"{min_hoop:,.0f} psi", "tensile-prone")
kpi(k3, "Breakout azimuth", f"{breakout_az:.0f}°", "max σθθ around wall")
kpi(k4, "Max principal σ₁", f"{max_s1:,.0f} psi", "peak wall stress")

st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_polar, tab_azim, tab_traj, tab_data = st.tabs(
    ["🧭 Polar view", "📈 Azimuthal profile", "🎯 Trajectory compare", "🗂️ Data"]
)

# ---- Polar view -----------------------------------------------------------
with tab_polar:
    st.caption("Stress magnitude around the wellbore wall. 0° = Top-of-Hole, clockwise.")
    picks = st.multiselect(
        "Components to overlay",
        list(COMPONENT_STYLE.keys()),
        default=["σθθ (hoop)", "σ₁ (max principal)"],
    )
    fig = go.Figure()
    for name in picks:
        col, color = COMPONENT_STYLE[name]
        single = len(picks) == 1
        fig.add_trace(go.Scatterpolar(
            r=df[col], theta=theta, mode="lines", name=name,
            line=dict(color=color, width=2.5),
            fill="toself" if single else None,
            hovertemplate="%{theta:.0f}°<br>%{r:,.0f} psi<extra>" + name + "</extra>",
        ))
    # markers for breakout / tensile azimuths on the hoop stress
    fig.add_trace(go.Scatterpolar(
        r=[max_hoop, min_hoop], theta=[breakout_az, tensile_az], mode="markers",
        name="breakout / tensile", marker=dict(size=11, color=["#dc2626", "#0ea5e9"],
                                                symbol=["circle", "diamond"], line=dict(color="#fff", width=1.5)),
        hovertemplate="%{theta:.0f}°<br>%{r:,.0f} psi<extra></extra>",
    ))
    _base_layout(fig, height=560, title="Borehole-wall stresses (psi)")
    fig.update_layout(polar=dict(
        bgcolor=CARD_BG,
        radialaxis=dict(gridcolor=GRID, linecolor=ZERO, angle=45, tickangle=45,
                        tickfont=dict(size=10, color=MUTED), ticksuffix=""),
        angularaxis=dict(direction="clockwise", rotation=90, gridcolor=GRID,
                         linecolor=ZERO, tickfont=dict(size=11, color=MUTED), dtick=45),
    ))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

# ---- Azimuthal profile ----------------------------------------------------
with tab_azim:
    st.caption("All wall-stress components versus azimuth. Tortuosity angle on the right axis.")
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for name, (col, color) in COMPONENT_STYLE.items():
        dash = "dash" if col == "sigma_rr" else None
        fig.add_trace(go.Scatter(
            x=theta, y=df[col], mode="lines", name=name,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="%{x:.0f}°: %{y:,.0f} psi<extra>" + name + "</extra>",
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=theta, y=df["tortuosity"], mode="lines", name="tortuosity θ",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        hovertemplate="%{x:.0f}°: %{y:.1f}°<extra>tortuosity</extra>",
    ), secondary_y=True)
    fig.add_hline(y=0, line=dict(color=ZERO, width=1))
    for az_mark, lbl, clr in [(breakout_az, "breakout", "#dc2626"), (tensile_az, "tensile", "#0ea5e9")]:
        fig.add_vline(x=az_mark, line=dict(color=clr, width=1, dash="dot"))
    _base_layout(fig, height=520, title="Stress vs azimuth (psi)")
    _style_cartesian(fig)
    fig.update_xaxes(title_text="θ — azimuth from Top-of-Hole (deg)", range=[0, 360], dtick=45)
    fig.update_yaxes(title_text="Stress (psi)", secondary_y=False)
    fig.update_yaxes(title_text="Tortuosity angle (deg)", secondary_y=True,
                     showgrid=False, tickfont=dict(size=12, color=MUTED))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

# ---- Trajectory compare ---------------------------------------------------
with tab_traj:
    st.caption("How the hoop / principal stress concentration changes with borehole deviation "
               "(same far-field stresses).")
    metric = st.radio("Quantity", ["σθθ (hoop)", "σ₁ (max principal)"], horizontal=True)
    devs = st.multiselect("Deviations to compare (deg)", [0, 15, 30, 45, 60, 75, 90],
                          default=[0, 30, 60, 90])
    mcol = "sigma_tt" if metric.startswith("σθθ") else "sigma_1"
    palette = ["#1e3a8a", "#2563eb", "#0ea5e9", "#059669", "#d97706", "#dc2626", "#7c3aed"]
    fig = go.Figure()
    for j, dv in enumerate(sorted(devs)):
        d = compute(s.shmin, s.shmax, s.svert, s.pp, s.mud, s.az, float(dv), s.bh_az, s.pr, s.n)
        fig.add_trace(go.Scatter(
            x=d["theta"], y=d[mcol], mode="lines", name=f"dev {dv}°",
            line=dict(color=palette[j % len(palette)], width=2),
            hovertemplate="%{x:.0f}°: %{y:,.0f} psi<extra>dev " + str(dv) + "°</extra>",
        ))
    _base_layout(fig, height=520, title=f"{metric} vs azimuth by deviation (psi)")
    _style_cartesian(fig)
    fig.update_xaxes(title_text="θ — azimuth from Top-of-Hole (deg)", range=[0, 360], dtick=45)
    fig.update_yaxes(title_text="Stress (psi)")
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

# ---- Data -----------------------------------------------------------------
with tab_data:
    st.caption("Computed borehole-wall stresses at every azimuth (psi).")
    show = df.copy()
    show.columns = ["θ (deg)", "σrr", "σθθ", "σzz", "σtz", "σ₁", "σ₂", "tortuosity (deg)"]
    st.dataframe(show.style.format("{:.1f}"), use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
        file_name="near_wellbore_stresses.csv", mime="text/csv",
    )

st.markdown(
    f"<p style='color:{MUTED};font-size:.8rem;margin-top:1.4rem'>"
    "Model: Kirsch solution · <code>geomechpy.near_wellbore_stresses</code>. "
    "σrt and σrz are identically zero at the wall by the Kirsch boundary conditions.</p>",
    unsafe_allow_html=True,
)
