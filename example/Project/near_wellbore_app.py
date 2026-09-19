"""Near-Wellbore Stresses — interactive Streamlit app.

Standalone visualization tool for the Kirsch solution of stresses on the
borehole wall, built on GeomechPy's
``geomechpy.near_wellbore_stresses.NearWellboreStressesCalculation``.

Includes a Bratton et al. (SPWLA, 1999) style failure analysis: principal
wall stresses and Delta-Stability for each shear / tensile failure mode as a
function of mud density, with the safe mud-weight window.

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
PSI_PER_PPG_FT = 0.052  # pressure gradient of a 1 ppg fluid, psi/ft

# Wall-stress components (canonical df column -> display name, colour)
COMPONENT_STYLE = {
    "σrr (radial)": ("sigma_rr", "#6b7280"),
    "σθθ (hoop)": ("sigma_tt", "#2563eb"),
    "σzz (axial)": ("sigma_zz", "#059669"),
    "σtz (shear)": ("sigma_tz", "#d97706"),
    "σ₁ (max principal)": ("sigma_1", "#dc2626"),
    "σ₂ (intermediate)": ("sigma_2", "#7c3aed"),
    "σ₃ (min principal)": ("sigma_3", "#0f766e"),
}

# Bratton failure modes: label -> (sigma1_key, sigma3_key) for shear,
# or (key, None) for tensile. Keys index the effective wall stresses r/t/a.
SHEAR_MODES = {
    "Swbo — wide breakout":   ("t", "r"),
    "Ssko — shallow knockout": ("a", "r"),
    "Shae — high-angle echelon": ("a", "t"),
    "Snbo — narrow breakout": ("r", "t"),
    "Slae — low-angle echelon": ("t", "a"),
    "Sdko — deep knockout":   ("r", "a"),
}
TENSILE_MODES = {
    "Tcyl — cylindrical": "r",
    "Thor — horizontal":  "a",
    "Tver — vertical":    "t",
}
MODE_COLORS = {
    "Swbo — wide breakout": "#dc2626", "Ssko — shallow knockout": "#ea580c",
    "Shae — high-angle echelon": "#d97706", "Snbo — narrow breakout": "#b45309",
    "Slae — low-angle echelon": "#a16207", "Sdko — deep knockout": "#854d0e",
    "Tcyl — cylindrical": "#2563eb", "Thor — horizontal": "#0ea5e9",
    "Tver — vertical": "#7c3aed",
}

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
      .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1400px; }

      .nw-hero {
        background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #0ea5e9 100%);
        border-radius: 18px; padding: 1.5rem 1.75rem;
        box-shadow: 0 10px 30px rgba(37,99,235,.25); margin-bottom: 1.1rem;
      }
      .nw-hero h1, .nw-hero p { color: #ffffff !important; }
      .nw-hero h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing:-.01em; }
      .nw-hero p  { margin: .35rem 0 0; opacity: .92; font-size: .95rem; }

      .nw-kpi {
        background: #fff; border: 1px solid #eef2f7; border-radius: 14px;
        padding: .85rem 1rem; box-shadow: 0 2px 10px rgba(16,24,40,.05); height: 100%;
      }
      .nw-kpi .label { color: #6b7280 !important; font-size: .78rem; font-weight: 600;
                       text-transform: uppercase; letter-spacing:.04em; }
      .nw-kpi .value { color: #111827 !important; font-size: 1.35rem; font-weight: 700; margin-top:.15rem; }
      .nw-kpi .sub   { color: #9ca3af !important; font-size: .74rem; margin-top:.1rem; }

      .nw-note {
        background:#f8fafc; border:1px solid #eef2f7; border-left:4px solid #2563eb;
        border-radius:10px; padding:.85rem 1.1rem; margin:.4rem 0 .2rem;
        color:#374151 !important; font-size:.9rem; line-height:1.5;
      }
      .nw-note b { color:#111827; }

      div[data-testid="stPlotlyChart"] {
        background: #fff; border: 1px solid #eef2f7; border-radius: 14px;
        padding: .35rem; box-shadow: 0 2px 10px rgba(16,24,40,.05);
      }

      /* Sidebar: force readable dark text on the light panel, on ANY theme */
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
    ucs=7500.0, fang=30.0, tstr=300.0, biot=1.0,
)
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _apply_preset() -> None:
    name = st.session_state.get("preset")
    if name in PRESETS:
        for k, v in PRESETS[name].items():
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Compute (cached) — near-wellbore wall stresses (library, all in psi)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_wall(shmin, shmax, svert, pp, mud, az, dev, bh_az, pr, n) -> pd.DataFrame:
    theta = np.linspace(0.0, 360.0, int(n))
    wall = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(
        shmin=shmin, shmax=shmax, svert=svert, pore_pressure=pp,
        shmax_azimuth=az, mud_pressure=mud, theta=theta,
        poisson_ratio_static=pr, borehole_deviation=dev, borehole_azimuth=bh_az,
    )
    prin = NearWellboreStressesCalculation.calculate_principal_stresses_analytical_boreholewall(
        sigma_boreholewall_rr=wall.sigma_boreholewall_rr,
        sigma_boreholewall_tt=wall.sigma_boreholewall_tt,
        sigma_boreholewall_zz=wall.sigma_boreholewall_zz,
        sigma_boreholewall_tz=wall.sigma_boreholewall_tz,
    )
    return pd.DataFrame({
        "theta": theta,
        "sigma_rr": wall.sigma_boreholewall_rr,
        "sigma_tt": wall.sigma_boreholewall_tt,
        "sigma_zz": wall.sigma_boreholewall_zz,
        "sigma_tz": wall.sigma_boreholewall_tz,
        "sigma_1": prin.sigma_boreholewall_1,
        "sigma_2": prin.sigma_boreholewall_2,
        "sigma_3": prin.sigma_boreholewall_3,
        "tortuosity": prin.theta_boreholewall_tortuosity,
    })


@st.cache_data(show_spinner=False)
def bratton_analysis(shmax, shmin, svert, pp, tvd, nu, biot, ucs, fang, tstr,
                     mw_min, mw_max, npts=121):
    """Bratton et al. (1999) failure analysis for a vertical well.

    Effective wall stresses (total - biot*Pp) at the two critical azimuths, and
    the Delta-Stability (Eq. 5/6) of each shear and tensile failure mode versus
    mud density. Returns a dict of per-azimuth DataFrames plus the safe
    mud-weight window (collapse -> fracture).
    """
    mw = np.linspace(mw_min, mw_max, int(npts))
    pw = PSI_PER_PPG_FT * mw * tvd  # psi
    N = np.tan(np.deg2rad(45.0 + fang / 2.0)) ** 2
    d_sh = shmax - shmin

    def eff_stresses(cos2b):
        # cos2b = -1 at Shmin azimuth (breakout), +1 at SHmax azimuth (fracture)
        sr = pw - biot * pp
        st_ = (shmax + shmin) - 2.0 * d_sh * cos2b - pw - biot * pp
        sa = svert - 2.0 * nu * d_sh * cos2b - biot * pp
        return {"r": sr, "t": st_, "a": sa}

    out = {}
    for tag, cos2b in (("Shmin azimuth (breakout)", -1.0), ("SHmax azimuth (fracture)", 1.0)):
        s = eff_stresses(cos2b)
        df = pd.DataFrame({"MW": mw, "sigma_r": s["r"], "sigma_t": s["t"], "sigma_a": s["a"]})
        for mode, (k1, k3) in SHEAR_MODES.items():
            df[mode] = ucs + s[k3] * N - s[k1]
        for mode, k in TENSILE_MODES.items():
            df[mode] = s[k] + tstr
        out[tag] = df

    # Safe window: collapse from Shmin azimuth (all shear modes >= 0),
    # fracture from SHmax azimuth (all tensile modes >= 0). Combined = intersection.
    bo = out["Shmin azimuth (breakout)"]
    fr = out["SHmax azimuth (fracture)"]
    shear_ok = (bo[list(SHEAR_MODES)] >= 0).all(axis=1).to_numpy()
    tensile_ok = (fr[list(TENSILE_MODES)] >= 0).all(axis=1).to_numpy()
    stable = shear_ok & tensile_ok
    mw_safe = mw[stable]
    window = (float(mw_safe.min()), float(mw_safe.max())) if mw_safe.size else (np.nan, np.nan)
    return {"by_azimuth": out, "mw": mw, "window": window,
            "shear_ok": shear_ok, "tensile_ok": tensile_ok}


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

    st.markdown("**Rock strength & elasticity**")
    st.slider("Static Poisson's ratio", 0.05, 0.45, key="pr", step=0.01)
    st.slider("Biot coefficient α", 0.5, 1.0, key="biot", step=0.05)
    st.slider("Unconfined compressive strength · C₀ (psi)", 0.0, 20000.0, key="ucs", step=250.0)
    st.slider("Friction angle · φ (deg)", 0.0, 55.0, key="fang", step=1.0)
    st.slider("Tensile strength · T₀ (psi)", 0.0, 3000.0, key="tstr", step=50.0)

    st.markdown("**Sampling**")
    st.slider("Azimuthal samples", 91, 721, key="n", step=90,
              help="Number of points around the wellbore circumference.")

    if st.session_state.shmin > st.session_state.shmax:
        st.warning("Shmin > SHmax — check your inputs (Shmin should be the minimum).")

s = st.session_state
df = compute_wall(s.shmin, s.shmax, s.svert, s.pp, s.mud, s.az, s.dev, s.bh_az, s.pr, s.n)
theta = df["theta"].to_numpy()

i_max = int(df["sigma_tt"].idxmax())
i_min = int(df["sigma_tt"].idxmin())
breakout_az = float(df["theta"].iloc[i_max])
tensile_az = float(df["theta"].iloc[i_min])
max_hoop = float(df["sigma_tt"].max())
min_hoop = float(df["sigma_tt"].min())
max_s1 = float(df["sigma_1"].max())


# ---------------------------------------------------------------------------
# Header + unit toggle
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="nw-hero">
      <h1>🛢️ Near-Wellbore Stresses</h1>
      <p>Kirsch borehole-wall stresses &amp; Bratton (1999) failure analysis · powered by
      <b>GeomechPy</b> · SHmax az {s.az:.0f}° · deviation {s.dev:.0f}° ·
      Pw {s.mud:.0f} psi · TVD {s.tvd:.0f} ft</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uc, tc = st.columns([2, 3])
with uc:
    units = st.radio("Pressure units", ["psi", "ppg (mud weight)"],
                     horizontal=True, key="units",
                     help="ppg = psi / (0.052 × TVD). Compare stresses directly against mud weight.")
PPG = units.startswith("ppg")
USUFFIX = "ppg" if PPG else "psi"


def to_unit(v):
    return v / (PSI_PER_PPG_FT * s.tvd) if PPG else v


def fmt(v):
    return f"{to_unit(v):.2f} ppg" if PPG else f"{v:,.0f} psi"


Pw_u = to_unit(s.mud)
Pp_u = to_unit(s.pp)
MW_now = s.mud / (PSI_PER_PPG_FT * s.tvd)  # current mud weight in ppg

with tc:
    st.markdown(
        f"<div style='padding-top:1.9rem;color:{MUTED};font-size:.85rem'>"
        f"Showing stresses in <b>{USUFFIX}</b> · current mud weight Pw = {MW_now:.2f} ppg"
        "</div>",
        unsafe_allow_html=True,
    )


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="nw-kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# Bratton safe window (auto mud-density range from Pp/Sv equivalents)
mw_pp = s.pp / (PSI_PER_PPG_FT * s.tvd)
mw_sv = s.svert / (PSI_PER_PPG_FT * s.tvd)
MW_LO = max(1.0, round(mw_pp - 2.0, 1))
MW_HI = round(mw_sv + 4.0, 1)
bratton = bratton_analysis(s.shmax, s.shmin, s.svert, s.pp, s.tvd, s.pr, s.biot,
                           s.ucs, s.fang, s.tstr, MW_LO, MW_HI)
win_lo, win_hi = bratton["window"]

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Max hoop σθθ", fmt(max_hoop), "breakout-prone")
kpi(k2, "Breakout azimuth", f"{breakout_az:.0f}°", "max σθθ around wall")
if np.isfinite(win_lo):
    kpi(k3, "Min mud weight", f"{win_lo:.2f} ppg", "below → breakout / collapse")
    kpi(k4, "Max mud weight", f"{win_hi:.2f} ppg", "above → tensile fracture")
else:
    kpi(k3, "Safe mud window", "none", "no stable MW in range")
    kpi(k4, "Max principal σ₁", fmt(max_s1), "peak wall stress")

st.write("")


# ---------------------------------------------------------------------------
# Figure helpers (no fixed width -> always fits its container)
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


CHART_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_azim, tab_traj, tab_fail, tab_data = st.tabs(
    ["📈 Azimuthal profile", "🎯 Trajectory compare",
     "🧱 Failure modes (Bratton 1999)", "🗂️ Data"]
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
    fig.add_trace(go.Scatter(
        x=theta, y=df["tortuosity"], mode="lines", name="tortuosity θ (deg)",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
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
        (θ = 0° at the Top-of-Hole, sweeping clockwise). Each curve is a Kirsch wall-stress
        component in <b>{USUFFIX}</b>; σ₁ ≥ σ₂ ≥ σ₃ are the principal wall stresses.
        <ul style="margin:.4rem 0 0 .1rem">
          <li><b>σθθ (hoop)</b> peaks (red dotted line, ≈ {breakout_az:.0f}°) where the wall is most
              compressed — <b>breakouts</b> nucleate here; it is lowest (blue dotted line ≈ {tensile_az:.0f}°)
              where <b>tensile fractures</b> initiate.</li>
          <li><b>σzz</b> is axial, <b>σrr</b> = Pw − Pp is the radial support from the mud, and
              <b>σtz</b> is the tangential-axial shear (non-zero only in deviated wells).</li>
          <li>Use the psi ⇄ ppg toggle to read the curves against your mud weight.</li>
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
        d = compute_wall(s.shmin, s.shmax, s.svert, s.pp, s.mud, s.az, float(dv), s.bh_az, s.pr, s.n)
        fig.add_trace(go.Scatter(
            x=d["theta"], y=to_unit(d[mcol]), mode="lines", name=f"deviation {dv}°",
            line=dict(color=palette[j % len(palette)], width=2),
            hovertemplate="%{x:.0f}°: %{y:,.1f} " + USUFFIX + "<extra>dev " + str(dv) + "°</extra>",
        ))
    _base_layout(fig, height=520, title=f"{metric} vs azimuth by deviation ({USUFFIX})")
    fig.update_xaxes(title_text="θ — azimuth from Top-of-Hole (deg)", range=[0, 360], dtick=45)
    fig.update_yaxes(title_text=f"Stress ({USUFFIX})")
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    st.markdown(
        f"""
        <div class="nw-note">
        <b>How to read this.</b> Each curve is <b>{metric}</b> around the wall for a different
        <b>borehole deviation</b> (0° = vertical, 90° = horizontal), far-field stresses fixed —
        so it isolates the effect of the <b>well trajectory</b>. A higher peak means a stronger
        stress concentration (more breakout-prone); a flatter curve is a more stable trajectory.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Failure modes (Bratton 1999) -----------------------------------------
with tab_fail:
    st.caption(
        "Bratton et al. (SPWLA, 1999) analysis for a **vertical well**: effective wall stresses "
        "(total − αPp) and the Delta-Stability of each failure mode vs mud density. "
        "Positive Delta-Stability = stable; negative = failed."
    )
    if s.dev > 0:
        st.info("This analysis assumes a vertical well; the deviation set in the sidebar is ignored here.")

    azkey = st.radio("Evaluate at",
                     ["Shmin azimuth (breakout)", "SHmax azimuth (fracture)"],
                     horizontal=True)
    bdf = bratton["by_azimuth"][azkey]
    mw = bdf["MW"].to_numpy()

    def _mw_shade(fig):
        if np.isfinite(win_lo):
            fig.add_vrect(x0=win_lo, x1=win_hi, fillcolor="rgba(16,185,129,0.10)",
                          line_width=0, layer="below")
        fig.add_vline(x=MW_now, line=dict(color="#0b6e4f", width=2, dash="dash"))

    # --- Plot 1: principal (effective) wall stresses vs mud density ---
    fig1 = go.Figure()
    for col, name, color in [("sigma_r", "Radial σr", "#6b7280"),
                             ("sigma_t", "Tangential σt (hoop)", "#2563eb"),
                             ("sigma_a", "Axial σa", "#059669")]:
        fig1.add_trace(go.Scatter(
            x=mw, y=to_unit(bdf[col]), mode="lines", name=name,
            line=dict(color=color, width=2.5),
            hovertemplate="MW %{x:.2f} ppg → %{y:,.1f} " + USUFFIX + "<extra>" + name + "</extra>"))
    _mw_shade(fig1)
    _base_layout(fig1, height=430, title=f"Principal wall stresses vs mud density — {azkey}")
    fig1.update_xaxes(title_text="Mud density (ppg)")
    fig1.update_yaxes(title_text=f"Effective stress ({USUFFIX})")
    fig1.update_layout(showlegend=True)
    st.plotly_chart(fig1, use_container_width=True, config=CHART_CONFIG)

    # --- Plot 2: stability plot (Delta-Stability per failure mode) ---
    show_modes = st.multiselect(
        "Failure modes",
        list(SHEAR_MODES) + list(TENSILE_MODES),
        default=["Swbo — wide breakout", "Snbo — narrow breakout",
                 "Tver — vertical", "Tcyl — cylindrical"],
    )
    fig2 = go.Figure()
    for mode in show_modes:
        dash = "dashdot" if mode in TENSILE_MODES else None
        fig2.add_trace(go.Scatter(
            x=mw, y=to_unit(bdf[mode]), mode="lines", name=mode,
            line=dict(color=MODE_COLORS.get(mode, "#334155"), width=2, dash=dash),
            hovertemplate="MW %{x:.2f} ppg → Δ %{y:,.1f} " + USUFFIX + "<extra>" + mode + "</extra>"))
    fig2.add_hline(y=0, line=dict(color="#111827", width=1.2))
    _mw_shade(fig2)
    _base_layout(fig2, height=470, title=f"Stability plot — Delta-Stability vs mud density ({azkey})")
    fig2.update_xaxes(title_text="Mud density (ppg)")
    fig2.update_yaxes(title_text=f"Delta-Stability ({USUFFIX})")
    fig2.update_layout(showlegend=True)
    st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)

    if np.isfinite(win_lo):
        status = "inside" if win_lo <= MW_now <= win_hi else "OUTSIDE"
        st.markdown(
            f"""
            <div class="nw-note">
            <b>Safe mud-weight window: {win_lo:.2f} – {win_hi:.2f} ppg.</b>
            The green band is where every shear mode is stable at the Shmin azimuth <i>and</i>
            every tensile mode is stable at the SHmax azimuth. Your current mud weight
            (<b>{MW_now:.2f} ppg</b>, dashed green line) is <b>{status}</b> the window.
            <ul style="margin:.4rem 0 0 .1rem">
              <li><b>Below the window</b> the hoop stress is too high → shear <b>breakouts</b>
                  (Swbo/Ssko…). Raise mud weight.</li>
              <li><b>Above the window</b> the hoop stress goes tensile → <b>fracturing</b>
                  (Tver) and losses. Lower mud weight.</li>
              <li>Delta-Stability follows Bratton Eq. 5 (shear: C₀ + σ₃·tan²(45+φ/2) − σ₁) and
                  Eq. 6 (tensile: σ + T₀).</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "No stable mud weight exists in the scanned range — the window is closed. "
            "Check the stresses, strength (C₀, φ, T₀) and pore pressure."
        )

# ---- Data -----------------------------------------------------------------
with tab_data:
    st.caption(f"Computed borehole-wall stresses at every azimuth (values in {USUFFIX}).")
    out = pd.DataFrame({"theta": df["theta"], "tortuosity": df["tortuosity"]})
    for label, (col, _c) in COMPONENT_STYLE.items():
        out[col] = to_unit(df[col])
    out = out[["theta", "sigma_rr", "sigma_tt", "sigma_zz", "sigma_tz",
               "sigma_1", "sigma_2", "sigma_3", "tortuosity"]]
    show = out.copy()
    show.columns = ["θ (deg)", f"σrr ({USUFFIX})", f"σθθ ({USUFFIX})", f"σzz ({USUFFIX})",
                    f"σtz ({USUFFIX})", f"σ₁ ({USUFFIX})", f"σ₂ ({USUFFIX})", f"σ₃ ({USUFFIX})",
                    "tortuosity (deg)"]
    st.dataframe(show.style.format("{:.2f}"), use_container_width=True, height=360)
    st.download_button(
        f"⬇️ Wall stresses CSV ({USUFFIX})", out.to_csv(index=False).encode("utf-8"),
        file_name=f"near_wellbore_stresses_{USUFFIX}.csv", mime="text/csv",
    )
    st.divider()
    st.caption(f"Bratton failure analysis at the Shmin azimuth (Delta-Stability in {USUFFIX}).")
    bexp = bratton["by_azimuth"]["Shmin azimuth (breakout)"].copy()
    for c in bexp.columns:
        if c != "MW":
            bexp[c] = to_unit(bexp[c])
    st.download_button(
        f"⬇️ Bratton stability CSV ({USUFFIX})", bexp.to_csv(index=False).encode("utf-8"),
        file_name=f"bratton_stability_{USUFFIX}.csv", mime="text/csv",
    )

st.markdown(
    f"<p style='color:{MUTED};font-size:.8rem;margin-top:1.4rem'>"
    "Kirsch wall stresses · <code>geomechpy.near_wellbore_stresses</code>. "
    "Failure analysis after Bratton, Bornemann, Li, Plumb, Rasmus &amp; Krabbe, "
    "<i>Logging-While-Drilling Images for Geomechanical, Geological and Petrophysical "
    "Interpretations</i>, SPWLA 1999. ppg equivalent = psi ÷ (0.052 × TVD).</p>",
    unsafe_allow_html=True,
)
