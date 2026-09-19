"""Quick 1D MEM-WBS Calculator - standalone single-file Streamlit app.

A one-page Mechanical Earth Model (MEM) builder on top of the GeomechPy
library: dynamic elastic properties -> static conversion -> rock strength
(selectable methods) -> overburden & pore pressure -> horizontal stresses ->
vertical-well wellbore stability, with QC flagging, tornado sensitivity
analysis and interactive Plotly displays. Supports Oilfield and Metric
unit systems for both input and display.

This file bundles the calculation engine and the Streamlit front end into a
single module so it can be hosted directly:

    cd example/Project
    streamlit run MEM_Calculator.py

Canonical internal units (everything is converted to these before
calculation, regardless of the selected input unit system):
    DTCO/DTSM us/ft . RHOB g/cc . velocities m/s . moduli GPa .
    strength MPa . friction angle deg
"""
from __future__ import annotations

import io
import math
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

from geomechpy.elastic_properties import ElasticPropertiesConverter
from geomechpy.overburden_stress import OverburdenStressCalculation
from geomechpy.pore_pressure import PorePressureCalculation
from geomechpy.rock_strength import RockStrengthPropertiesConverter
from geomechpy.static_elastic_properties import StaticElasticPropertiesConverter
from geomechpy.stress_calculations import HorizontalStressesCalculation
from geomechpy.wellbore_stability import WellboreStabilityCalculation


# ===========================================================================
# CALCULATION ENGINE
# ===========================================================================

# ---------------------------------------------------------------------------
# Constants & unit conversion
# ---------------------------------------------------------------------------

FT_TO_M_US = 304800.0          # us/ft slowness -> m/s velocity: v = 304800 / dt
M_PER_FT = 0.3048              # us/ft = us/m * 0.3048 ; ft/s = m/s / 0.3048
PA_TO_GPA = 1.0e-9             # Pascal -> GigaPascal
PA_TO_PSI = 1.0 / 6894.757293  # Pascal -> psi
PA_TO_MPSI = PA_TO_PSI * 1e-6  # Pascal -> Mega-psi
PSI_TO_MPA = 6894.757293e-6    # psi -> MPa
MPA_TO_PSI = 1.0 / PSI_TO_MPA  # MPa -> psi
GPA_TO_PSI = PA_TO_PSI / PA_TO_GPA  # GPa -> psi (~145037.7)
GPA_TO_MPSI = PA_TO_MPSI / PA_TO_GPA  # GPa -> Mpsi (~0.145)
GCC_TO_KGM3 = 1000.0           # g/cc -> kg/m3
M_TO_FT = 1.0 / M_PER_FT       # metres -> feet
PSI_FT_PER_GCC = 0.4335        # hydrostatic gradient of 1 g/cc fluid in psi/ft
PPG_PER_GCC = 8.3454           # mud weight: 1 g/cc = 8.3454 ppg

# Common well-log null sentinels replaced with NaN on load.
NULL_SENTINELS = [-999.0, -999.25, -9999.0, -9999.25, -99999.0, 9999.0]

# Curves the app can map. POROSITY is optional unless the Morales method is used.
REQUIRED_CURVES = ["DEPTH", "GR", "RHOB", "DTCO", "DTSM"]
OPTIONAL_CURVES = ["POROSITY"]
ALL_CURVES = REQUIRED_CURVES + OPTIONAL_CURVES

# ---------------------------------------------------------------------------
# Unit systems
# ---------------------------------------------------------------------------

OILFIELD = "Oilfield Units"
METRIC = "Metric Units"
UNIT_SYSTEMS = [OILFIELD, METRIC]

# Expected INPUT units per system (shown in the UI and used to convert to
# canonical units before calculation).
INPUT_UNITS = {
    OILFIELD: {"DEPTH": "m", "GR": "gAPI", "RHOB": "g/cc", "DTCO": "µs/ft", "DTSM": "µs/ft", "POROSITY": "frac"},
    METRIC: {"DEPTH": "m", "GR": "gAPI", "RHOB": "kg/m³", "DTCO": "µs/m", "DTSM": "µs/m", "POROSITY": "frac"},
}

# Display spec: canonical column -> (display name, (oilfield unit, factor),
# (metric unit, factor)). Factor converts FROM the canonical value TO the
# displayed value. Order here defines display column order.
DISPLAY_SPEC: dict[str, tuple[str, tuple[str, float], tuple[str, float]]] = {
    "DEPTH": ("MD", ("", 1.0), ("", 1.0)),  # passed through in the input depth unit
    "LITHO_CODE": ("LITHO", ("code", 1.0), ("code", 1.0)),  # NEW: mechanical stratigraphy flag
    "GR": ("GR", ("gAPI", 1.0), ("gAPI", 1.0)),
    "RHOB": ("RHOB", ("g/cc", 1.0), ("kg/m³", GCC_TO_KGM3)),
    "DTCO": ("DTCO", ("µs/ft", 1.0), ("µs/m", 1.0 / M_PER_FT)),
    "DTSM": ("DTSM", ("µs/ft", 1.0), ("µs/m", 1.0 / M_PER_FT)),
    "POROSITY": ("POROSITY", ("frac", 1.0), ("frac", 1.0)),
    "VP_MS": ("VP", ("ft/s", 1.0 / M_PER_FT), ("m/s", 1.0)),
    "VS_MS": ("VS", ("ft/s", 1.0 / M_PER_FT), ("m/s", 1.0)),
    "VPVS": ("VP/VS", ("-", 1.0), ("-", 1.0)),
    "YME_DYN_GPA": ("YME_DYN", ("psi", GPA_TO_PSI), ("psi", GPA_TO_PSI)),  # forced psi
    "PR_DYN": ("PR_DYN", ("-", 1.0), ("-", 1.0)),
    "K_DYN_GPA": ("K_DYN", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "G_DYN_GPA": ("G_DYN", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "LAME_DYN_GPA": ("LAME_DYN", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "M_DYN_GPA": ("M_DYN", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "YME_STA_GPA": ("YME_STA", ("psi", GPA_TO_PSI), ("psi", GPA_TO_PSI)),  # forced psi
    "PR_STA": ("PR_STA", ("-", 1.0), ("-", 1.0)),
    "UCS_MPA": ("UCS", ("psi", MPA_TO_PSI), ("psi", MPA_TO_PSI)),  # forced psi
    "TSTR_MPA": ("TSTR", ("psi", MPA_TO_PSI), ("psi", MPA_TO_PSI)),  # forced psi
    "FANG_DEG": ("FANG", ("deg", 1.0), ("deg", 1.0)),
    # --- NEW: stress profile & wellbore stability columns ---
    "SV_MPA": ("SV", ("psi", MPA_TO_PSI), ("psi", MPA_TO_PSI)),  # forced psi
    "PP_MPA": ("PP", ("psi", MPA_TO_PSI), ("psi", MPA_TO_PSI)),  # forced psi
    "SHMIN_MPA": ("SHMIN", ("psi", MPA_TO_PSI), ("psi", MPA_TO_PSI)),  # forced psi
    "SHMAX_MPA": ("SHMAX", ("psi", MPA_TO_PSI), ("psi", MPA_TO_PSI)),  # forced psi
    "Q_FACTOR": ("Q_FACTOR", ("-", 1.0), ("-", 1.0)),
    "SH_RATIO": ("SHMAX/SHMIN", ("-", 1.0), ("-", 1.0)),
    "PW_BREAKOUT_MPA": ("PW_BREAKOUT", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "PW_BREAKDOWN_MPA": ("PW_BREAKDOWN", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "LOSS_P_MPA": ("LOSS_GRAD_P", ("psi", MPA_TO_PSI), ("MPa", 1.0)),  # NEW
    "MW_LOSS_GCC": ("LOSS_GRADIENT", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),  # NEW
    "MW_PP_GCC": ("EMW_PP", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),
    "MW_BREAKOUT_GCC": ("MW_BREAKOUT", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),
    "MW_BREAKDOWN_GCC": ("MW_BREAKDOWN", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),
    "MW_SV_GCC": ("EMW_SV", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),
}


def _spec(canonical: str, unit_system: str) -> tuple[str, str, float]:
    """(display base name, unit string, factor from canonical) for a column."""
    name, oilfield, metric = DISPLAY_SPEC[canonical]
    unit, factor = oilfield if unit_system == OILFIELD else metric
    return name, unit, factor


def display_name(canonical: str, unit_system: str) -> str:
    """Display column header, e.g. 'YME_DYN [Mpsi]'."""
    name, unit, _ = _spec(canonical, unit_system)
    return f"{name} [{unit}]" if unit not in ("", "-") else name


def display_unit(canonical: str, unit_system: str) -> str:
    """Unit string for the selected system, e.g. 'GPa' or 'Mpsi'."""
    return _spec(canonical, unit_system)[1]


def display_results(df: pd.DataFrame, unit_system: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert a canonical results frame into the selected unit system.

    Returns:
        disp: converted DataFrame with unit-labelled column names.
        names: mapping canonical column -> display column name (for plots
               and for renaming QC flag columns).
    """
    disp = pd.DataFrame(index=df.index)
    names: dict[str, str] = {}
    for canonical in DISPLAY_SPEC:
        if canonical not in df.columns:
            continue
        _, _, factor = _spec(canonical, unit_system)
        label = display_name(canonical, unit_system)
        disp[label] = pd.to_numeric(df[canonical], errors="coerce") * factor
        names[canonical] = label
    return disp, names


def normalize_input_units(df: pd.DataFrame, unit_system: str) -> pd.DataFrame:
    """Convert mapped input columns (canonical names) into canonical units.

    Oilfield input is already canonical. Metric input: DT µs/m -> µs/ft,
    RHOB kg/m³ -> g/cc.
    """
    out = df.copy()
    if unit_system == METRIC:
        for col in ("DTCO", "DTSM"):
            if col in out.columns:
                out[col] = out[col] * M_PER_FT
        if "RHOB" in out.columns:
            out["RHOB"] = out["RHOB"] / GCC_TO_KGM3
    return out


def check_unit_sanity(data: pd.DataFrame, column_map: dict[str, str], unit_system: str) -> list[str]:
    """Heuristic warnings when the data magnitudes contradict the selected units."""
    warnings: list[str] = []

    def _median(curve: str) -> float:
        src = column_map.get(curve)
        if not src or src not in data.columns:
            return float("nan")
        return float(pd.to_numeric(data[src], errors="coerce").median())

    dt = _median("DTCO")
    rhob = _median("RHOB")
    if unit_system == OILFIELD:
        if np.isfinite(dt) and dt > 250:
            warnings.append(
                f"Median DTCO is {dt:.0f} — that looks like µs/m, but Oilfield Units expects µs/ft. "
                "Consider switching to Metric Units."
            )
        if np.isfinite(rhob) and rhob > 100:
            warnings.append(
                f"Median RHOB is {rhob:.0f} — that looks like kg/m³, but Oilfield Units expects g/cc. "
                "Consider switching to Metric Units."
            )
    else:
        if np.isfinite(dt) and dt < 130:
            warnings.append(
                f"Median DTCO is {dt:.0f} — that looks like µs/ft, but Metric Units expects µs/m. "
                "Consider switching to Oilfield Units."
            )
        if np.isfinite(rhob) and rhob < 10:
            warnings.append(
                f"Median RHOB is {rhob:.2f} — that looks like g/cc, but Metric Units expects kg/m³. "
                "Consider switching to Oilfield Units."
            )
    return warnings


# QC validation ranges in CANONICAL units: column -> (min, max, unit).
QC_RANGES = {
    "GR": (0.0, 250.0, "gAPI"),
    "RHOB": (1.5, 3.2, "g/cc"),
    "DTCO": (40.0, 240.0, "us/ft"),
    "DTSM": (60.0, 450.0, "us/ft"),
    "POROSITY": (0.0, 0.5, "frac"),
    "VPVS": (1.4, 2.4, "ratio"),
    "PR_DYN": (0.0, 0.5, "unitless"),
    "YME_DYN_GPA": (0.5, 130.0, "GPa"),
    "PR_STA": (0.0, 0.5, "unitless"),
    "YME_STA_GPA": (0.1, 110.0, "GPa"),
    "UCS_MPA": (1.0, 400.0, "MPa"),
    "TSTR_MPA": (0.1, 60.0, "MPa"),
    "FANG_DEG": (10.0, 55.0, "deg"),
    # NEW: stress profile sanity ranges
    "SV_MPA": (1.0, 300.0, "MPa"),
    "PP_MPA": (0.5, 200.0, "MPa"),
    "SHMIN_MPA": (0.5, 300.0, "MPa"),
    "SHMAX_MPA": (0.5, 400.0, "MPa"),
}

# NEW: selectable rock strength methods. geomechpy.rock_strength currently
# ships one correlation per property (Plumb UCS, Lal FANG); a constant-value
# option is provided as a generic app-level fallback for calibration work.
UCS_METHODS = {
    "Plumb (1994) — from static YME [geomechpy]": "plumb",
    "McNally (1987) — from DTCO, sandstone [app]": "mcnally",
    "Constant value": "constant",
}
FANG_METHODS = {
    "Lal (1999) — from DTCO, shale [geomechpy]": "lal",
    "GR custom linear (sand→shale, GRmin/GRmax) [app]": "gr_linear",
    "Constant value": "constant",
}

# Static Young's modulus correlations exposed in the UI.
# input_unit tells us which unit the geomechpy function expects for dynamic YME.
STATIC_YME_METHODS = {
    "Bradford (power law, North Sea sandstone)": {"key": "bradford", "input_unit": "Mpsi"},
    "Najibi (power law, Iranian carbonates)": {"key": "najibi", "input_unit": "Mpsi"},
    "Fuller (power law, sandstone/shale)": {"key": "fuller", "input_unit": "GPa"},
    "Morales (porosity-dependent, sandstone)": {"key": "morales", "input_unit": "Mpsi"},
    "Custom power law (y = a*x^b)": {"key": "custom_power", "input_unit": "Mpsi"},
    "Custom linear law (y = a*x + b)": {"key": "custom_linear", "input_unit": "Mpsi"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _drop_unit_rows(df: pd.DataFrame, max_rows: int = 3) -> tuple[pd.DataFrame, int]:
    """Drop leading rows that hold unit strings instead of data.

    A leading row is treated as a unit row when it is non-numeric in at least
    half of the columns whose remaining values are mostly numeric
    (e.g. a 'M | GAPI | G/CC | US/F' line under the header).
    """
    dropped = 0
    while len(df) > 1 and dropped < max_rows:
        first = df.iloc[0]
        body = df.iloc[1:]
        checkable = 0
        non_numeric = 0
        for col in df.columns:
            body_numeric_share = pd.to_numeric(body[col], errors="coerce").notna().mean()
            if body_numeric_share >= 0.6:
                checkable += 1
                first_val = pd.to_numeric(pd.Series([first[col]]), errors="coerce").iloc[0]
                if pd.isna(first_val):
                    non_numeric += 1
        if checkable and non_numeric / checkable >= 0.5:
            df = df.iloc[1:].reset_index(drop=True)
            dropped += 1
        else:
            break
    return df, dropped


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Robust cleanup of a freshly parsed log table.

    - drops leading unit rows,
    - coerces mostly-numeric columns to numeric (bad cells -> NaN),
    - replaces well-log null sentinels (-999.25, -9999, ...) and ±inf with NaN.

    Returns the cleaned frame and a list of informational messages.
    """
    messages: list[str] = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df, dropped = _drop_unit_rows(df)
    if dropped:
        messages.append(f"Skipped {dropped} leading unit/header row(s) that contained no data.")

    n_nulls = 0
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.5:  # mostly numeric -> treat as a log curve
            sentinel_mask = numeric.isin(NULL_SENTINELS) | ~np.isfinite(numeric.fillna(0.0))
            n_nulls += int(sentinel_mask.sum())
            numeric[sentinel_mask] = np.nan
            df[col] = numeric
    if n_nulls:
        messages.append(f"Replaced {n_nulls} null sentinel value(s) (e.g. -999.25 / -9999) with NaN.")

    df = df.dropna(how="all").reset_index(drop=True)
    return df, messages


def load_data(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    """Read an uploaded CSV or Excel file and clean it up.

    Returns (dataframe, informational messages).
    Raises ValueError with a user-friendly message on failure.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".las"):
            import lasio  # optional dependency; only needed for LAS input
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
            text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
            las = lasio.read(io.StringIO(text))
            # las.df() is indexed by the depth curve (e.g. DEPT); expose it as a column.
            df = las.df().reset_index()
        elif name.endswith((".csv", ".txt")):
            df = pd.read_csv(uploaded_file, skip_blank_lines=True)
            # Single-column result usually means a non-comma delimiter: re-sniff.
            if df.shape[1] == 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=None, engine="python", skip_blank_lines=True)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload a .las, .csv, .xls or .xlsx file.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse file '{uploaded_file.name}': {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file contains no rows.")
    if df.shape[1] < 2:
        raise ValueError("The uploaded file needs at least a depth column and one log curve.")

    df, messages = clean_dataframe(df)
    if df.empty:
        raise ValueError("No data rows remained after cleaning the file.")
    return df, messages


def guess_column(curve: str, columns: list[str]) -> int:
    """Best-effort index of the column matching a curve mnemonic (for selectbox defaults).

    Returns 0 ('-- not mapped --') when nothing matches.
    """
    aliases = {
        "DEPTH": ["depth", "dept", "md", "tvd"],
        "GR": ["gr", "gamma", "gapi", "cgr", "sgr"],
        "RHOB": ["rhob", "den", "density", "rho", "zden"],
        "DTCO": ["dtco", "dtc", "dt_p", "dtp", "ac", "dt4p", "dtcomp"],
        "DTSM": ["dtsm", "dts", "dt_s", "dt4s", "dtshear"],
        "POROSITY": ["porosity", "phit", "phie", "nphi", "por", "phi"],
    }
    lowered = [c.lower().strip() for c in columns]
    for alias in aliases[curve]:
        for i, col in enumerate(lowered):
            if col == alias or col.startswith(alias):
                return i + 1  # +1 for the '-- not mapped --' placeholder at index 0
    return 0


def missing_required_curves(columns: list[str]) -> list[str]:
    """Required curves that could not be auto-detected in the given columns."""
    return [c for c in REQUIRED_CURVES if guess_column(c, columns) == 0]


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

def generate_sample_data(n_points: int = 201, seed: int = 42, unit_system: str = OILFIELD) -> pd.DataFrame:
    """Generate a synthetic sand/shale well-log interval (2500-3000 m MD).

    Values are geologically plausible so all downstream calculations produce
    sensible magnitudes. Output columns: MD, GR, RHOB, DTCO, DTSM, POROSITY,
    expressed in the requested unit system.
    """
    rng = np.random.default_rng(seed)
    depth = np.linspace(2500.0, 3000.0, n_points)

    # Smooth sand/shale alternation driver (0 = clean sand, 1 = shale)
    vsh = 0.5 + 0.35 * np.sin(depth / 18.0) + 0.15 * np.sin(depth / 61.0)
    vsh = np.clip(vsh + rng.normal(0, 0.05, n_points), 0.02, 0.98)

    compaction = (depth - 2500.0) / 500.0  # 0 -> 1 over the interval

    gr = 25.0 + 110.0 * vsh + rng.normal(0, 4.0, n_points)
    rhob = 2.30 + 0.25 * compaction + 0.12 * vsh + rng.normal(0, 0.02, n_points)  # g/cc
    dtco = 95.0 - 25.0 * compaction + 18.0 * vsh + rng.normal(0, 1.5, n_points)   # us/ft
    dtsm = dtco * (1.65 + 0.25 * vsh) + rng.normal(0, 3.0, n_points)              # us/ft
    porosity = np.clip(0.28 - 0.12 * compaction - 0.08 * vsh + rng.normal(0, 0.01, n_points), 0.03, 0.35)

    if unit_system == METRIC:
        dtco = dtco / M_PER_FT       # us/ft -> us/m
        dtsm = dtsm / M_PER_FT
        rhob = rhob * GCC_TO_KGM3    # g/cc -> kg/m3

    return pd.DataFrame(
        {
            "MD": np.round(depth, 2),
            "GR": np.round(gr, 2),
            "RHOB": np.round(rhob, 3),
            "DTCO": np.round(dtco, 2),
            "DTSM": np.round(dtsm, 2),
            "POROSITY": np.round(porosity, 3),
        }
    )


def sample_csv_bytes(unit_system: str = OILFIELD) -> bytes:
    """Example CSV for the 'Download Example File' button."""
    buffer = io.StringIO()
    generate_sample_data(unit_system=unit_system).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Dynamic elastic properties (geomechpy.elastic_properties)
# ---------------------------------------------------------------------------

def compute_dynamic_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dynamic elastic properties from DTCO/DTSM/RHOB (canonical units).

    Unit handling:
        DTCO, DTSM : us/ft  -> Vp, Vs in m/s   (v = 304800 / dt)
        RHOB       : g/cc   -> kg/m3           (x1000)
        geomechpy returns moduli in Pa -> reported in GPa (+ Mpsi for YME)

    Invalid rows (non-positive slowness/density, NaN) yield NaN outputs
    instead of raising.
    """
    out = df.copy()

    dtco = pd.to_numeric(out["DTCO"], errors="coerce").to_numpy(dtype=float)
    dtsm = pd.to_numeric(out["DTSM"], errors="coerce").to_numpy(dtype=float)
    rhob = pd.to_numeric(out["RHOB"], errors="coerce").to_numpy(dtype=float)

    n = len(out)
    vp = np.full(n, np.nan)
    vs = np.full(n, np.nan)
    cols = {
        k: np.full(n, np.nan)
        for k in ["YME_DYN_GPA", "PR_DYN", "K_DYN_GPA", "G_DYN_GPA", "LAME_DYN_GPA", "M_DYN_GPA"]
    }

    valid = (dtco > 0) & (dtsm > 0) & (rhob > 0)
    vp[valid] = FT_TO_M_US / dtco[valid]
    vs[valid] = FT_TO_M_US / dtsm[valid]

    for i in np.flatnonzero(valid):
        try:
            # geomechpy expects slowness in us/ft and density in kg/m3, returns Pa
            props = ElasticPropertiesConverter.convert_dynamic_elastic_properties_from_slowness(
                p_wave_slowness=float(dtco[i]),
                s_wave_slowness=float(dtsm[i]),
                density=float(rhob[i]) * GCC_TO_KGM3,
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        cols["YME_DYN_GPA"][i] = props.youngs_modulus * PA_TO_GPA
        cols["PR_DYN"][i] = props.poissons_ratio
        cols["K_DYN_GPA"][i] = props.bulk_modulus * PA_TO_GPA
        cols["G_DYN_GPA"][i] = props.shear_modulus * PA_TO_GPA
        cols["LAME_DYN_GPA"][i] = props.lame_parameter * PA_TO_GPA
        cols["M_DYN_GPA"][i] = props.p_wave_modulus * PA_TO_GPA

    out["VP_MS"] = vp
    out["VS_MS"] = vs
    with np.errstate(divide="ignore", invalid="ignore"):
        out["VPVS"] = np.where(vs > 0, vp / vs, np.nan)
    for k, v in cols.items():
        out[k] = v
    out["YME_DYN_MPSI"] = out["YME_DYN_GPA"] * GPA_TO_MPSI
    return out


# ---------------------------------------------------------------------------
# Static elastic properties (geomechpy.static_elastic_properties)
# ---------------------------------------------------------------------------

def compute_static_properties(
    df: pd.DataFrame,
    method_label: str,
    calibration_multiplier: float = 1.0,
    pr_multiplier: float = 1.0,
    custom_a: float = 0.5,
    custom_b: float = 1.0,
) -> pd.DataFrame:
    """Convert dynamic to static elastic properties.

    Args:
        df: DataFrame that already contains YME_DYN_GPA / YME_DYN_MPSI / PR_DYN
            (and POROSITY when using the Morales method).
        method_label: key of STATIC_YME_METHODS selected in the UI.
        calibration_multiplier: global calibration factor (0.5-2.0 slider)
            applied to the correlation output.
        pr_multiplier: static/dynamic Poisson's ratio multiplier.
        custom_a, custom_b: coefficients for the custom power/linear laws
            (a = multiplier/slope, b = exponent/intercept).
    """
    method = STATIC_YME_METHODS[method_label]
    out = df.copy()
    conv = StaticElasticPropertiesConverter

    n = len(out)
    yme_sta_native = np.full(n, np.nan)  # in the method's native unit (Mpsi or GPa)
    yme_dyn_mpsi = out["YME_DYN_MPSI"].to_numpy(dtype=float)
    yme_dyn_gpa = out["YME_DYN_GPA"].to_numpy(dtype=float)

    if method["key"] == "morales":
        if "POROSITY" not in out.columns or out["POROSITY"].isna().all():
            raise ValueError("The Morales correlation requires a mapped POROSITY column.")
        por = pd.to_numeric(out["POROSITY"], errors="coerce").to_numpy(dtype=float)

    for i in range(n):
        yd_mpsi, yd_gpa = yme_dyn_mpsi[i], yme_dyn_gpa[i]
        if not np.isfinite(yd_mpsi) or yd_mpsi <= 0:
            continue
        try:
            if method["key"] == "bradford":
                val = conv.dyn2sta_yme_bradord(yme_dyn=yd_mpsi)
            elif method["key"] == "najibi":
                val = conv.dyn2sta_yme_najib(yme_dyn=yd_mpsi)
            elif method["key"] == "fuller":
                val = conv.dyn2sta_yme_fuller(yme_dyn=yd_gpa)  # Fuller works in GPa
            elif method["key"] == "morales":
                if not np.isfinite(por[i]):
                    continue
                val = conv.dyn2sta_yme_morales(yme_dyn=yd_mpsi, porosity=float(por[i]))
                if val == -9999:  # library's low-porosity exclusion flag
                    val = np.nan
            elif method["key"] == "custom_power":
                val = conv.convert_dyn2sta_yme_custom_power_law(
                    yme_dyn=yd_mpsi, multiplier=custom_a, exponent=custom_b
                )
            else:  # custom_linear
                val = conv.dyn2sta_yme_custom_linear_law(
                    yme_dyn=yd_mpsi, slope=custom_a, intercept=custom_b
                )
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        yme_sta_native[i] = val

    # Apply the user's calibration multiplier, then normalise units.
    yme_sta_native = yme_sta_native * calibration_multiplier
    if method["input_unit"] == "GPa":
        out["YME_STA_GPA"] = yme_sta_native
        out["YME_STA_MPSI"] = yme_sta_native * GPA_TO_MPSI
    else:
        out["YME_STA_MPSI"] = yme_sta_native
        out["YME_STA_GPA"] = yme_sta_native / GPA_TO_MPSI

    # Static Poisson's ratio via geomechpy constant-multiplier law.
    pr_dyn = out["PR_DYN"].to_numpy(dtype=float)
    out["PR_STA"] = [
        conv.dyn2sta_poissons_ratio(pr_dyn=float(v), multiplier=pr_multiplier)
        if np.isfinite(v)
        else np.nan
        for v in pr_dyn
    ]

    # Negative static moduli (possible with a custom linear intercept) are unphysical.
    out.loc[out["YME_STA_GPA"] <= 0, ["YME_STA_GPA", "YME_STA_MPSI"]] = np.nan
    return out


# ---------------------------------------------------------------------------
# Rock strength (geomechpy.rock_strength)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# App-provided correlations (not in this repo's geomechpy build)
# ---------------------------------------------------------------------------

def _mcnally_ucs_sandstone_psi(dtco_usft: float) -> float:
    """McNally (1987) sandstone UCS from compressional slowness.

    UCS[MPa] = 1200 * exp(-0.036 * DTCO[us/ft]); returned here in psi.
    """
    ucs_mpa = 1200.0 * math.exp(-0.036 * float(dtco_usft))
    return ucs_mpa * MPA_TO_PSI


def _gr_linear_fang(gr: float, gr_min: float, gr_max: float) -> float:
    """Linear friction angle from GR: 45 deg at gr_min (clean sand) ->
    15 deg at gr_max (pure shale), clamped to [15, 45]."""
    if gr_max == gr_min:
        raise ValueError("GR max must differ from GR min.")
    frac = (float(gr) - gr_min) / (gr_max - gr_min)
    fang = 45.0 + (15.0 - 45.0) * frac
    return float(min(45.0, max(15.0, fang)))


def compute_rock_strength(
    df: pd.DataFrame,
    tstr_multiplier: float = 0.15,
    ucs_method: str = "plumb",
    fang_method: str = "lal",
    ucs_constant_mpa: float = 50.0,
    fang_constant_deg: float = 30.0,
    fang_gr_min: float = 15.0,
    fang_gr_max: float = 120.0,
    ucs_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Compute UCS, tensile strength and friction angle with selectable methods.

    ucs_method:  'plumb' (geomechpy Plumb 1994 from static YME),
                 'mcnally' (geomechpy McNally 1987 from DTCO, sandstone) or 'constant'.
    fang_method: 'lal' (geomechpy Lal 1999 from DTCO),
                 'gr_linear' (geomechpy custom GR sand->shale linear,
                 45 deg at GRmin -> 15 deg at GRmax) or 'constant'.
    TSTR is always tstr_multiplier x UCS (geomechpy constant-multiplier law).

    Unit handling:
        UCS  : static YME passed in MPa, geomechpy returns psi -> also report MPa.
               Note: the geomechpy docstring says "Mpsi" but its coefficient
               (0.2103 psi per unit input, i.e. UCS[MPa] ~ 1.45 x E[GPa]) only
               yields physical UCS magnitudes with MPa input, so MPa is used.
        TSTR : psi -> also MPa
        FANG : from DTCO in us/ft, returned in degrees
    """
    out = df.copy()
    conv = RockStrengthPropertiesConverter

    n = len(out)
    ucs_psi = np.full(n, np.nan)
    tstr_psi = np.full(n, np.nan)
    fang = np.full(n, np.nan)

    yme_sta = out["YME_STA_GPA"].to_numpy(dtype=float) * 1000.0  # GPa -> MPa
    dtco = pd.to_numeric(out["DTCO"], errors="coerce").to_numpy(dtype=float)
    gr = (
        pd.to_numeric(out["GR"], errors="coerce").to_numpy(dtype=float)
        if "GR" in out.columns
        else np.full(n, np.nan)
    )

    for i in range(n):
        # --- UCS ---
        if ucs_method == "constant":
            ucs_psi[i] = ucs_constant_mpa * MPA_TO_PSI
        elif ucs_method == "mcnally":
            if np.isfinite(dtco[i]) and dtco[i] > 0:
                ucs_psi[i] = _mcnally_ucs_sandstone_psi(float(dtco[i]))
        elif np.isfinite(yme_sta[i]) and yme_sta[i] > 0:  # plumb
            ucs_psi[i] = conv.convert_yme_sta_to_ucs_plumb(yme_sta=float(yme_sta[i]))
        # --- UCS calibration multiplier (like the static YME multiplier) ---
        if np.isfinite(ucs_psi[i]):
            ucs_psi[i] = ucs_psi[i] * ucs_multiplier
        # --- TSTR (always derived from the calibrated UCS) ---
        if np.isfinite(ucs_psi[i]):
            tstr_psi[i] = conv.convert_ucs_to_tstr(ucs=float(ucs_psi[i]), multiplier=tstr_multiplier)
        # --- FANG ---
        if fang_method == "constant":
            fang[i] = fang_constant_deg
        elif fang_method == "gr_linear":
            if np.isfinite(gr[i]):
                try:
                    fang[i] = _gr_linear_fang(float(gr[i]), fang_gr_min, fang_gr_max)
                except (ValueError, ZeroDivisionError):  # gr_max == gr_min guard
                    pass
        elif np.isfinite(dtco[i]) and dtco[i] > 0:  # lal
            try:
                fang[i] = conv.convert_friction_angle_lal(dtco=float(dtco[i]))
            except (ValueError, ZeroDivisionError):  # asin domain / dt=0 guards
                pass

    out["UCS_PSI"] = ucs_psi
    out["UCS_MPA"] = ucs_psi * PSI_TO_MPA
    out["TSTR_PSI"] = tstr_psi
    out["TSTR_MPA"] = tstr_psi * PSI_TO_MPA
    out["FANG_DEG"] = fang
    return out


# ---------------------------------------------------------------------------
# QC validation
# ---------------------------------------------------------------------------

def run_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate every known column against QC_RANGES (canonical units).

    Returns:
        qc_summary: one row per checked curve with counts and % in range.
        flags: per-sample flag DataFrame ('OK' / 'LOW' / 'HIGH' / 'MISSING')
               aligned with df, for color-coded display.
    """
    summary_rows = []
    flags = pd.DataFrame(index=df.index)

    for col, (lo, hi, unit) in QC_RANGES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        col_flags = pd.Series("OK", index=df.index)
        col_flags[values.isna()] = "MISSING"
        col_flags[values < lo] = "LOW"
        col_flags[values > hi] = "HIGH"
        flags[col] = col_flags

        n_total = len(values)
        n_missing = int(values.isna().sum())
        n_low = int((values < lo).sum())
        n_high = int((values > hi).sum())
        n_ok = n_total - n_missing - n_low - n_high
        summary_rows.append(
            {
                "Curve": col,
                "Unit": unit,
                "Valid range": f"{lo:g} - {hi:g}",
                "Samples": n_total,
                "OK": n_ok,
                "Below range": n_low,
                "Above range": n_high,
                "Missing": n_missing,
                "% in range": round(100.0 * n_ok / n_total, 1) if n_total else 0.0,
            }
        )

    return pd.DataFrame(summary_rows), flags


def qc_status(qc_summary: pd.DataFrame) -> str:
    """Overall traffic-light status: PASS / WARNING / FAIL."""
    if qc_summary.empty:
        return "FAIL"
    worst = qc_summary["% in range"].min()
    if worst >= 95.0:
        return "PASS"
    if worst >= 70.0:
        return "WARNING"
    return "FAIL"


# ---------------------------------------------------------------------------
# NEW: Overburden, pore pressure, horizontal stress & wellbore stability
# (geomechpy.overburden_stress / pore_pressure / stress_calculations /
#  wellbore_stability)
# ---------------------------------------------------------------------------

WELL_SETTINGS = ["Onshore", "Offshore"]
SHMAX_METHODS = {
    "Poroelastic (Thiercelin & Plumb, 1994) [geomechpy]": "poroelastic",
    "Shmin × anisotropy multiplier [geomechpy]": "multiplier",
}
OVB_GRADIENT_SOURCES = {
    "Constant lithostatic gradient": "constant",
    "Derived from mean RHOB log": "density",
}
DEPTH_UNITS = ["m", "ft"]


def default_stress_params() -> dict:
    """Default parameter set for compute_stress_profile (all user-adjustable)."""
    return {
        "setting": "Onshore",            # 'Onshore' | 'Offshore'
        "depth_unit": "m",               # unit of the DEPTH column ('m' | 'ft')
        "air_gap": 0.0,                  # in depth_unit
        "water_depth": 0.0,              # in depth_unit (offshore only)
        "sea_gradient_psift": 0.47,      # sea water pressure gradient [psi/ft]
        "ovb_source": "constant",        # 'constant' | 'density'
        "ovb_gradient_psift": 1.05,      # lithostatic gradient [psi/ft]
        "pp_gradient_psift": 0.47,       # formation pore pressure gradient [psi/ft]
        "shmax_method": "poroelastic",   # 'poroelastic' | 'multiplier'
        "shmax_multiplier": 1.1,         # used when shmax_method == 'multiplier'
        "biot": 1.0,                     # Biot coefficient (constant law)
        "ex": 0.0001,                    # tectonic strain term EX (poroelastic)
        "ey": 0.009,                     # tectonic strain term EY (poroelastic)
    }


def compute_stress_profile(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """NEW: full stress profile + vertical-well stability from geomechpy.

    Steps (all library calls, per depth sample; MD is assumed ~ TVD, i.e. a
    vertical well, which matches the analytical wellbore stability solution):
      1. SV  : OverburdenStressCalculation onshore/offshore gradient method.
               The lithostatic gradient is either a constant or derived from
               the mean RHOB log (mean g/cc x 0.4335 psi/ft).
      2. PP  : PorePressureCalculation onshore/offshore gradient method.
      3. SHMIN/SHMAX : poroelastic equation (static PR + static YME [Mpsi],
               Biot, tectonic strains EX/EY), or SHMAX = SHMIN x multiplier.
               q-factor and SHMAX/SHMIN ratio from the library helpers.
      4. Wellbore stability (vertical well): Mohr-Coulomb breakout pressure
               and breakdown (fracture initiation) pressure, converted to
               equivalent mud weights with the true vertical depth.

    Canonical outputs: pressures/stresses in MPa, mud weights in g/cc.
    Rows with missing prerequisites yield NaN instead of raising.
    """
    p = {**default_stress_params(), **(params or {})}
    out = df.copy()
    n = len(out)

    depth = pd.to_numeric(out["DEPTH"], errors="coerce").to_numpy(dtype=float)
    tvd_ft = depth * M_TO_FT if p["depth_unit"] == "m" else depth.copy()
    to_ft = M_TO_FT if p["depth_unit"] == "m" else 1.0
    air_gap_ft = float(p["air_gap"]) * to_ft
    water_depth_ft = float(p["water_depth"]) * to_ft

    # 1. Overburden gradient
    ovb_gradient = float(p["ovb_gradient_psift"])
    if p["ovb_source"] == "density" and "RHOB" in out.columns:
        mean_rhob = float(pd.to_numeric(out["RHOB"], errors="coerce").mean())
        if np.isfinite(mean_rhob) and mean_rhob > 0:
            ovb_gradient = mean_rhob * PSI_FT_PER_GCC

    sv_psi = np.full(n, np.nan)
    pp_psi = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(tvd_ft[i]) or tvd_ft[i] < 0:
            continue
        if p["setting"] == "Offshore":
            sv_psi[i] = OverburdenStressCalculation.calculate_overburden_stress_offshore(
                tvd=float(tvd_ft[i]),
                lithostatic_gradient=ovb_gradient,
                air_gap=air_gap_ft,
                water_depth=water_depth_ft,
                sea_water_pressure_gradient=float(p["sea_gradient_psift"]),
            )
            pp_psi[i] = PorePressureCalculation.calculate_pore_pressure_offshore(
                tvd=float(tvd_ft[i]),
                formation_pore_pressure_gradient=float(p["pp_gradient_psift"]),
                air_gap=air_gap_ft,
                water_depth=water_depth_ft,
                sea_water_pressure_gradient=float(p["sea_gradient_psift"]),
            )
        else:
            sv_psi[i] = OverburdenStressCalculation.calculate_overburden_stress_onshore(
                tvd=float(tvd_ft[i]),
                lithostatic_gradient=ovb_gradient,
                air_gap=air_gap_ft,
            )
            pp_psi[i] = PorePressureCalculation.calculate_pore_pressure_onshore(
                tvd=float(tvd_ft[i]),
                formation_pore_pressure_gradient=float(p["pp_gradient_psift"]),
                air_gap=air_gap_ft,
            )

    # 2. Horizontal stresses (poroelastic needs static PR + static YME in Mpsi)
    pr_sta = pd.to_numeric(out.get("PR_STA"), errors="coerce").to_numpy(dtype=float) if "PR_STA" in out.columns else np.full(n, np.nan)
    yme_sta_mpsi = pd.to_numeric(out.get("YME_STA_MPSI"), errors="coerce").to_numpy(dtype=float) if "YME_STA_MPSI" in out.columns else np.full(n, np.nan)

    shmin_psi = np.full(n, np.nan)
    shmax_psi = np.full(n, np.nan)
    q_factor = np.full(n, np.nan)
    sh_ratio = np.full(n, np.nan)
    for i in range(n):
        if not (np.isfinite(sv_psi[i]) and np.isfinite(pp_psi[i]) and np.isfinite(pr_sta[i]) and np.isfinite(yme_sta_mpsi[i])):
            continue
        if not (0.0 < pr_sta[i] < 0.5):
            continue
        try:
            hs = HorizontalStressesCalculation.calculate_poroelastic_horizontal_stresses(
                overburden_stress=float(sv_psi[i]),
                pore_pressure=float(pp_psi[i]),
                poisson_ratio=float(pr_sta[i]),
                youngs_modulus=float(yme_sta_mpsi[i]),
                biot_coefficient=float(p["biot"]),
                EX=float(p["ex"]),
                EY=float(p["ey"]),
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        shmin_psi[i] = hs.shmin
        if p["shmax_method"] == "multiplier":
            shmax_psi[i] = HorizontalStressesCalculation.calculate_shmax_multiplier(
                shmin=float(hs.shmin), shmax_multiplier=float(p["shmax_multiplier"])
            )
        else:
            shmax_psi[i] = hs.shmax
        try:
            q_factor[i] = HorizontalStressesCalculation.calculate_stress_regime_q_factor(
                sigv=float(sv_psi[i]), shmax=float(shmax_psi[i]), shmin=float(shmin_psi[i])
            )
            sh_ratio[i] = HorizontalStressesCalculation.calculate_horizontal_stress_ratio(
                shmax=float(shmax_psi[i]), shmin=float(shmin_psi[i])
            )
        except (ValueError, ZeroDivisionError):
            pass

    # 3. Wellbore stability (vertical well, analytical)
    ucs_psi = pd.to_numeric(out.get("UCS_PSI"), errors="coerce").to_numpy(dtype=float) if "UCS_PSI" in out.columns else np.full(n, np.nan)
    tstr_psi = pd.to_numeric(out.get("TSTR_PSI"), errors="coerce").to_numpy(dtype=float) if "TSTR_PSI" in out.columns else np.full(n, np.nan)
    fang = pd.to_numeric(out.get("FANG_DEG"), errors="coerce").to_numpy(dtype=float) if "FANG_DEG" in out.columns else np.full(n, np.nan)

    pw_bo_psi = np.full(n, np.nan)   # breakout (shear failure) limit
    pw_bd_psi = np.full(n, np.nan)   # breakdown (fracture initiation) limit
    for i in range(n):
        if not (np.isfinite(shmin_psi[i]) and np.isfinite(shmax_psi[i]) and np.isfinite(pp_psi[i])):
            continue
        if np.isfinite(tstr_psi[i]):
            try:
                pw_bd_psi[i] = WellboreStabilityCalculation.calculate_breakdown_calculation_vertical_well_analytical(
                    shmax=float(shmax_psi[i]), shmin=float(shmin_psi[i]),
                    pprs=float(pp_psi[i]), tstr=float(tstr_psi[i]),
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        if np.isfinite(sv_psi[i]) and np.isfinite(ucs_psi[i]) and np.isfinite(fang[i]) and np.isfinite(pr_sta[i]):
            try:
                pw_bo_psi[i] = WellboreStabilityCalculation.calculate_breakout_calculation_vertical_well_mohr_coulomb_analytical(
                    shmax=float(shmax_psi[i]), shmin=float(shmin_psi[i]),
                    pprs=float(pp_psi[i]), overburden_stress=float(sv_psi[i]),
                    ucs=float(ucs_psi[i]), fang=float(fang[i]), pr_sta=float(pr_sta[i]),
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                pass

    # NEW: loss gradient = minimum principal stress among Sv, SHmax, Shmin.
    # NaN if any of the three is missing (a partial minimum would mislead).
    loss_psi = np.minimum.reduce([sv_psi, shmax_psi, shmin_psi])

    # 4. Equivalent mud weights (g/cc canonical): EMW = P / (0.4335 * TVD_ft)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = PSI_FT_PER_GCC * tvd_ft
        mw = lambda pressure_psi: np.where(denom > 0, pressure_psi / denom, np.nan)  # noqa: E731
        out["MW_PP_GCC"] = mw(pp_psi)
        out["MW_SV_GCC"] = mw(sv_psi)
        out["MW_BREAKOUT_GCC"] = mw(pw_bo_psi)
        out["MW_BREAKDOWN_GCC"] = mw(pw_bd_psi)
        out["MW_LOSS_GCC"] = mw(loss_psi)

    out["SV_MPA"] = sv_psi * PSI_TO_MPA
    out["PP_MPA"] = pp_psi * PSI_TO_MPA
    out["SHMIN_MPA"] = shmin_psi * PSI_TO_MPA
    out["SHMAX_MPA"] = shmax_psi * PSI_TO_MPA
    out["Q_FACTOR"] = q_factor
    out["SH_RATIO"] = sh_ratio
    out["PW_BREAKOUT_MPA"] = pw_bo_psi * PSI_TO_MPA
    out["PW_BREAKDOWN_MPA"] = pw_bd_psi * PSI_TO_MPA
    out["LOSS_P_MPA"] = loss_psi * PSI_TO_MPA
    return out


# ---------------------------------------------------------------------------
# NEW: Stress barrier analysis & perforation zone screening
# ---------------------------------------------------------------------------

PERF_QUALITIES = ["Good", "Moderate", "Poor"]


def analyze_stress_barriers(
    results: pd.DataFrame,
    contrast_threshold_mpa: float = 1.0,
    trend_window: int = 25,
    search_window: int = 20,
    min_zone_samples: int = 3,
) -> dict:
    """NEW: simple stress barrier analysis on the computed Shmin profile.

    Rationale: hydraulic fractures initiate where Shmin is locally LOW and
    stay contained when intervals of locally HIGH Shmin (stress barriers)
    exist above and below. The analysis therefore:

      1. Removes the depth trend from Shmin (centered rolling median over
         trend_window samples) and works with the residual stress contrast.
      2. Classifies each sample: contrast >= +threshold  -> 'Barrier',
         contrast <= -threshold -> 'Target', otherwise 'Neutral'.
      3. Rates each sample as a perforation candidate:
           Good     : Target with a Barrier within search_window samples
                      both above AND below (contained low-stress interval).
           Moderate : Target with a Barrier on one side only.
           Poor     : everything else (barriers themselves, neutral rock,
                      uncontained targets).
      4. Groups contiguous Good/Moderate samples into recommended
         perforation intervals (at least min_zone_samples thick).

    All stress values are canonical MPa; the caller converts for display.

    Returns dict with:
        detail   : per-depth DataFrame (DEPTH, SHMIN_MPA, TREND_MPA,
                   CONTRAST_MPA, CLASS, PERF_QUALITY).
        zones    : recommended perforation intervals (Top, Base, Thickness,
                   Samples, Mean Shmin, Mean contrast, Quality).
        barriers : barrier intervals (Top, Base, Thickness, Mean contrast).
    """
    if "SHMIN_MPA" not in results.columns:
        raise ValueError("Stress barrier analysis needs the stress profile — enable stress computation and re-run.")

    depth = pd.to_numeric(results["DEPTH"], errors="coerce")
    shmin = pd.to_numeric(results["SHMIN_MPA"], errors="coerce")
    valid = shmin.notna() & depth.notna()
    if int(valid.sum()) < max(10, min_zone_samples):
        raise ValueError("Not enough valid Shmin samples for a barrier analysis — check the QC report.")

    trend = shmin.rolling(int(trend_window), center=True, min_periods=1).median()
    contrast = shmin - trend

    n = len(results)
    cls = np.full(n, "N/A", dtype=object)
    t = float(contrast_threshold_mpa)
    c = contrast.to_numpy(dtype=float)
    ok = valid.to_numpy()
    cls[ok & (c >= t)] = "Barrier"
    cls[ok & (c <= -t)] = "Target"
    cls[ok & (np.abs(c) < t)] = "Neutral"

    # Barrier presence within search_window samples above/below each sample.
    is_barrier = (cls == "Barrier").astype(int)
    above = np.zeros(n, dtype=bool)
    below = np.zeros(n, dtype=bool)
    w = int(search_window)
    for i in range(n):
        above[i] = is_barrier[max(0, i - w):i].any()
        below[i] = is_barrier[i + 1:i + 1 + w].any()

    quality = np.full(n, "Poor", dtype=object)
    target = cls == "Target"
    quality[target & above & below] = "Good"
    quality[target & (above ^ below)] = "Moderate"
    quality[~ok] = "N/A"

    detail = pd.DataFrame(
        {
            "DEPTH": depth,
            "SHMIN_MPA": shmin,
            "TREND_MPA": trend,
            "CONTRAST_MPA": contrast,
            "CLASS": cls,
            "PERF_QUALITY": quality,
        }
    )

    def _intervals(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
        """Contiguous [start, end] index runs where mask is True."""
        runs, start = [], None
        for i in range(n):
            if mask[i] and start is None:
                start = i
            elif not mask[i] and start is not None:
                if i - start >= min_len:
                    runs.append((start, i - 1))
                start = None
        if start is not None and n - start >= min_len:
            runs.append((start, n - 1))
        return runs

    zone_rows = []
    for grade in ("Good", "Moderate"):
        for s, e in _intervals(quality == grade, int(min_zone_samples)):
            zone_rows.append(
                {
                    "Top": float(depth.iloc[s]),
                    "Base": float(depth.iloc[e]),
                    "Thickness": float(depth.iloc[e] - depth.iloc[s]),
                    "Samples": e - s + 1,
                    "Mean Shmin (MPa)": float(shmin.iloc[s:e + 1].mean()),
                    "Mean contrast (MPa)": float(contrast.iloc[s:e + 1].mean()),
                    "Quality": grade,
                }
            )
    zones = pd.DataFrame(zone_rows).sort_values("Top").reset_index(drop=True) if zone_rows else pd.DataFrame(
        columns=["Top", "Base", "Thickness", "Samples", "Mean Shmin (MPa)", "Mean contrast (MPa)", "Quality"]
    )

    barrier_rows = [
        {
            "Top": float(depth.iloc[s]),
            "Base": float(depth.iloc[e]),
            "Thickness": float(depth.iloc[e] - depth.iloc[s]),
            "Mean contrast (MPa)": float(contrast.iloc[s:e + 1].mean()),
        }
        for s, e in _intervals(cls == "Barrier", 2)
    ]
    barriers = pd.DataFrame(barrier_rows) if barrier_rows else pd.DataFrame(
        columns=["Top", "Base", "Thickness", "Mean contrast (MPa)"]
    )

    return {"detail": detail, "zones": zones, "barriers": barriers}


# ---------------------------------------------------------------------------
# NEW: Mechanical stratigraphy (GR-based lithology flag)
# ---------------------------------------------------------------------------

# Simplified lithology: sandstone (0) vs shale (1) split by a single GR cutoff.
LITHO_NAME_BY_CODE = {0: "Sandstone", 1: "Shale"}
LITHO_CODE_BY_NAME = {v: k for k, v in LITHO_NAME_BY_CODE.items()}
# Display colours per code (+ -1 / NaN = undefined).
LITHO_COLORS = {0: "#f4d03f", 1: "#7f8c8d", -1: "#ecf0f1"}

DEFAULT_GR_CUTOFF = 75.0  # gAPI: GR < cutoff -> sandstone (0), GR >= cutoff -> shale (1)


def compute_mechanical_stratigraphy(df: pd.DataFrame, gr_cutoff: float = DEFAULT_GR_CUTOFF) -> pd.DataFrame:
    """NEW: add a LITHO_CODE column classifying each sample from GR by one cutoff.

    GR < gr_cutoff  -> sandstone (code 0)
    GR >= gr_cutoff -> shale     (code 1)
    Missing GR      -> NaN
    """
    out = df.copy()
    gr = (
        pd.to_numeric(out["GR"], errors="coerce").to_numpy(dtype=float)
        if "GR" in out.columns
        else np.full(len(out), np.nan)
    )
    code = np.where(np.isfinite(gr), np.where(gr < float(gr_cutoff), 0.0, 1.0), np.nan)
    out["LITHO_CODE"] = code
    return out


def lithology_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-lithology sample counts and fraction, for the stratigraphy summary."""
    if "LITHO_CODE" not in df.columns:
        return pd.DataFrame(columns=["Lithology", "Code", "Samples", "Fraction %"])
    codes = pd.to_numeric(df["LITHO_CODE"], errors="coerce")
    total = int(codes.notna().sum())
    rows = []
    for code, name in LITHO_NAME_BY_CODE.items():
        cnt = int((codes == code).sum())
        rows.append({"Lithology": name, "Code": code, "Samples": cnt,
                     "Fraction %": round(100.0 * cnt / total, 1) if total else 0.0})
    undef = int(codes.isna().sum())
    if undef:
        rows.append({"Lithology": "Undefined", "Code": -1, "Samples": undef, "Fraction %": 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full pipeline + export
# ---------------------------------------------------------------------------

def run_full_workflow(
    data: pd.DataFrame,
    column_map: dict[str, str],
    method_label: str,
    calibration_multiplier: float,
    pr_multiplier: float,
    tstr_multiplier: float,
    custom_a: float = 0.5,
    custom_b: float = 1.0,
    unit_system: str = OILFIELD,
    ucs_method: str = "plumb",
    fang_method: str = "lal",
    ucs_constant_mpa: float = 50.0,
    fang_constant_deg: float = 30.0,
    fang_gr_min: float = 15.0,
    fang_gr_max: float = 120.0,
    ucs_multiplier: float = 1.0,
    gr_cutoff: float | None = None,
    stress_params: dict | None = None,
) -> pd.DataFrame:
    """Rename mapped columns to standard mnemonics, convert the input to
    canonical units and run the geomechpy modules: dynamic -> static ->
    rock strength (selectable methods) -> optionally the full stress
    profile + wellbore stability when stress_params is provided."""
    missing = [c for c in REQUIRED_CURVES if not column_map.get(c)]
    if missing:
        raise ValueError(f"Missing required column mapping(s): {', '.join(missing)}")

    rename = {src: curve for curve, src in column_map.items() if src}
    df = data[[c for c in rename]].rename(columns=rename).copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin(NULL_SENTINELS), col] = np.nan  # safety net
    df = df.sort_values("DEPTH").reset_index(drop=True)

    df = normalize_input_units(df, unit_system)

    df = compute_dynamic_properties(df)
    df = compute_static_properties(
        df,
        method_label=method_label,
        calibration_multiplier=calibration_multiplier,
        pr_multiplier=pr_multiplier,
        custom_a=custom_a,
        custom_b=custom_b,
    )
    df = compute_rock_strength(
        df,
        tstr_multiplier=tstr_multiplier,
        ucs_method=ucs_method,
        fang_method=fang_method,
        ucs_constant_mpa=ucs_constant_mpa,
        fang_constant_deg=fang_constant_deg,
        fang_gr_min=fang_gr_min,
        fang_gr_max=fang_gr_max,
        ucs_multiplier=ucs_multiplier,
    )
    if gr_cutoff is not None:  # None => user disabled lithology flagging
        df = compute_mechanical_stratigraphy(df, gr_cutoff)
    if stress_params is not None:
        df = compute_stress_profile(df, stress_params)
    return df


def results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize results for the Streamlit download button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, float_format="%.4f")
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Sensitivity analysis (Tornado plot)
# ---------------------------------------------------------------------------

# Canonical result columns selectable as tornado targets (display order).
TORNADO_TARGETS = [
    "YME_STA_GPA",
    "UCS_MPA",
    "TSTR_MPA",
    "PR_STA",
    "YME_DYN_GPA",
    "PR_DYN",
    "FANG_DEG",
]
# NEW: additional targets available when the stress profile is computed.
TORNADO_STRESS_TARGETS = [
    "SHMIN_MPA",
    "SHMAX_MPA",
    "MW_BREAKOUT_GCC",
    "MW_BREAKDOWN_GCC",
]

# Input curves perturbed one at a time, plus the static YME calibration multiplier.
TORNADO_INPUT_CURVES = ["GR", "RHOB", "DTCO", "DTSM", "POROSITY"]
STATIC_MULT_PARAM = "Static YME multiplier"
TORNADO_PARAMS = TORNADO_INPUT_CURVES + [STATIC_MULT_PARAM]


def run_tornado_analysis(
    data: pd.DataFrame,
    column_map: dict[str, str],
    target_output: str,
    variation_pct: float = 10.0,
    *,
    method_label: str,
    calibration_multiplier: float = 1.0,
    pr_multiplier: float = 1.0,
    tstr_multiplier: float = 0.15,
    custom_a: float = 0.5,
    custom_b: float = 1.0,
    unit_system: str = OILFIELD,
    ucs_method: str = "plumb",
    fang_method: str = "lal",
    ucs_constant_mpa: float = 50.0,
    fang_constant_deg: float = 30.0,
    fang_gr_min: float = 15.0,
    fang_gr_max: float = 120.0,
    ucs_multiplier: float = 1.0,
    gr_cutoff: float | None = None,
    stress_params: dict | None = None,
) -> tuple[pd.DataFrame, float, list[str]]:
    """One-at-a-time sensitivity of a target output to the main inputs.

    The current data is the base case. Each parameter in TORNADO_PARAMS is
    varied by ±variation_pct while everything else is held fixed, and the
    full workflow is recomputed. The compared statistic is the depth-averaged
    (NaN-ignoring mean) value of the target column, in canonical units.

    Returns:
        tornado: DataFrame with one row per varied parameter:
                 Parameter, low, high (target means at -/+ variation),
                 pct_low, pct_high (% change vs base), swing (|high - low|),
                 sorted by swing descending.
        base_value: base-case target mean (canonical units).
        skipped: parameters that could not be varied (unmapped column or
                 failed recomputation).
    """
    if target_output not in DISPLAY_SPEC:
        raise ValueError(f"Unknown target output: {target_output}")

    workflow_kwargs = dict(
        column_map=column_map,
        method_label=method_label,
        pr_multiplier=pr_multiplier,
        tstr_multiplier=tstr_multiplier,
        custom_a=custom_a,
        custom_b=custom_b,
        unit_system=unit_system,
        ucs_method=ucs_method,
        fang_method=fang_method,
        ucs_constant_mpa=ucs_constant_mpa,
        fang_constant_deg=fang_constant_deg,
        fang_gr_min=fang_gr_min,
        fang_gr_max=fang_gr_max,
        ucs_multiplier=ucs_multiplier,
        gr_cutoff=gr_cutoff,
        stress_params=stress_params,
    )

    def _target_mean(frame: pd.DataFrame, cal_multiplier: float) -> float:
        res = run_full_workflow(frame, calibration_multiplier=cal_multiplier, **workflow_kwargs)
        if target_output not in res.columns:
            raise ValueError(f"Target '{target_output}' was not produced by the workflow.")
        return float(np.nanmean(pd.to_numeric(res[target_output], errors="coerce")))

    base_value = _target_mean(data, calibration_multiplier)
    if not np.isfinite(base_value):
        raise ValueError(
            "The base case produced no valid values for the selected target — "
            "check the column mapping and QC report."
        )

    frac = float(variation_pct) / 100.0
    rows: list[dict] = []
    skipped: list[str] = []

    for param in TORNADO_PARAMS:
        try:
            if param == STATIC_MULT_PARAM:
                low = _target_mean(data, calibration_multiplier * (1.0 - frac))
                high = _target_mean(data, calibration_multiplier * (1.0 + frac))
            else:
                src = column_map.get(param)
                if not src or src not in data.columns:
                    skipped.append(param)
                    continue
                values = pd.to_numeric(data[src], errors="coerce")
                lo_frame = data.copy()
                lo_frame[src] = values * (1.0 - frac)
                hi_frame = data.copy()
                hi_frame[src] = values * (1.0 + frac)
                low = _target_mean(lo_frame, calibration_multiplier)
                high = _target_mean(hi_frame, calibration_multiplier)
        except (ValueError, ZeroDivisionError, OverflowError):
            skipped.append(param)
            continue

        rows.append(
            {
                "Parameter": param,
                "low": low,
                "high": high,
                "pct_low": 100.0 * (low - base_value) / base_value if base_value else np.nan,
                "pct_high": 100.0 * (high - base_value) / base_value if base_value else np.nan,
            }
        )

    tornado = pd.DataFrame(rows)
    if tornado.empty:
        raise ValueError("No input parameters could be varied — check the column mapping.")
    tornado["swing"] = (tornado["high"] - tornado["low"]).abs()
    tornado = tornado.sort_values("swing", ascending=False).reset_index(drop=True)
    return tornado, base_value, skipped


def _build_tornado_figure(tornado: pd.DataFrame, base_value: float, target_output: str,
                          variation_pct: float, unit_system: str, subtitle: str = "") -> tuple[go.Figure, pd.DataFrame, float]:
    """Shared horizontal-bar tornado figure + display table from a tornado frame
    (Parameter, low, high, pct_low, pct_high, swing) in canonical units."""
    _, _, factor = _spec(target_output, unit_system)
    target_label = display_name(target_output, unit_system)
    base_display = base_value * factor

    t = tornado.copy()
    t["low"] = t["low"] * factor
    t["high"] = t["high"] * factor
    t["swing"] = t["swing"] * factor
    t["delta_low"] = t["low"] - base_display
    t["delta_high"] = t["high"] - base_display

    # Plotly draws category bars bottom-up: ascending swing puts the biggest on top.
    plot = t.sort_values("swing", ascending=True)
    pct = f"{variation_pct:g}"

    fig = go.Figure()
    fig.add_bar(
        y=plot["Parameter"], x=plot["delta_low"], base=base_display, orientation="h",
        name=f"-{pct}%", marker_color="#d95f02",
        customdata=np.stack([plot["low"], plot["pct_low"]], axis=-1),
        hovertemplate=("%{y} -" + pct + "%<br>" + target_label
                       + ": %{customdata[0]:.3f} (%{customdata[1]:+.2f}% vs base)<extra></extra>"),
    )
    fig.add_bar(
        y=plot["Parameter"], x=plot["delta_high"], base=base_display, orientation="h",
        name=f"+{pct}%", marker_color="#1f77b4",
        customdata=np.stack([plot["high"], plot["pct_high"]], axis=-1),
        hovertemplate=("%{y} +" + pct + "%<br>" + target_label
                       + ": %{customdata[0]:.3f} (%{customdata[1]:+.2f}% vs base)<extra></extra>"),
    )
    fig.add_vline(x=base_display, line_dash="dash", line_color="gray",
                  annotation_text=f"base = {base_display:.3f}", annotation_position="top")
    title = f"Tornado plot — sensitivity of {target_label} to ±{pct}% variation"
    if subtitle:
        title += f"<br><sub>{subtitle}</sub>"
    fig.update_layout(
        barmode="overlay", title=title,
        xaxis_title=f"Depth-averaged {target_label}", yaxis_title="Varied parameter",
        height=max(360, 90 * len(plot) + 150),
        legend=dict(orientation="h", yanchor="bottom", y=1.04),
        margin=dict(t=120, b=40),
    )
    table = t[["Parameter", "low", "high", "pct_low", "pct_high", "swing"]].rename(
        columns={
            "low": f"Target @ -{pct}%",
            "high": f"Target @ +{pct}%",
            "pct_low": "Δ% @ low",
            "pct_high": "Δ% @ high",
            "swing": f"Swing [{display_unit(target_output, unit_system)}]",
        }
    )
    return fig, table, base_display


def generate_tornado_plot(
    df: pd.DataFrame,
    column_map: dict[str, str],
    target_output: str,
    variation_pct: float = 10.0,
    **settings,
) -> tuple[go.Figure, pd.DataFrame, float, list[str]]:
    """Input-log tornado: perturb raw logs (+ static YME multiplier) and
    recompute the whole workflow. Returns (fig, table, base_display, skipped)."""
    unit_system = settings.get("unit_system", OILFIELD)
    tornado, base_value, skipped = run_tornado_analysis(
        df, column_map, target_output, variation_pct, **settings
    )
    fig, table, base_display = _build_tornado_figure(
        tornado, base_value, target_output, variation_pct, unit_system,
        subtitle="Varying input logs (full workflow recomputed)",
    )
    return fig, table, base_display, skipped


# ---------------------------------------------------------------------------
# NEW: Parameter-mode tornado — vary the governing-equation inputs of a
# derived output (e.g. Shmin from Sv, Pp, Poisson's ratio, YME, Biot, EX, EY)
# instead of the raw logs.
# ---------------------------------------------------------------------------

# Derived outputs whose governing parameters can be perturbed directly.
TORNADO_PARAM_TARGETS = ["SHMIN_MPA", "SHMAX_MPA", "MW_BREAKOUT_GCC", "MW_BREAKDOWN_GCC"]

_PARAM_LABEL = {
    "sv": "Sv (overburden)", "pp": "Pp (pore pressure)",
    "pr": "Poisson's ratio (static)", "yme": "YME (static)",
    "biot": "Biot coefficient", "ex": "Tectonic strain EX", "ey": "Tectonic strain EY",
    "shmin": "Shmin", "shmax": "SHmax", "ucs": "UCS", "fang": "Friction angle", "tstr": "TSTR",
}

# Governing parameters that drive each target (order = display order).
TORNADO_TARGET_PARAMS = {
    "SHMIN_MPA": ["sv", "pp", "pr", "yme", "biot", "ex", "ey"],
    "SHMAX_MPA": ["sv", "pp", "pr", "yme", "biot", "ex", "ey"],
    "MW_BREAKOUT_GCC": ["sv", "pp", "shmin", "shmax", "ucs", "fang", "pr"],
    "MW_BREAKDOWN_GCC": ["shmin", "shmax", "pp", "tstr"],
}


def run_parameter_tornado(
    results: pd.DataFrame,
    target_output: str,
    variation_pct: float = 10.0,
    *,
    stress_params: dict | None = None,
    tstr_multiplier: float = 0.15,
) -> tuple[pd.DataFrame, float, list[str]]:
    """One-at-a-time sensitivity of a DERIVED output to its governing-equation
    parameters (not the raw logs). The target's equation is recomputed directly
    from the base-case arrays in `results`, scaling one parameter at a time by
    ±variation_pct while holding the others fixed.

    Returns (tornado_df, base_value_canonical, skipped) — same shape as
    run_tornado_analysis so the figure builder is shared.
    """
    if target_output not in TORNADO_TARGET_PARAMS:
        raise ValueError(f"Parameter tornado is not available for '{target_output}'.")
    needed = {"SV_MPA", "PP_MPA", "SHMIN_MPA", "SHMAX_MPA", "PR_STA", "YME_STA_MPSI", "DEPTH"}
    if not needed.issubset(results.columns):
        raise ValueError("Parameter tornado needs a completed stress run (Sv, Pp, Shmin, SHmax, PR, YME).")

    n = len(results)
    p = {**default_stress_params(), **(stress_params or {})}
    to_ft = M_TO_FT if p["depth_unit"] == "m" else 1.0
    tvd_ft = pd.to_numeric(results["DEPTH"], errors="coerce").to_numpy(float) * to_ft

    def arr(col: str, conv: float = 1.0):
        return pd.to_numeric(results[col], errors="coerce").to_numpy(float) * conv

    base: dict = {
        "sv": arr("SV_MPA", MPA_TO_PSI),
        "pp": arr("PP_MPA", MPA_TO_PSI),
        "pr": arr("PR_STA"),
        "yme": arr("YME_STA_MPSI"),
        "shmin": arr("SHMIN_MPA", MPA_TO_PSI),
        "shmax": arr("SHMAX_MPA", MPA_TO_PSI),
        "ucs": arr("UCS_PSI") if "UCS_PSI" in results.columns else arr("UCS_MPA", MPA_TO_PSI),
        "fang": arr("FANG_DEG"),
        "tstr": arr("TSTR_PSI") if "TSTR_PSI" in results.columns else arr("TSTR_MPA", MPA_TO_PSI),
        "biot": float(p["biot"]), "ex": float(p["ex"]), "ey": float(p["ey"]),
    }
    shmax_mult = float(p.get("shmax_multiplier", 1.1))
    shmax_is_mult = p.get("shmax_method") == "multiplier"
    _ARRAY_KEYS = ("sv", "pp", "pr", "yme", "shmin", "shmax", "ucs", "fang", "tstr")

    def make_vals(scale_key: str | None = None, factor: float = 1.0) -> dict:
        v = dict(base)
        for k in _ARRAY_KEYS:
            v[k] = base[k].copy()
        if scale_key is not None:
            v[scale_key] = base[scale_key] * factor
        return v

    def recompute(vals: dict):
        """Return the target array in CANONICAL units (MPa for stresses, g/cc for MW)."""
        if target_output in ("SHMIN_MPA", "SHMAX_MPA"):
            out = np.full(n, np.nan)
            for i in range(n):
                sv, pp, pr, yme = vals["sv"][i], vals["pp"][i], vals["pr"][i], vals["yme"][i]
                if not (np.isfinite(sv) and np.isfinite(pp) and np.isfinite(pr) and np.isfinite(yme)):
                    continue
                if not (0.0 < pr < 0.5):
                    continue
                try:
                    hs = HorizontalStressesCalculation.calculate_poroelastic_horizontal_stresses(
                        overburden_stress=sv, pore_pressure=pp, poisson_ratio=pr,
                        youngs_modulus=yme, biot_coefficient=vals["biot"],
                        EX=vals["ex"], EY=vals["ey"])
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
                if target_output == "SHMIN_MPA":
                    out[i] = hs.shmin
                else:
                    out[i] = hs.shmin * shmax_mult if shmax_is_mult else hs.shmax
            return out * PSI_TO_MPA
        # mud-weight targets: recompute limit pressure (psi) then EMW (g/cc)
        pw = np.full(n, np.nan)
        for i in range(n):
            if target_output == "MW_BREAKDOWN_GCC":
                sx, sn, pp, ts = vals["shmax"][i], vals["shmin"][i], vals["pp"][i], vals["tstr"][i]
                if not all(np.isfinite(x) for x in (sx, sn, pp, ts)):
                    continue
                try:
                    pw[i] = WellboreStabilityCalculation.calculate_breakdown_calculation_vertical_well_analytical(
                        shmax=sx, shmin=sn, pprs=pp, tstr=ts)
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
            else:  # MW_BREAKOUT_GCC
                sx, sn, pp, sv = vals["shmax"][i], vals["shmin"][i], vals["pp"][i], vals["sv"][i]
                uc, fa, pr = vals["ucs"][i], vals["fang"][i], vals["pr"][i]
                if not all(np.isfinite(x) for x in (sx, sn, pp, sv, uc, fa, pr)):
                    continue
                try:
                    pw[i] = WellboreStabilityCalculation.calculate_breakout_calculation_vertical_well_mohr_coulomb_analytical(
                        shmax=sx, shmin=sn, pprs=pp, overburden_stress=sv, ucs=uc, fang=fa, pr_sta=pr)
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(tvd_ft > 0, pw / (PSI_FT_PER_GCC * tvd_ft), np.nan)

    base_value = float(np.nanmean(recompute(make_vals())))
    if not np.isfinite(base_value):
        raise ValueError("The base case produced no valid values for the selected target.")

    frac = float(variation_pct) / 100.0
    rows: list[dict] = []
    skipped: list[str] = []
    for key in TORNADO_TARGET_PARAMS[target_output]:
        try:
            low = float(np.nanmean(recompute(make_vals(key, 1.0 - frac))))
            high = float(np.nanmean(recompute(make_vals(key, 1.0 + frac))))
        except (ValueError, ZeroDivisionError, OverflowError):
            skipped.append(_PARAM_LABEL.get(key, key))
            continue
        if not (np.isfinite(low) and np.isfinite(high)):
            skipped.append(_PARAM_LABEL.get(key, key))
            continue
        rows.append({
            "Parameter": _PARAM_LABEL.get(key, key), "low": low, "high": high,
            "pct_low": 100.0 * (low - base_value) / base_value if base_value else np.nan,
            "pct_high": 100.0 * (high - base_value) / base_value if base_value else np.nan,
        })

    tornado = pd.DataFrame(rows)
    if tornado.empty:
        raise ValueError("No governing parameters could be varied for this target.")
    tornado["swing"] = (tornado["high"] - tornado["low"]).abs()
    tornado = tornado.sort_values("swing", ascending=False).reset_index(drop=True)
    return tornado, base_value, skipped


def generate_parameter_tornado_plot(
    results: pd.DataFrame,
    target_output: str,
    variation_pct: float = 10.0,
    *,
    unit_system: str = OILFIELD,
    stress_params: dict | None = None,
    tstr_multiplier: float = 0.15,
) -> tuple[go.Figure, pd.DataFrame, float, list[str]]:
    """Governing-parameter tornado + figure. Same return shape as generate_tornado_plot."""
    tornado, base_value, skipped = run_parameter_tornado(
        results, target_output, variation_pct,
        stress_params=stress_params, tstr_multiplier=tstr_multiplier,
    )
    fig, table, base_display = _build_tornado_figure(
        tornado, base_value, target_output, variation_pct, unit_system,
        subtitle="Varying governing-equation parameters (others held at base)",
    )
    return fig, table, base_display, skipped


# Front-end code below uses the `mc.` namespace for the engine above. Snapshot
# the engine's public globals into a namespace so this works regardless of how
# the file is launched (streamlit run, python -m, importlib, ...).
import types as _types

mc = _types.SimpleNamespace(**{_k: _v for _k, _v in dict(globals()).items()
                               if not _k.startswith("_")})


# ===========================================================================
# STREAMLIT FRONT END
# ===========================================================================

st.set_page_config(
    page_title="Quick 1D MEM-WBS Calculator",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Persist data across reruns
for key, default in {
    "raw_df": None,        # uploaded / sample input data
    "results_df": None,    # full workflow output (canonical units)
    "qc_summary": None,
    "qc_flags": None,      # per-sample QC flags (canonical column names)
    "data_source": None,   # label shown in the sidebar
    "load_messages": [],   # info messages from the data cleaner
    "unit_warnings": [],   # unit sanity warnings from the last run
    "tornado": None,       # last tornado analysis (fig, table, meta)
}.items():
    st.session_state.setdefault(key, default)

FLAG_COLORS = {
    "OK": "background-color: #1e7e34; color: white",
    "LOW": "background-color: #d39e00; color: black",
    "HIGH": "background-color: #c82333; color: white",
    "MISSING": "background-color: #6c757d; color: white",
}


def style_flags(table: pd.DataFrame, flags: pd.DataFrame, columns: list[str]) -> "pd.io.formats.style.Styler":
    """Color-code table cells according to their QC flag (matching column names)."""
    shown = [c for c in columns if c in table.columns]

    def _color(row_df: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=row_df.index, columns=row_df.columns)
        for col in row_df.columns:
            if col in flags.columns:
                css[col] = flags.loc[row_df.index, col].map(FLAG_COLORS).fillna("")
        return css

    return table[shown].style.apply(_color, axis=None).format(precision=3)


def _add_lithology_column(fig: go.Figure, depth_series, code_series, col: int) -> None:
    """Draw the lithology flag as a colored track (contiguous runs shaded by code)
    into subplot column `col`, plus one legend entry per lithology present."""
    depth = pd.to_numeric(depth_series, errors="coerce").to_numpy(dtype=float)
    codes = pd.to_numeric(code_series, errors="coerce").to_numpy(dtype=float)
    filled = np.where(np.isfinite(codes), codes, -1.0)
    n = len(filled)
    present = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and filled[j + 1] == filled[i]:
            j += 1
        c = int(filled[i])
        y0 = depth[i]
        y1 = depth[j + 1] if (j + 1) < n else depth[j]
        fig.add_shape(type="rect", x0=0.0, x1=1.0, y0=y0, y1=y1,
                      fillcolor=mc.LITHO_COLORS.get(c, "#ecf0f1"), line_width=0, layer="below",
                      row=1, col=col)
        present.append(c)
        i = j + 1
    for c in sorted(set(present)):
        label = mc.LITHO_NAME_BY_CODE.get(c, "Undefined")
        name = f"{label} ({c})" if c >= 0 else "Undefined"
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=11, color=mc.LITHO_COLORS.get(c, "#ecf0f1")),
                       name=name, legendgroup="lithology"),
            row=1, col=col,
        )
    fig.update_xaxes(visible=False, range=[0.0, 1.0], row=1, col=col)


def depth_track_figure(df: pd.DataFrame, depth_col: str, tracks: list[tuple[str, list[tuple[str, str]]]], height: int = 750, litho_codes=None) -> go.Figure:
    """Build a multi-track log plot (property vs depth, depth increasing downwards).

    tracks: list of (track_title, [(column, legend_name), ...])
    litho_codes: optional series (aligned to df) — when given, a colored
    lithology flag track is added as the first (leftmost) column.
    """
    has_litho_track = litho_codes is not None
    n_prop = len(tracks)
    ncols = n_prop + (1 if has_litho_track else 0)
    titles = (["Litho"] if has_litho_track else []) + [t[0] for t in tracks]
    widths = None
    if has_litho_track:
        raw = [0.5] + [2.2] * n_prop  # narrow lithology strip, wider property tracks
        total = sum(raw)
        widths = [w / total for w in raw]
    fig = make_subplots(
        rows=1,
        cols=ncols,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=titles,
        column_widths=widths,
    )
    offset = 1 if has_litho_track else 0
    if has_litho_track:
        _add_lithology_column(fig, df[depth_col], litho_codes, col=1)
    for i, (_, curves) in enumerate(tracks, start=1 + offset):
        for col, name in curves:
            if col not in df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df[col],
                    y=df[depth_col],
                    mode="lines",
                    name=name,
                    hovertemplate=f"{name}: %{{x:.3f}}<br>Depth: %{{y:.1f}}<extra></extra>",
                ),
                row=1,
                col=i,
            )
    fig.update_yaxes(autorange="reversed", title_text=depth_col, col=1)
    fig.update_layout(
        height=height,
        # legend well above the subplot titles so the two never overlap
        legend=dict(orientation="h", yanchor="bottom", y=1.14),
        margin=dict(t=120, b=40),
    )
    return fig


def mud_window_figure(disp: pd.DataFrame, depth_col: str, N: dict[str, str], unit: str, height: int = 720,
                      litho_codes=None, mw_line=None, ecd_line=None, casing_depths=None) -> go.Figure:
    """NEW: mud weight window plot.

    Safe window (green) is shaded between the breakout limit (min MW, shear
    failure) and the LOSS GRADIENT (max MW = minimum principal stress among
    Sv/SHmax/Shmin) — exceeding the loss gradient risks losses into
    natural/reopened fractures. Breakdown (fracture initiation), Pp and Sv
    are shown as reference lines. A lithology flag track is prepended when
    litho_codes is provided.
    """
    has_litho_track = litho_codes is not None
    if has_litho_track:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=["Litho", "Mud weight window"],
                            column_widths=[0.12, 0.88])
        _add_lithology_column(fig, disp[depth_col], litho_codes, col=1)
        mw_col = 2
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=["Mud weight window"])
        mw_col = 1

    def _line(canonical, label, color, dash=None, fill=None):
        c = N.get(canonical)
        if c in disp.columns:
            fig.add_trace(
                go.Scatter(
                    x=disp[c], y=disp[depth_col], mode="lines", name=label,
                    line=dict(color=color, dash=dash), fill=fill,
                    fillcolor="rgba(40, 167, 69, 0.18)" if fill else None,
                    hovertemplate=f"{label}: %{{x:.3f}} {unit}<br>Depth: %{{y:.1f}}<extra></extra>",
                ),
                row=1, col=mw_col,
            )

    # safe window: breakout (lower) -> loss gradient (upper, filled green)
    _line("MW_BREAKOUT_GCC", "Breakout limit (min MW)", "#c82333")
    _line("MW_LOSS_GCC", "Loss gradient / min σ (max MW)", "#6f42c1", fill="tonextx")
    # reference lines
    _line("MW_BREAKDOWN_GCC", "Breakdown limit (fracture)", "#1f77b4", dash="dashdot")
    _line("MW_PP_GCC", "Pore pressure EMW", "gray", dash="dot")
    _line("MW_SV_GCC", "Overburden EMW", "gray", dash="dash")

    # (5) user-planned MW and ECD as vertical lines (move via the tab sliders)
    depth_vals = pd.to_numeric(disp[depth_col], errors="coerce")
    y_top, y_bot = float(depth_vals.min()), float(depth_vals.max())
    if mw_line is not None:
        fig.add_trace(
            go.Scatter(x=[mw_line, mw_line], y=[y_top, y_bot], mode="lines",
                       name=f"MW = {mw_line:.2f} {unit}", line=dict(color="#0b6e4f", width=3)),
            row=1, col=mw_col,
        )
    if ecd_line is not None:
        fig.add_trace(
            go.Scatter(x=[ecd_line, ecd_line], y=[y_top, y_bot], mode="lines",
                       name=f"ECD = {ecd_line:.2f} {unit}", line=dict(color="#e67e22", width=3, dash="dash")),
            row=1, col=mw_col,
        )
    # (5) casing setting depths as horizontal lines
    for k, cd in enumerate(casing_depths or []):
        fig.add_hline(y=cd, line=dict(color="#111", width=1.5, dash="dot"), row=1, col=mw_col)
        fig.add_annotation(x=1.0, xref="x domain", y=cd, yref="y", showarrow=False,
                           text=f"Casing {k + 1}: {cd:g}", font=dict(size=10, color="#111"),
                           bgcolor="rgba(255,255,255,0.6)", xanchor="right", yanchor="bottom",
                           row=1, col=mw_col)

    fig.update_yaxes(autorange="reversed", title_text=depth_col, col=1)
    fig.update_xaxes(title_text=f"Equivalent mud weight ({unit})", row=1, col=mw_col)
    fig.update_layout(
        height=height,
        # title at the very top, legend BELOW the plot -> no overlap
        title=dict(text="Mud weight window (green = safe window: breakout → loss gradient)",
                   y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="top", y=-0.12),
        margin=dict(t=60, b=90),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar: units, data input & calculation settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🪨 Quick 1D MEM-WBS Calculator User Inputs")
    st.caption(
        "1D Mechanical Earth Model builder powered by "
        "[geomechpy](https://github.com/0xsmolrun/GeomechPy_smolrun)."
    )

    # (4) Sidebar sections 1-7 are collapsible via st.expander.
    with st.expander("1. Units", expanded=True):
        unit_system = st.selectbox(
            "Input/Output Units",
            mc.UNIT_SYSTEMS,
            index=0,
            help="Controls how the uploaded data is interpreted AND how results are displayed. "
            "Note: YME, UCS, TSTR, Sv, Pp, Shmin and SHmax are always shown in psi.",
        )
        depth_unit = st.radio(
            "Depth (MD) unit",
            mc.DEPTH_UNITS,
            index=0,
            horizontal=True,
            help="Unit of the depth column. MD is assumed ≈ TVD (vertical well) for the stress calculations.",
        )
        expected = dict(mc.INPUT_UNITS[unit_system])
        expected["DEPTH"] = depth_unit
        st.caption(
            "Expected input units — "
            + " · ".join(f"{curve}: {unit}" for curve, unit in expected.items())
        )

    with st.expander("2. Data input", expanded=True):
        uploaded = st.file_uploader(
            "Upload well log data (LAS / CSV / Excel)",
            type=["las", "csv", "txt", "xls", "xlsx"],
            help="One row per depth sample. LAS files are read with lasio and their curves "
            "auto-extracted. Required curves: DEPTH/MD, GR, RHOB, DTCO, DTSM. POROSITY is "
            "optional. Unit rows and -999.25/-9999 nulls are handled automatically.",
        )
        if uploaded is not None:
            try:
                st.session_state.raw_df, st.session_state.load_messages = mc.load_data(uploaded)
                st.session_state.data_source = f"📄 {uploaded.name}"
            except ValueError as exc:
                st.error(str(exc))

        if st.button("🧪 Load Sample Data", use_container_width=True):
            st.session_state.raw_df = mc.generate_sample_data(unit_system=unit_system)
            st.session_state.data_source = f"🧪 Synthetic sample well (2500-3000 m, {unit_system.lower()})"
            st.session_state.load_messages = []

        st.download_button(
            "⬇️ Download Example File",
            data=mc.sample_csv_bytes(unit_system=unit_system),
            file_name="mem_example_data.csv",
            mime="text/csv",
            use_container_width=True,
            help=f"Clean sample CSV (MD, GR, RHOB, DTCO, DTSM, POROSITY) in {unit_system.lower()}.",
        )

        if st.session_state.data_source:
            st.success(f"Loaded: {st.session_state.data_source}")
        for msg in st.session_state.load_messages:
            st.info(msg)
        if st.session_state.raw_df is not None:
            undetected = mc.missing_required_curves(list(st.session_state.raw_df.columns))
            if undetected:
                st.warning(
                    "Could not auto-detect column(s) for: "
                    + ", ".join(undetected)
                    + ". Map them manually below or check your file."
                )

    with st.expander("3. Column mapping", expanded=True):
        column_map: dict[str, str] = {}
        if st.session_state.raw_df is not None:
            options = ["-- not mapped --"] + list(st.session_state.raw_df.columns)
            for curve in mc.ALL_CURVES:
                required = curve in mc.REQUIRED_CURVES
                label = f"{curve} [{expected[curve]}] {'(required)' if required else '(optional)'}"
                choice = st.selectbox(
                    label,
                    options,
                    index=mc.guess_column(curve, list(st.session_state.raw_df.columns)),
                    key=f"map_{curve}",
                )
                column_map[curve] = "" if choice == "-- not mapped --" else choice
        else:
            st.info("Load data first to map columns.")

    with st.expander("4. Mechanical stratigraphy", expanded=False):
        # (2) Simplified lithology: one GR cutoff -> sandstone (0) vs shale (1)
        compute_litho = st.checkbox(
            "Flag lithology from GR",
            value=True,
            help="Split each sample into sandstone (0) or shale (1) with a single GR cutoff. "
            "The flag is shown on the stratigraphy tab and as a track on every output plot.",
        )
        gr_cutoff = None
        if compute_litho:
            gr_cutoff = st.slider(
                "GR cutoff (gAPI)",
                min_value=0.0, max_value=200.0, value=mc.DEFAULT_GR_CUTOFF, step=1.0,
                help="GR below the cutoff = sandstone (0); at or above = shale (1).",
            )
    
    with st.expander("5. Static properties", expanded=False):
        method_label = st.selectbox(
            "Dynamic → static YME correlation",
            list(mc.STATIC_YME_METHODS.keys()),
            help="Correlations from geomechpy.static_elastic_properties. "
            "Morales additionally requires a mapped POROSITY column.",
        )
        calibration_multiplier = st.slider(
            "Static YME calibration multiplier",
            min_value=0.5, max_value=2.0, value=1.0, step=0.05,
            help="Scales the correlation output — use it to calibrate against core test data.",
        )
        pr_multiplier = st.slider(
            "Static Poisson's ratio multiplier",
            min_value=0.5, max_value=2.0, value=1.0, step=0.05,
        )
        custom_a, custom_b = 0.5, 1.0
        if "power" in method_label:
            custom_a = st.number_input("Custom multiplier a", value=0.5, format="%.4f")
            custom_b = st.number_input("Custom exponent b", value=1.0, format="%.4f")
        elif "linear" in method_label:
            custom_a = st.number_input("Custom slope a", value=0.8, format="%.4f")
            custom_b = st.number_input("Custom intercept b (Mpsi)", value=0.0, format="%.4f")

    with st.expander("6. Rock strength", expanded=False):
        ucs_method_label = st.selectbox(
            "UCS method",
            list(mc.UCS_METHODS.keys()),
            help="All UCS correlations available in geomechpy.rock_strength, plus a constant-value fallback.",
        )
        ucs_method = mc.UCS_METHODS[ucs_method_label]
        ucs_constant_mpa = 50.0
        if ucs_method == "constant":
            ucs_constant_mpa = st.number_input("Constant UCS (MPa)", value=50.0, min_value=0.1, format="%.1f")
        fang_method_label = st.selectbox(
            "Friction angle method",
            list(mc.FANG_METHODS.keys()),
            help="All FANG correlations available in geomechpy.rock_strength, plus a constant-value fallback.",
        )
        fang_method = mc.FANG_METHODS[fang_method_label]
        fang_constant_deg = 30.0
        fang_gr_min, fang_gr_max = 15.0, 120.0
        if fang_method == "constant":
            fang_constant_deg = st.number_input("Constant friction angle (deg)", value=30.0, min_value=1.0, max_value=60.0, format="%.1f")
        elif fang_method == "gr_linear":
            cgr1, cgr2 = st.columns(2)
            fang_gr_min = cgr1.number_input("GR min (clean sand, gAPI)", value=15.0, format="%.1f",
                                            help="GR at clean sand → FANG = 45°.")
            fang_gr_max = cgr2.number_input("GR max (pure shale, gAPI)", value=120.0, format="%.1f",
                                            help="GR at pure shale → FANG = 15°. Must differ from GR min.")
        ucs_multiplier = st.slider(
            "UCS calibration multiplier",
            min_value=0.5, max_value=2.0, value=1.0, step=0.05,
            help="Scales the UCS output (and TSTR derived from it) — calibrate against core, "
            "like the static YME multiplier.",
        )
        tstr_multiplier = st.slider(
            "Tensile strength / UCS ratio",
            min_value=0.05, max_value=0.30, value=0.15, step=0.01,
            help="TSTR = ratio × UCS (geomechpy default is 0.15).",
        )

    with st.expander("7. Stress & wellbore stability", expanded=False):
        compute_stress = st.checkbox(
            "Compute stresses & mud weight window",
            value=True,
            help="Adds overburden, pore pressure, horizontal stresses and the vertical-well "
            "mud weight window (geomechpy gradient-based + poroelastic methods).",
        )
        stress_params = None
        if compute_stress:
            setting = st.radio("Well setting", mc.WELL_SETTINGS, horizontal=True)
            air_gap = st.number_input(
                f"Air gap / KB elevation ({depth_unit})", value=0.0, min_value=0.0, format="%.1f",
                help="Drill floor to ground level (onshore) or to mean sea level (offshore).",
            )
            water_depth, sea_gradient = 0.0, 0.47
            if setting == "Offshore":
                water_depth = st.number_input(f"Water depth ({depth_unit})", value=0.0, min_value=0.0, format="%.1f")
                sea_gradient = st.number_input("Sea water gradient (psi/ft)", value=0.47, min_value=0.30, max_value=0.60, format="%.3f")
            ovb_source_label = st.selectbox(
                "Overburden gradient source",
                list(mc.OVB_GRADIENT_SOURCES.keys()),
                help="Constant lithostatic gradient, or a gradient derived from the mean of the "
                "mapped RHOB log (mean g/cc × 0.4335 psi/ft).",
            )
            ovb_source = mc.OVB_GRADIENT_SOURCES[ovb_source_label]
            ovb_gradient = st.number_input(
                "Lithostatic gradient (psi/ft)", value=1.05, min_value=0.5, max_value=1.5, format="%.3f",
                disabled=(ovb_source == "density"),
                help="Typical 1.0–1.1 psi/ft. Ignored when the gradient is derived from RHOB.",
            )
            pp_gradient = st.number_input(
                "Pore pressure gradient (psi/ft)", value=0.47, min_value=0.30, max_value=1.0, format="%.3f",
                help="Hydrostatic ≈ 0.433–0.47 psi/ft; higher = overpressure.",
            )
            shmax_method_label = st.selectbox("SHmax method", list(mc.SHMAX_METHODS.keys()))
            shmax_method = mc.SHMAX_METHODS[shmax_method_label]
            shmax_multiplier = 1.1
            if shmax_method == "multiplier":
                shmax_multiplier = st.slider("SHmax / Shmin multiplier", 1.0, 2.0, 1.1, 0.05)
            biot = st.slider("Biot coefficient", 0.5, 1.0, 1.0, 0.05)
            c_ex, c_ey = st.columns(2)
            ex = c_ex.number_input("Tectonic strain EX", value=0.0001, format="%.5f",
                                   help="Poroelastic tectonic strain term (Shmin direction).")
            ey = c_ey.number_input("Tectonic strain EY", value=0.009, format="%.5f",
                                   help="Poroelastic tectonic strain term (SHmax direction). Keep EY ≥ EX.")
            stress_params = {
                "setting": setting,
                "depth_unit": depth_unit,
                "air_gap": air_gap,
                "water_depth": water_depth,
                "sea_gradient_psift": sea_gradient,
                "ovb_source": ovb_source,
                "ovb_gradient_psift": ovb_gradient,
                "pp_gradient_psift": pp_gradient,
                "shmax_method": shmax_method,
                "shmax_multiplier": shmax_multiplier,
                "biot": biot,
                "ex": ex,
                "ey": ey,
            }

    st.divider()
    run_clicked = st.button(
        "🚀 Run MEM Calculation",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.raw_df is None,
    )

# Bundle the workflow settings once — used by the run button and the tornado tab.
workflow_settings = dict(
    method_label=method_label,
    calibration_multiplier=calibration_multiplier,
    pr_multiplier=pr_multiplier,
    tstr_multiplier=tstr_multiplier,
    custom_a=custom_a,
    custom_b=custom_b,
    unit_system=unit_system,
    ucs_method=ucs_method,
    fang_method=fang_method,
    ucs_constant_mpa=ucs_constant_mpa,
    fang_constant_deg=fang_constant_deg,
    fang_gr_min=fang_gr_min,
    fang_gr_max=fang_gr_max,
    ucs_multiplier=ucs_multiplier,
    gr_cutoff=gr_cutoff,
    stress_params=stress_params,
)

# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------

if run_clicked:
    try:
        st.session_state.unit_warnings = mc.check_unit_sanity(
            st.session_state.raw_df, column_map, unit_system
        )
        with st.spinner("Computing properties, stresses and stability..."):
            results = mc.run_full_workflow(
                data=st.session_state.raw_df,
                column_map=column_map,
                **workflow_settings,
            )
        st.session_state.results_df = results
        st.session_state.qc_summary, st.session_state.qc_flags = mc.run_qc(results)
        st.toast("MEM calculation complete ✅")
    except ValueError as exc:
        st.error(f"⚠️ {exc}")
    except Exception as exc:  # keep the app alive on unexpected input
        st.error(f"Unexpected error during calculation: {exc}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

for warning in st.session_state.unit_warnings:
    st.warning(f"⚠️ Unit check: {warning}")

results = st.session_state.results_df
flags = st.session_state.qc_flags

# Convert canonical results into the selected display unit system.
# N maps canonical column names -> display names (e.g. YME_DYN_GPA -> 'YME_DYN [Mpsi]').
# has_stress requires the FULL current stress column set (incl. the loss gradient)
# so results from an older run/app version prompt a re-run instead of KeyError.
STRESS_COLUMNS = ["SV_MPA", "PP_MPA", "SHMIN_MPA", "SHMAX_MPA", "LOSS_P_MPA",
                  "MW_BREAKOUT_GCC", "MW_BREAKDOWN_GCC", "MW_LOSS_GCC"]
STRESS_INFO = (
    "Enable **Compute stresses & mud weight window** in the sidebar and click "
    "**🚀 Run MEM Calculation** — the results currently in memory don't include "
    "the full stress profile (they may be from an older run or app version)."
)
if results is not None:
    disp, N = mc.display_results(results, unit_system)
    flags_disp = flags.rename(columns=N) if flags is not None else None
    DEPTH = N["DEPTH"]
    has_stress = all(c in results.columns for c in STRESS_COLUMNS)
    has_litho = "LITHO_CODE" in results.columns and results["LITHO_CODE"].notna().any()
else:
    disp, N, flags_disp, DEPTH, has_stress, has_litho = None, {}, None, None, False, False


def with_litho(cols: list[str]) -> list[str]:
    """Insert the lithology display column right after DEPTH when available,
    so every results table shows lithology next to the output properties."""
    if has_litho and N.get("LITHO_CODE") and N["LITHO_CODE"] not in cols:
        return [cols[0], N["LITHO_CODE"]] + cols[1:]
    return cols


# Lithology code series passed to every depth plot so the flag renders as a track.
litho_arg = disp[N["LITHO_CODE"]] if (has_litho and disp is not None) else None


def lithology_figure(depth_series, code_series, height: int = 650, title: str = "Lithology (from GR)") -> go.Figure:
    """Colored lithology strip vs depth (contiguous runs shaded by code)."""
    depth = pd.to_numeric(depth_series, errors="coerce").to_numpy(dtype=float)
    codes = pd.to_numeric(code_series, errors="coerce").to_numpy(dtype=float)
    filled = np.where(np.isfinite(codes), codes, -1.0)
    n = len(filled)
    fig = go.Figure()
    present = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and filled[j + 1] == filled[i]:
            j += 1
        code = int(filled[i])
        y0 = depth[i]
        y1 = depth[j + 1] if (j + 1) < n else depth[j]
        fig.add_shape(type="rect", xref="x", yref="y", x0=0.0, x1=1.0, y0=y0, y1=y1,
                      fillcolor=mc.LITHO_COLORS.get(code, "#ecf0f1"), line_width=0, layer="below")
        present.append(code)
        i = j + 1
    for code in sorted(set(present)):
        label = mc.LITHO_NAME_BY_CODE.get(code, "Undefined")
        name = f"{label} ({code})" if code >= 0 else "Undefined"
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=mc.LITHO_COLORS.get(code, "#ecf0f1")),
                                 name=name))
    fig.update_xaxes(visible=False, range=[0.0, 1.0])
    fig.update_yaxes(autorange="reversed", title_text=DEPTH if DEPTH else "Depth")
    fig.update_layout(height=height, title=title,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(t=70, b=40))
    return fig


st.title("Quick 1D MEM-WBS Calculator")
st.markdown(
    "Build a quick-look **Mechanical Earth Model** from standard well logs: "
    "mechanical stratigraphy → stresses → rock properties → wellbore stability, with built-in QC."
)

for warning in st.session_state.unit_warnings:
    st.warning(f"⚠️ Unit check: {warning}")

(
    tab_input,
    tab_strat,
    tab_ovb,
    tab_rock,
    tab_hstress,
    tab_wbs,
    tab_qc,
    tab_tornado,
) = st.tabs(
    [
        "📥 Data Input",
        "🪨 Mechanical Stratigraphy",
        "🏔️ Overburden & Pore Pressure",
        "🧱 Rock Properties",
        "↔️ Horizontal Stress",
        "🛢️ Wellbore Stability",
        "✅ QC & Results",
        "🌪️ Sensitivity (Tornado)",
    ]
)

# --- Tab 1: Data input ------------------------------------------------------
with tab_input:
    st.subheader("Data input")
    # Clear expected-columns reference as a table, in the selected unit system.
    curve_meta = [
        ("DEPTH", "MD", "Measured depth (assumed ≈ TVD, vertical well)", "Required", "2500.0"),
        ("GR", "Gamma ray", "Shale/lithology indicator; drives mechanical stratigraphy", "Required", "75.0"),
        ("RHOB", "Bulk density", "Formation bulk density (elastic properties, Sv option)", "Required", "2.45"),
        ("DTCO", "Compressional slowness", "P-sonic transit time (Vp, moduli, McNally UCS, Lal FANG)", "Required", "85.0"),
        ("DTSM", "Shear slowness", "S-sonic transit time (Vs, moduli)", "Required", "150.0"),
        ("POROSITY", "Porosity", "Total/effective porosity (only for the Morales static method)", "Optional", "0.18"),
    ]
    ref = pd.DataFrame(
        [
            {
                "Curve": c,
                "Description": f"{name} — {desc}",
                f"Unit ({unit_system})": expected[c],
                "Requirement": req,
                "Example value": ex,
            }
            for c, name, desc, req, ex in curve_meta
        ]
    )
    st.markdown(
        f"**Expected input columns** — one row per depth sample. Column *names* can be anything; "
        f"map them to these curves in the sidebar. Units below follow the selected **{unit_system}** system."
    )
    st.dataframe(ref, use_container_width=True, hide_index=True)
    st.caption(
        "Tip: use **⬇️ Download Example File** in the sidebar for a correctly formatted CSV template, "
        "or **🧪 Load Sample Data** to try the app immediately. Unit rows under the header and "
        "-999.25 / -9999 null flags are handled automatically on upload."
    )

    if st.session_state.raw_df is None:
        st.info("👈 Upload a CSV/Excel file or click **Load Sample Data** in the sidebar to get started.")
    else:
        df_in = st.session_state.raw_df
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_in):,}")
        c2.metric("Columns", df_in.shape[1])
        depth_col = column_map.get("DEPTH") if column_map else None
        if depth_col:
            c3.metric("Depth range", f"{df_in[depth_col].min():.0f} – {df_in[depth_col].max():.0f} {depth_unit}")
        st.subheader("Uploaded data preview")
        st.dataframe(df_in, use_container_width=True, height=380)
        with st.expander("Basic statistics"):
            st.dataframe(df_in.describe().T, use_container_width=True)

# --- Tab 2: Mechanical Stratigraphy -----------------------------------------
with tab_strat:
    st.subheader("Mechanical Stratigraphy")
    st.markdown(
        "Each depth is flagged from **GR** using the single cutoff defined in the sidebar "
        "(**6. Mechanical stratigraphy**): GR below the cutoff = **sandstone (code 0)**, at or above "
        "= **shale (code 1)**. The flag is carried through and shown as a track on every output plot."
    )
    if results is None:
        st.info("Run the calculation from the sidebar to generate the lithology flag.")
    elif not has_litho:
        st.info("Enable **Flag lithology from GR** in the sidebar (section 6) and re-run.")
    else:
        counts = mc.lithology_counts(results)
        cols = st.columns(len(mc.LITHO_NAME_BY_CODE))
        for col, (code, name) in zip(cols, mc.LITHO_NAME_BY_CODE.items()):
            row = counts[counts["Code"] == code]
            pct = float(row["Fraction %"].iloc[0]) if not row.empty else 0.0
            col.metric(f"{name} ({code})", f"{pct:.1f}%")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Lithology fractions**")
            st.dataframe(counts, use_container_width=True, hide_index=True)
        with c2:
            st.plotly_chart(
                depth_track_figure(
                    disp, DEPTH,
                    [("GR (gAPI)", [(N["GR"], "GR")])],
                    height=620, litho_codes=litho_arg,
                ),
                use_container_width=True,
            )
        st.plotly_chart(
            lithology_figure(disp[DEPTH], disp[N["LITHO_CODE"]], height=620),
            use_container_width=True,
            key="litho_strat",
        )

# --- Tab 3: Overburden & Pore Pressure --------------------------------------
with tab_ovb:
    if results is None:
        st.info("Run the calculation from the sidebar to see the overburden and pore pressure profiles.")
    elif not has_stress:
        st.info(STRESS_INFO)
    else:
        st.subheader(f"Overburden stress & pore pressure ({unit_system})")
        sp = stress_params or {}
        st.caption(
            f"Setting: **{sp.get('setting', '-')}** · air gap {sp.get('air_gap', 0):g} {depth_unit}"
            + (f" · water depth {sp.get('water_depth', 0):g} {depth_unit}" if sp.get("setting") == "Offshore" else "")
            + f" · Sv gradient source: {sp.get('ovb_source', '-')}"
            + f" · Pp gradient {sp.get('pp_gradient_psift', 0):.3f} psi/ft. MD assumed ≈ TVD (vertical well)."
        )
        ovb_cols = [DEPTH] + [N[c] for c in ["SV_MPA", "PP_MPA", "MW_SV_GCC", "MW_PP_GCC"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(ovb_cols)), use_container_width=True, height=380)
        mw_unit = mc.display_unit("MW_PP_GCC", unit_system)
        st.plotly_chart(
            depth_track_figure(
                disp, DEPTH,
                [
                    (f"Pressure ({mc.display_unit('SV_MPA', unit_system)})", [(N["SV_MPA"], "Sv"), (N["PP_MPA"], "Pp")]),
                    (f"Equivalent gradients ({mw_unit})", [(N["MW_SV_GCC"], "Sv EMW"), (N["MW_PP_GCC"], "Pp EMW")]),
                ],
                height=650, litho_codes=litho_arg,
            ),
            use_container_width=True,
        )

# --- Tab 4: Rock Properties (dynamic + static + strength) -------------------
with tab_rock:
    if results is None:
        st.info("Run the calculation from the sidebar to see rock properties.")
    else:
        st.subheader(f"Rock properties ({unit_system})")
        st.caption(
            f"Static YME: **{method_label}** ×{calibration_multiplier:.2f} · PR ×{pr_multiplier:.2f} · "
            f"UCS: **{ucs_method_label}** ×{ucs_multiplier:.2f} · FANG: **{fang_method_label}** · "
            f"TSTR = {tstr_multiplier:.2f} × UCS."
        )

        st.markdown("**Dynamic elastic properties** (from DTCO / DTSM / RHOB)")
        dyn_cols = [DEPTH] + [N[c] for c in ["VP_MS", "VS_MS", "VPVS", "YME_DYN_GPA", "PR_DYN", "K_DYN_GPA", "G_DYN_GPA", "LAME_DYN_GPA", "M_DYN_GPA"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(dyn_cols)), use_container_width=True, height=300)

        st.markdown("**Static elastic + rock strength**")
        sta_cols = [DEPTH] + [N[c] for c in ["YME_DYN_GPA", "YME_STA_GPA", "PR_DYN", "PR_STA", "UCS_MPA", "TSTR_MPA", "FANG_DEG"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(sta_cols)), use_container_width=True, height=300)

        st.plotly_chart(
            depth_track_figure(
                disp, DEPTH,
                [
                    (f"Velocities ({mc.display_unit('VP_MS', unit_system)})", [(N["VP_MS"], "Vp"), (N["VS_MS"], "Vs")]),
                    (f"Young's mod. ({mc.display_unit('YME_DYN_GPA', unit_system)})", [(N["YME_DYN_GPA"], "E dyn"), (N["YME_STA_GPA"], "E sta")]),
                    ("Poisson's ratio", [(N["PR_DYN"], "ν dyn"), (N["PR_STA"], "ν sta")]),
                    (f"UCS / TSTR ({mc.display_unit('UCS_MPA', unit_system)})", [(N["UCS_MPA"], "UCS"), (N["TSTR_MPA"], "TSTR")]),
                    ("Friction angle (°)", [(N["FANG_DEG"], "FANG")]),
                ],
                height=680, litho_codes=litho_arg,
            ),
            use_container_width=True,
        )

# --- Tab 5: Horizontal Stress -----------------------------------------------
with tab_hstress:
    if results is None:
        st.info("Run the calculation from the sidebar to see horizontal stresses.")
    elif not has_stress:
        st.info(STRESS_INFO)
    else:
        st.subheader(f"Horizontal stresses ({unit_system})")
        sp = stress_params or {}
        method_txt = [k for k, v in mc.SHMAX_METHODS.items() if v == sp.get("shmax_method")]
        st.caption(
            f"Method: **{method_txt[0] if method_txt else '-'}** · Biot {sp.get('biot', 1.0):.2f} · "
            f"EX {sp.get('ex', 0):g} · EY {sp.get('ey', 0):g}"
            + (f" · SHmax multiplier ×{sp.get('shmax_multiplier', 1.1):.2f}" if sp.get("shmax_method") == "multiplier" else "")
            + ". Shmin from the poroelastic equation (static PR & YME, Biot, tectonic strains)."
        )
        hs_cols = [DEPTH] + [N[c] for c in ["SV_MPA", "PP_MPA", "SHMIN_MPA", "SHMAX_MPA", "Q_FACTOR", "SH_RATIO"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(hs_cols)), use_container_width=True, height=380)

        q_med = pd.to_numeric(results["Q_FACTOR"], errors="coerce").median()
        if pd.notna(q_med):
            regime = "Normal" if q_med < 1 else ("Strike-slip" if q_med < 2 else "Reverse")
            st.metric("Median stress regime q-factor", f"{q_med:.2f}", help="q<1 normal · 1–2 strike-slip · 2–3 reverse")
            st.caption(f"Dominant stress regime over the interval: **{regime} faulting**.")

        st.plotly_chart(
            depth_track_figure(
                disp, DEPTH,
                [
                    (
                        f"Stresses ({mc.display_unit('SV_MPA', unit_system)})",
                        [(N["SV_MPA"], "Sv"), (N["SHMAX_MPA"], "SHmax"), (N["SHMIN_MPA"], "Shmin"), (N["PP_MPA"], "Pp")],
                    ),
                    ("q-factor (-)", [(N["Q_FACTOR"], "q")]),
                    ("SHmax/Shmin (-)", [(N["SH_RATIO"], "ratio")]),
                ],
                height=650, litho_codes=litho_arg,
            ),
            use_container_width=True,
        )

# --- Tab 6: Wellbore Stability ----------------------------------------------
with tab_wbs:
    if results is None:
        st.info("Run the calculation from the sidebar to see the wellbore stability results.")
    elif not has_stress:
        st.info(STRESS_INFO)
    else:
        st.subheader(f"Wellbore stability — vertical well ({unit_system})")
        st.caption(
            "Breakout limit: Mohr-Coulomb shear failure (Kirsch, analytical) — drilling below it risks breakouts. "
            "Loss gradient: minimum principal stress among Sv, SHmax and Shmin — drilling above it risks losses. "
            "The green band is the safe mud weight window (**breakout limit → loss gradient**). "
            "Breakdown (fracture initiation, Hubbert & Willis) is shown as a reference only."
        )

        mw_unit = mc.display_unit("MW_BREAKOUT_GCC", unit_system)
        # Safe window: breakout (lower bound) up to the loss gradient (upper bound = min principal stress)
        window = results["MW_LOSS_GCC"] - results["MW_BREAKOUT_GCC"]
        n_valid = int(window.notna().sum())
        n_closed = int((window < 0).sum())
        c1, c2, c3 = st.columns(3)
        factor = mc.PPG_PER_GCC if unit_system == mc.OILFIELD else 1.0
        c1.metric(f"Median breakout limit ({mw_unit})", f"{results['MW_BREAKOUT_GCC'].median() * factor:.2f}")
        c2.metric(f"Median loss gradient ({mw_unit})", f"{results['MW_LOSS_GCC'].median() * factor:.2f}")
        c3.metric(f"Median window width ({mw_unit})", f"{window.median() * factor:.2f}")
        if n_closed:
            st.warning(
                f"⚠️ The mud weight window is closed (breakout limit above the loss gradient) at "
                f"{n_closed} of {n_valid} depth samples — no safe static mud weight exists there."
            )

        wbs_cols = [DEPTH] + [N[c] for c in ["PW_BREAKOUT_MPA", "PW_BREAKDOWN_MPA", "LOSS_P_MPA", "MW_BREAKOUT_GCC", "MW_BREAKDOWN_GCC", "MW_LOSS_GCC", "MW_PP_GCC", "MW_SV_GCC"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(wbs_cols)), use_container_width=True, height=340)

        # (5) Interactive mud-weight planning: MW + ECD sliders and casing setting depths.
        st.markdown("#### 🛠️ Mud weight planning")
        bo_disp = results["MW_BREAKOUT_GCC"] * factor  # display MW units (ppg or g/cc)
        loss_disp = results["MW_LOSS_GCC"] * factor
        lo = float(np.nanmin(bo_disp)) if bo_disp.notna().any() else 8.0
        hi = float(np.nanmax(loss_disp)) if loss_disp.notna().any() else 18.0
        pad = max(0.5, 0.1 * (hi - lo))
        slo, shi = round(lo - pad, 1), round(hi + pad, 1)
        mid = round((lo + hi) / 2, 2)
        step = 0.05 if unit_system == mc.OILFIELD else 0.01

        cmw, cecd = st.columns(2)
        mw_value = cmw.slider(
            f"Mud weight — MW ({mw_unit})", slo, shi, value=mid, step=step, key="wbs_mw",
            help="Planned static mud weight. Slide left/right to move the green MW line on the plot; "
            "keep it inside the safe window (breakout → loss gradient).",
        )
        ecd_value = cecd.slider(
            f"Equivalent circulating density — ECD ({mw_unit})", slo, shi + pad,
            value=round(min(mw_value + (0.3 if unit_system == mc.OILFIELD else 0.04), shi + pad), 2),
            step=step, key="wbs_ecd",
            help="Dynamic (circulating) density. Slide to move the orange ECD line; keep it below the loss gradient.",
        )

        st.caption(
            "**Casing setting depths** — add one row per casing shoe. These depths split the well into "
            "sections and draw horizontal markers on the plot to help pick MW per section."
        )
        casing_default = pd.DataFrame({f"Casing shoe depth ({depth_unit})": pd.Series([], dtype=float)})
        casing_edit = st.data_editor(
            casing_default, num_rows="dynamic", hide_index=True, use_container_width=True, key="wbs_casing",
        )
        casing_depths = sorted(
            float(v) for v in casing_edit.iloc[:, 0].tolist()
            if pd.notna(v) and np.isfinite(v)
        )

        # Per-section safe MW range (breakout .. loss gradient) between casing shoes.
        depth_num = pd.to_numeric(results["DEPTH"], errors="coerce")
        edges = [float(depth_num.min())] + casing_depths + [float(depth_num.max())]
        edges = sorted(set(round(e, 3) for e in edges))
        section_rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (depth_num >= a) & (depth_num <= b)
            if not m.any():
                continue
            sec_lo = float((results.loc[m, "MW_BREAKOUT_GCC"] * factor).max())  # highest breakout = min safe MW
            sec_hi = float((results.loc[m, "MW_LOSS_GCC"] * factor).min())      # lowest loss = max safe MW
            ok = "✅" if (np.isfinite(sec_lo) and np.isfinite(sec_hi) and sec_lo <= mw_value <= sec_hi) else "⚠️"
            section_rows.append({
                f"Top ({depth_unit})": round(a, 1),
                f"Base ({depth_unit})": round(b, 1),
                f"Min MW ({mw_unit})": round(sec_lo, 2),
                f"Max MW ({mw_unit})": round(sec_hi, 2),
                f"Planned MW ({mw_unit})": round(mw_value, 2),
                "MW in window?": ok,
            })
        if section_rows:
            st.markdown("**Safe MW range per section** (min = highest breakout, max = lowest loss gradient):")
            st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)

        st.plotly_chart(
            mud_window_figure(disp, DEPTH, N, mw_unit, litho_codes=litho_arg,
                              mw_line=mw_value, ecd_line=ecd_value, casing_depths=casing_depths),
            use_container_width=True,
        )

# --- Tab 7: QC & Results ----------------------------------------------------
with tab_qc:
    if results is None or st.session_state.qc_summary is None:
        st.info("Run the calculation from the sidebar to generate the QC report.")
    else:
        qc = st.session_state.qc_summary
        status = mc.qc_status(qc)
        badge = {"PASS": "🟢 PASS", "WARNING": "🟡 WARNING", "FAIL": "🔴 FAIL"}[status]
        st.subheader(f"QC report — overall status: {badge}")
        st.caption(
            "Each curve is checked against standard geomechanical ranges "
            "(units shown in the Unit column). LOW/HIGH = outside range, MISSING = null/non-numeric."
        )

        def _pct_color(v):
            if v >= 95:
                return "background-color: #1e7e34; color: white"
            if v >= 70:
                return "background-color: #d39e00; color: black"
            return "background-color: #c82333; color: white"

        st.dataframe(
            qc.style.map(_pct_color, subset=["% in range"]).format({"% in range": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

        flagged_canonical = [c for c in flags.columns if (flags[c] != "OK").any()]
        if flagged_canonical:
            st.markdown("**Flagged samples** (rows where at least one curve is out of range or missing):")
            bad_rows = flags[flagged_canonical].ne("OK").any(axis=1)
            show_cols = with_litho([DEPTH] + [N[c] for c in flagged_canonical if c in N])
            st.dataframe(
                style_flags(disp.loc[bad_rows], flags_disp.loc[bad_rows], show_cols),
                use_container_width=True,
                height=300,
            )
        else:
            st.success("All samples passed QC — no flags raised. 🎉")

        st.divider()
        st.subheader(f"Composite MEM display ({unit_system})")
        composite_tracks = [
            ("GR (gAPI)", [(N["GR"], "GR")]),
            (f"Slowness ({mc.display_unit('DTCO', unit_system)})", [(N["DTCO"], "DTCO"), (N["DTSM"], "DTSM")]),
            (f"E ({mc.display_unit('YME_DYN_GPA', unit_system)})", [(N["YME_DYN_GPA"], "E dyn"), (N["YME_STA_GPA"], "E sta")]),
            (f"UCS / TSTR ({mc.display_unit('UCS_MPA', unit_system)})", [(N["UCS_MPA"], "UCS"), (N["TSTR_MPA"], "TSTR")]),
        ]
        if has_stress:
            mw_unit = mc.display_unit("MW_PP_GCC", unit_system)
            composite_tracks += [
                (
                    f"Stresses ({mc.display_unit('SV_MPA', unit_system)})",
                    [(N["SV_MPA"], "Sv"), (N["SHMAX_MPA"], "SHmax"), (N["SHMIN_MPA"], "Shmin"), (N["PP_MPA"], "Pp")],
                ),
                (
                    f"Mud window ({mw_unit})",
                    [(N["MW_BREAKOUT_GCC"], "MW min"), (N["MW_LOSS_GCC"], "MW max"), (N["MW_PP_GCC"], "Pp EMW")],
                ),
            ]
        st.plotly_chart(depth_track_figure(disp, DEPTH, composite_tracks, height=800, litho_codes=litho_arg), use_container_width=True)

        with st.expander("Crossplot explorer"):
            numeric_cols = [c for c in disp.columns if pd.api.types.is_numeric_dtype(disp[c])]
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X axis", numeric_cols, index=numeric_cols.index(N["YME_DYN_GPA"]) if N.get("YME_DYN_GPA") in numeric_cols else 0)
            y_col = c2.selectbox("Y axis", numeric_cols, index=numeric_cols.index(N["YME_STA_GPA"]) if N.get("YME_STA_GPA") in numeric_cols else 0)
            color_col = c3.selectbox("Color by", numeric_cols, index=numeric_cols.index(N["GR"]) if N.get("GR") in numeric_cols else 0)
            xfig = go.Figure(
                go.Scatter(
                    x=disp[x_col], y=disp[y_col], mode="markers",
                    marker=dict(color=disp[color_col], colorscale="Viridis", showscale=True, colorbar_title=color_col),
                    hovertemplate=f"{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<extra></extra>",
                )
            )
            xfig.update_layout(xaxis_title=x_col, yaxis_title=y_col, height=520)
            st.plotly_chart(xfig, use_container_width=True)

        st.divider()
        st.download_button(
            f"⬇️ Download results as CSV ({unit_system})",
            data=mc.results_to_csv_bytes(disp),
            file_name="mem_results.csv",
            mime="text/csv",
            type="primary",
        )

# --- Tab 8: Sensitivity analysis (Tornado plot) -----------------------------
with tab_tornado:
    st.subheader("Sensitivity Analysis (Tornado Plot)")
    if st.session_state.raw_df is None:
        st.info("👈 Load data in the sidebar first — the tornado plot needs a base case.")
    else:
        LOG_MODE = "Input logs (recompute full workflow)"
        PARAM_MODE = "Governing parameters (recompute one equation)"
        variation_mode = st.radio(
            "Variation type",
            [LOG_MODE, PARAM_MODE],
            key="tornado_mode",
            help=(
                "Input logs: perturb a raw curve (GR, RHOB, sonic, porosity) or the static-YME "
                "multiplier and re-run the whole workflow.\n\n"
                "Governing parameters: perturb the actual inputs of one result's equation — "
                "e.g. Shmin varies with Sv, Pp, Poisson's ratio, YME, Biot and the tectonic strains."
            ),
        )

        # ----- Mode A: vary the raw input logs, recompute the full workflow -----
        if variation_mode == LOG_MODE:
            st.markdown(
                "Using the loaded data as the *base case*, each input (GR, RHOB, DTCO, DTSM, POROSITY and "
                "the static YME multiplier) is varied one at a time by the selected percentage while "
                "everything else is held fixed, and the workflow is recomputed. Bars show how the "
                "depth-averaged target moves — longer bar = more sensitive. GR has no bar unless the "
                "target depends on it (e.g. GR-linear FANG); POROSITY only matters for the Morales "
                "static method."
            )
            c1, c2 = st.columns(2)
            tornado_targets = list(mc.TORNADO_TARGETS)
            if stress_params is not None:
                tornado_targets += mc.TORNADO_STRESS_TARGETS
            target_options = [mc.display_name(t, unit_system) for t in tornado_targets]
            target_label_sel = c1.selectbox("Target Output", target_options, index=0, key="tornado_target",
                                            help="Result whose sensitivity is analysed (depth-averaged mean).")
            target_canonical = tornado_targets[target_options.index(target_label_sel)]
            variation_pct = c2.select_slider("Variation Range", options=[5, 10, 20], value=10,
                                             format_func=lambda v: f"±{v}%", key="tornado_pct")

            if st.button("🌪️ Generate Tornado Plot", type="primary", key="tornado_btn"):
                try:
                    with st.spinner("Recomputing the workflow for each input variation..."):
                        fig, table, base_disp, skipped = mc.generate_tornado_plot(
                            st.session_state.raw_df, column_map, target_canonical, variation_pct,
                            **workflow_settings,
                        )
                    st.session_state.tornado = {
                        "fig": fig, "table": table, "base": base_disp, "skipped": skipped,
                        "target_label": mc.display_name(target_canonical, unit_system),
                        "pct": variation_pct, "units": unit_system, "method": method_label,
                        "skipped_note": "Skipped (not mapped or could not be recomputed): ",
                    }
                except ValueError as exc:
                    st.session_state.tornado = None
                    st.error(f"⚠️ {exc}")
                except Exception as exc:  # keep the app alive on unexpected input
                    st.session_state.tornado = None
                    st.error(f"Unexpected error during sensitivity analysis: {exc}")

        # ----- Mode B: vary the governing-equation parameters of one result -----
        else:
            st.markdown(
                "Pick a computed result and perturb the parameters of **its own equation** one at a "
                "time by ±the selected percentage, holding the others at their base value. This is a "
                "*local* sensitivity — the target's equation is recomputed directly from the base-case "
                "profile, so it isolates how each governing input drives the output:"
            )
            st.markdown(
                "- **Shmin / SHmax** (poroelastic): Sv, Pp, Poisson's ratio, YME, Biot coefficient, "
                "tectonic strains EX & EY\n"
                "- **Mud weight – breakout**: Sv, Pp, Shmin, SHmax, UCS, friction angle, Poisson's ratio\n"
                "- **Mud weight – breakdown**: Shmin, SHmax, Pp, tensile strength"
            )
            if results is None or not has_stress:
                st.warning(
                    "This mode needs a completed **stress** run (Sv, Pp, Shmin, SHmax and wellbore "
                    "stability). Enable *Advanced stress modelling* in the sidebar and press "
                    "**Run / update model**, then come back here."
                )
            else:
                c1, c2 = st.columns(2)
                param_options = [mc.display_name(t, unit_system) for t in mc.TORNADO_PARAM_TARGETS]
                param_label_sel = c1.selectbox("Target Output", param_options, index=0,
                                               key="tornado_param_target",
                                               help="Result whose governing parameters are perturbed.")
                target_canonical = mc.TORNADO_PARAM_TARGETS[param_options.index(param_label_sel)]
                variation_pct = c2.select_slider("Variation Range", options=[5, 10, 20], value=10,
                                                 format_func=lambda v: f"±{v}%", key="tornado_param_pct")

                if st.button("🌪️ Generate Tornado Plot", type="primary", key="tornado_param_btn"):
                    try:
                        with st.spinner("Recomputing the governing equation for each parameter..."):
                            fig, table, base_disp, skipped = mc.generate_parameter_tornado_plot(
                                results, target_canonical, variation_pct,
                                unit_system=unit_system, stress_params=stress_params,
                                tstr_multiplier=tstr_multiplier,
                            )
                        st.session_state.tornado = {
                            "fig": fig, "table": table, "base": base_disp, "skipped": skipped,
                            "target_label": mc.display_name(target_canonical, unit_system),
                            "pct": variation_pct, "units": unit_system, "method": method_label,
                            "skipped_note": "Skipped (could not be recomputed): ",
                        }
                    except ValueError as exc:
                        st.session_state.tornado = None
                        st.error(f"⚠️ {exc}")
                    except Exception as exc:  # keep the app alive on unexpected input
                        st.session_state.tornado = None
                        st.error(f"Unexpected error during sensitivity analysis: {exc}")

        tornado = st.session_state.tornado
        if tornado is not None:
            st.metric(f"Base case — depth-averaged {tornado['target_label']}", f"{tornado['base']:.3f}")
            st.plotly_chart(tornado["fig"], use_container_width=True)
            st.markdown("**Per-parameter results** (sorted by impact):")
            st.dataframe(tornado["table"].style.format(precision=3), use_container_width=True, hide_index=True)
            if tornado["skipped"]:
                st.info(tornado.get("skipped_note", "Skipped: ") + ", ".join(tornado["skipped"]))
            st.caption(
                f"Generated with ±{tornado['pct']}% variation · {tornado['units']} · "
                f"static method: {tornado['method']}. Re-generate after changing data or settings."
            )

st.divider()
st.caption(
    "Quick MEM Calculator · built with [Streamlit](https://streamlit.io) + "
    "[geomechpy](https://github.com/0xsmolrun/GeomechPy_smolrun) · "
    "correlations: Bradford (1998), Najibi (2015), Fuller, Morales (1993), Plumb (1994), Lal (1999), "
    "Thiercelin & Plumb (1994), Hubbert & Willis (1957), Al-Ajmi & Zimmerman (2006)."
)
