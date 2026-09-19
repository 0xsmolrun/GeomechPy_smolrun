"""Stress Barrier Analysis & Perforation Planner - standalone single-file app.

A focused, one-page Streamlit tool on top of GeomechPy. Upload well log data
(CSV / Excel / LAS) with rock properties (Young's modulus, Poisson's ratio),
pore pressure and overburden, define the horizontal strains manually, and the
app will:

  1. flag lithology from a GR cutoff (reservoir sand vs non-reservoir shale),
  2. compute the horizontal stresses Shmin / SHmax with GeomechPy
     (poroelastic equation),
  3. analyse the stress contrast between reservoir and non-reservoir sections
     to locate stress barriers, and
  4. recommend perforation zones graded Good / Moderate / Poor.

This file bundles the calculation engine and the Streamlit front end into a
single module so it can be hosted directly:

    cd example/Project
    streamlit run Stress_Barrier_Planner.py

Canonical internal units: stresses in psi, YME in Mpsi (the poroelastic
equation's unit), PR/strain unitless, depth kept in the input unit.
"""
from __future__ import annotations

import io
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

from geomechpy.overburden_stress import OverburdenStressCalculation
from geomechpy.pore_pressure import PorePressureCalculation
from geomechpy.stress_calculations import HorizontalStressesCalculation


# ===========================================================================
# CALCULATION ENGINE
# ===========================================================================

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------

PSI_TO_MPA = 6894.757293e-6      # psi -> MPa
MPA_TO_PSI = 1.0 / PSI_TO_MPA    # MPa -> psi
PSI_TO_MPSI = 1.0e-6             # psi -> Mega-psi
MPSI_TO_PSI = 1.0e6              # Mega-psi -> psi
GPA_TO_PSI = 145037.737797      # GPa -> psi
GPA_TO_MPSI = GPA_TO_PSI * 1e-6  # GPa -> Mpsi (~0.145)
M_PER_FT = 0.3048                # metres per foot
M_TO_FT = 1.0 / M_PER_FT         # metres -> feet

# Common well-log null sentinels replaced with NaN on load.
NULL_SENTINELS = [-999.0, -999.25, -9999.0, -9999.25, -99999.0, 9999.0]

# ---------------------------------------------------------------------------
# Curves the app understands
# ---------------------------------------------------------------------------

# DEPTH is always required. YME + PR are required to compute horizontal
# stresses. GR is needed for the lithology flag. PP and SV may either come
# from a log column or be derived from a gradient (see StressConfig).
REQUIRED_CURVES = ["DEPTH", "YME", "PR"]
OPTIONAL_CURVES = ["GR", "PP", "SV"]
ALL_CURVES = REQUIRED_CURVES + OPTIONAL_CURVES

CURVE_LABELS = {
    "DEPTH": "Depth (MD)",
    "GR": "Gamma Ray (GR)",
    "PP": "Pore Pressure (PP)",
    "SV": "Overburden Stress (Sv / OVB)",
    "YME": "Young's Modulus (YME)",
    "PR": "Poisson's Ratio (PR)",
}

# Input-unit options offered in the UI, with the factor that converts a value
# in that unit to the canonical unit used internally.
YME_INPUT_UNITS = {
    "Mpsi": 1.0,                 # canonical
    "psi": PSI_TO_MPSI,
    "GPa": GPA_TO_MPSI,
}
PRESSURE_INPUT_UNITS = {
    "psi": 1.0,                  # canonical
    "MPa": MPA_TO_PSI,
    "bar": 14.5037738,
}
DEPTH_UNITS = ["m", "ft"]

# Lithology flag: reservoir sand (0) vs non-reservoir shale (1).
LITHO_NAME_BY_CODE = {0: "Reservoir (Sand)", 1: "Non-reservoir (Shale)"}
LITHO_COLORS = {0: "#f4d03f", 1: "#7f8c8d", -1: "#ecf0f1"}
DEFAULT_GR_CUTOFF = 75.0

# Perforation quality grades and their display colours.
PERF_QUALITIES = ["Good", "Moderate", "Poor"]
PERF_COLORS = {
    "Good": "#2ecc71",
    "Moderate": "#f39c12",
    "Poor": "#e74c3c",
    "N/A": "#ecf0f1",
}

SHMAX_METHODS = {
    "Poroelastic (Thiercelin & Plumb, 1994) [geomechpy]": "poroelastic",
    "Shmin × anisotropy multiplier [geomechpy]": "multiplier",
}
PP_SV_SOURCES = {
    "From log column": "column",
    "From gradient (geomechpy)": "gradient",
}


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def _drop_unit_rows(df: pd.DataFrame, max_rows: int = 3) -> tuple[pd.DataFrame, int]:
    """Drop leading rows that hold unit strings instead of data.

    A leading row is treated as a unit row when it is non-numeric in at least
    half of the columns whose remaining values are mostly numeric.
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
    """Read an uploaded CSV, Excel or LAS file and clean it up.

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
            df = las.df().reset_index()  # depth curve is the index -> expose as a column
        elif name.endswith((".csv", ".txt")):
            df = pd.read_csv(uploaded_file, skip_blank_lines=True)
            if df.shape[1] == 1:  # non-comma delimiter: re-sniff
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
    """Best-effort index of the column matching a curve mnemonic (selectbox default).

    Returns 0 ('-- not mapped --') when nothing matches.
    """
    aliases = {
        "DEPTH": ["depth", "dept", "md", "tvd"],
        "GR": ["gr", "gamma", "gapi", "cgr", "sgr"],
        "PP": ["pp", "pore", "porepressure", "pore_pressure", "ppg", "pnorm"],
        "SV": ["sv", "ovb", "obg", "overburden", "vertical", "sigv", "sigmav"],
        "YME": ["yme", "ym", "young", "youngs", "e_sta", "estat", "emod", "e"],
        "PR": ["pr", "poisson", "nu", "poissons", "pois"],
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

def generate_sample_data(n_points: int = 241, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic sand/shale well interval (3000-3600 m MD).

    Produces a ready-to-run dataset with the exact columns this app expects:
    MD, GR, PP, SV, YME, PR. Values are geologically plausible so the stress
    calculation and the barrier / perforation screening return sensible
    results out of the box. Shale sections are given a higher Poisson's ratio
    (and thus a higher Shmin) than the sand, creating clear stress barriers.
    """
    rng = np.random.default_rng(seed)
    depth = np.linspace(3000.0, 3600.0, n_points)

    # Smooth sand/shale alternation driver (0 = clean sand, 1 = shale).
    vsh = 0.5 + 0.38 * np.sin(depth / 22.0) + 0.12 * np.sin(depth / 70.0)
    vsh = np.clip(vsh + rng.normal(0, 0.05, n_points), 0.02, 0.98)

    tvd_ft = depth * M_TO_FT
    gr = 25.0 + 110.0 * vsh + rng.normal(0, 4.0, n_points)                 # gAPI
    pp = 0.465 * tvd_ft + rng.normal(0, 30.0, n_points)                    # psi (~hydrostatic)
    sv = 1.02 * tvd_ft + rng.normal(0, 40.0, n_points)                     # psi (lithostatic)
    # Shale is softer + more ductile: lower YME, higher PR -> higher Shmin.
    yme = (4.2 - 2.4 * vsh) + rng.normal(0, 0.15, n_points)                # Mpsi
    yme = np.clip(yme, 0.8, 6.0)
    pr = (0.20 + 0.15 * vsh) + rng.normal(0, 0.01, n_points)               # unitless
    pr = np.clip(pr, 0.12, 0.42)

    return pd.DataFrame(
        {
            "MD": np.round(depth, 2),
            "GR": np.round(gr, 2),
            "PP": np.round(pp, 1),
            "SV": np.round(sv, 1),
            "YME": np.round(yme, 3),
            "PR": np.round(pr, 3),
        }
    )


def sample_csv_bytes() -> bytes:
    """Example CSV for the 'Download Example File' button."""
    buffer = io.StringIO()
    generate_sample_data().to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Lithology flag (GR cutoff)
# ---------------------------------------------------------------------------

def compute_lithology_flag(df: pd.DataFrame, gr_cutoff: float = DEFAULT_GR_CUTOFF) -> pd.DataFrame:
    """Add a LITHO_CODE column classifying each sample from GR by one cutoff.

    GR <  gr_cutoff -> reservoir sand (code 0)
    GR >= gr_cutoff -> non-reservoir shale (code 1)
    Missing GR      -> NaN
    """
    out = df.copy()
    gr = (
        pd.to_numeric(out["GR"], errors="coerce").to_numpy(dtype=float)
        if "GR" in out.columns
        else np.full(len(out), np.nan)
    )
    out["LITHO_CODE"] = np.where(np.isfinite(gr), np.where(gr < float(gr_cutoff), 0.0, 1.0), np.nan)
    return out


def lithology_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-lithology sample counts and fraction, for the summary display."""
    if "LITHO_CODE" not in df.columns:
        return pd.DataFrame(columns=["Lithology", "Code", "Samples", "Fraction %"])
    codes = pd.to_numeric(df["LITHO_CODE"], errors="coerce")
    total = int(codes.notna().sum())
    rows = []
    for code, name in LITHO_NAME_BY_CODE.items():
        cnt = int((codes == code).sum())
        rows.append({"Lithology": name, "Code": code, "Samples": cnt,
                     "Fraction %": round(100.0 * cnt / total, 1) if total else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stress configuration & horizontal-stress calculation
# ---------------------------------------------------------------------------

def default_stress_config() -> dict:
    """Default configuration for run_stress_workflow (all user-adjustable)."""
    return {
        # input units
        "yme_unit": "Mpsi",
        "pressure_unit": "psi",
        "depth_unit": "m",
        # pore pressure / overburden source ('column' | 'gradient')
        "pp_source": "column",
        "sv_source": "column",
        "pp_gradient_psift": 0.465,
        "ovb_gradient_psift": 1.02,
        "setting": "Onshore",
        "air_gap": 0.0,
        # horizontal strains (manual) and poroelastic parameters
        "eps_h": 0.0001,     # minimum horizontal strain  -> EX in geomechpy
        "eps_H": 0.0009,     # maximum horizontal strain  -> EY in geomechpy
        "biot": 1.0,
        "shmax_method": "poroelastic",
        "shmax_multiplier": 1.1,
        "gr_cutoff": DEFAULT_GR_CUTOFF,
    }


def _pp_sv_from_gradient(tvd_ft: np.ndarray, gradient_psift: float, air_gap_ft: float,
                         kind: str) -> np.ndarray:
    """Compute a pressure profile from a gradient using geomechpy (onshore)."""
    n = len(tvd_ft)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(tvd_ft[i]) or tvd_ft[i] < 0:
            continue
        if kind == "sv":
            out[i] = OverburdenStressCalculation.calculate_overburden_stress_onshore(
                tvd=float(tvd_ft[i]), lithostatic_gradient=gradient_psift, air_gap=air_gap_ft,
            )
        else:  # pp
            out[i] = PorePressureCalculation.calculate_pore_pressure_onshore(
                tvd=float(tvd_ft[i]), formation_pore_pressure_gradient=gradient_psift, air_gap=air_gap_ft,
            )
    return out


def run_stress_workflow(data: pd.DataFrame, column_map: dict[str, str], config: dict) -> pd.DataFrame:
    """Build the canonical results frame: horizontal stresses + lithology flag.

    Steps:
      1. Rename mapped columns to standard mnemonics; convert YME to Mpsi and
         PP/SV to psi (or derive PP/SV from a gradient via geomechpy).
      2. Flag lithology from GR (reservoir sand vs non-reservoir shale).
      3. Compute Shmin/SHmax per depth with the geomechpy poroelastic equation
         using the user's YME, PR, Sv, Pp and the manually-defined horizontal
         strains (eps_h -> EX, eps_H -> EY). SHmax can instead be Shmin × a
         multiplier. q-factor and the SHmax/Shmin ratio come from the library.

    Canonical outputs (all stresses in psi): SV_PSI, PP_PSI, SHMIN_PSI,
    SHMAX_PSI, Q_FACTOR, SH_RATIO, plus DEPTH, GR, YME_MPSI, PR, LITHO_CODE.
    Rows with missing prerequisites yield NaN instead of raising.
    """
    cfg = {**default_stress_config(), **(config or {})}

    missing = [c for c in REQUIRED_CURVES if not column_map.get(c)]
    if missing:
        raise ValueError(
            "Missing required column mapping(s): "
            + ", ".join(f"{c} ({CURVE_LABELS[c]})" for c in missing)
        )

    rename = {src: curve for curve, src in column_map.items() if src}
    df = data[list(rename)].rename(columns=rename).copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin(NULL_SENTINELS), col] = np.nan
    df = df.sort_values("DEPTH").reset_index(drop=True)
    n = len(df)

    # --- unit conversion to canonical ---
    yme_factor = YME_INPUT_UNITS.get(cfg["yme_unit"], 1.0)
    p_factor = PRESSURE_INPUT_UNITS.get(cfg["pressure_unit"], 1.0)

    out = pd.DataFrame(index=df.index)
    out["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")
    out["GR"] = pd.to_numeric(df["GR"], errors="coerce") if "GR" in df.columns else np.nan
    out["YME_MPSI"] = pd.to_numeric(df["YME"], errors="coerce") * yme_factor
    out["PR"] = pd.to_numeric(df["PR"], errors="coerce")

    depth = out["DEPTH"].to_numpy(dtype=float)
    tvd_ft = depth * M_TO_FT if cfg["depth_unit"] == "m" else depth.copy()
    air_gap_ft = float(cfg["air_gap"]) * (M_TO_FT if cfg["depth_unit"] == "m" else 1.0)

    # --- overburden (Sv) ---
    if cfg["sv_source"] == "gradient" or "SV" not in df.columns or df["SV"].isna().all():
        sv_psi = _pp_sv_from_gradient(tvd_ft, float(cfg["ovb_gradient_psift"]), air_gap_ft, "sv")
    else:
        sv_psi = (pd.to_numeric(df["SV"], errors="coerce") * p_factor).to_numpy(dtype=float)

    # --- pore pressure (Pp) ---
    if cfg["pp_source"] == "gradient" or "PP" not in df.columns or df["PP"].isna().all():
        pp_psi = _pp_sv_from_gradient(tvd_ft, float(cfg["pp_gradient_psift"]), air_gap_ft, "pp")
    else:
        pp_psi = (pd.to_numeric(df["PP"], errors="coerce") * p_factor).to_numpy(dtype=float)

    out["SV_PSI"] = sv_psi
    out["PP_PSI"] = pp_psi

    # --- lithology flag ---
    out = compute_lithology_flag(out, cfg["gr_cutoff"])

    # --- horizontal stresses (poroelastic, per sample) ---
    pr = out["PR"].to_numpy(dtype=float)
    yme = out["YME_MPSI"].to_numpy(dtype=float)
    shmin = np.full(n, np.nan)
    shmax = np.full(n, np.nan)
    q_factor = np.full(n, np.nan)
    sh_ratio = np.full(n, np.nan)

    for i in range(n):
        if not (np.isfinite(sv_psi[i]) and np.isfinite(pp_psi[i]) and np.isfinite(pr[i]) and np.isfinite(yme[i])):
            continue
        if not (0.0 < pr[i] < 0.5) or yme[i] <= 0:
            continue
        try:
            hs = HorizontalStressesCalculation.calculate_poroelastic_horizontal_stresses(
                overburden_stress=float(sv_psi[i]),
                pore_pressure=float(pp_psi[i]),
                poisson_ratio=float(pr[i]),
                youngs_modulus=float(yme[i]),
                biot_coefficient=float(cfg["biot"]),
                EX=float(cfg["eps_h"]),   # minimum horizontal strain
                EY=float(cfg["eps_H"]),   # maximum horizontal strain
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        shmin[i] = hs.shmin
        if cfg["shmax_method"] == "multiplier":
            shmax[i] = HorizontalStressesCalculation.calculate_shmax_multiplier(
                shmin=float(hs.shmin), shmax_multiplier=float(cfg["shmax_multiplier"])
            )
        else:
            shmax[i] = hs.shmax
        try:
            q_factor[i] = HorizontalStressesCalculation.calculate_stress_regime_q_factor(
                sigv=float(sv_psi[i]), shmax=float(shmax[i]), shmin=float(shmin[i])
            )
            sh_ratio[i] = HorizontalStressesCalculation.calculate_horizontal_stress_ratio(
                shmax=float(shmax[i]), shmin=float(shmin[i])
            )
        except (ValueError, ZeroDivisionError):
            pass

    out["SHMIN_PSI"] = shmin
    out["SHMAX_PSI"] = shmax
    out["Q_FACTOR"] = q_factor
    out["SH_RATIO"] = sh_ratio
    return out


# ---------------------------------------------------------------------------
# Stress barrier analysis & perforation zone screening
# ---------------------------------------------------------------------------

def _lithology_runs(codes: np.ndarray) -> list[tuple[int, int, float]]:
    """Contiguous runs of equal lithology code -> list of (start, end, code)."""
    n = len(codes)
    runs: list[tuple[int, int, float]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (
            codes[j + 1] == codes[i] or (np.isnan(codes[j + 1]) and np.isnan(codes[i]))
        ):
            j += 1
        runs.append((i, j, codes[i]))
        i = j + 1
    return runs


def analyze_stress_barriers(
    results: pd.DataFrame,
    contrast_threshold_psi: float = 300.0,
    trend_window: int = 25,
    min_zone_thickness: float = 5.0,
) -> dict:
    """Stress-barrier analysis and perforation-zone screening.

    Rationale: a hydraulic fracture placed in a reservoir (sand) stays
    contained when it is bounded by higher-stress non-reservoir (shale)
    intervals — the stress barriers. This routine therefore:

      1. Computes a per-sample **stress contrast** = Shmin minus its depth
         trend (centred rolling median over trend_window samples), used as the
         contrast curve on the plot.
      2. Splits the log into contiguous lithology intervals and, for every
         reservoir interval, measures the Shmin contrast against the
         immediately adjacent non-reservoir intervals above and below:
             contrast = mean Shmin(adjacent shale) − mean Shmin(reservoir).
         A side is a **barrier** when that contrast ≥ contrast_threshold_psi.
      3. Grades each reservoir interval:
             Good     — a barrier above AND below (fully contained),
             Moderate — a barrier on one side only,
             Poor     — no adequate barrier (or interval too thin).
      4. Assigns the interval grade to every sample it contains (PERF_QUALITY)
         and returns the recommended perforation zones and barrier intervals.

    All stresses are psi. Returns dict with:
        detail   : per-sample DataFrame (DEPTH, LITHO_CODE, SHMIN_PSI,
                   TREND_PSI, CONTRAST_PSI, PERF_QUALITY).
        zones    : reservoir intervals graded Good/Moderate/Poor.
        barriers : non-reservoir intervals acting as a barrier to a neighbour.
    """
    for col in ("SHMIN_PSI", "LITHO_CODE", "DEPTH"):
        if col not in results.columns:
            raise ValueError(
                "Barrier analysis needs the stress profile and lithology flag — "
                "run the stress calculation with GR mapped first."
            )

    depth = pd.to_numeric(results["DEPTH"], errors="coerce")
    shmin = pd.to_numeric(results["SHMIN_PSI"], errors="coerce")
    codes = pd.to_numeric(results["LITHO_CODE"], errors="coerce").to_numpy(dtype=float)
    n = len(results)

    if int(shmin.notna().sum()) < 5:
        raise ValueError("Not enough valid Shmin samples for a barrier analysis — check the inputs.")

    # 1. Per-sample stress contrast vs the local Shmin trend.
    trend = shmin.rolling(int(trend_window), center=True, min_periods=1).median()
    contrast = shmin - trend

    # 2-3. Interval-based reservoir vs adjacent-shale contrast + grading.
    runs = _lithology_runs(codes)
    mean_shmin = [float(shmin.iloc[s:e + 1].mean()) for (s, e, _c) in runs]

    quality = np.full(n, "N/A", dtype=object)
    t = float(contrast_threshold_psi)
    zone_rows: list[dict] = []
    barrier_pairs: set[int] = set()  # indices (into runs) of shale acting as a barrier

    for k, (s, e, code) in enumerate(runs):
        if code != 0.0:  # only reservoir sand intervals are perforation candidates
            continue
        thickness = float(depth.iloc[e] - depth.iloc[s])
        res_shmin = mean_shmin[k]

        # nearest shale interval above / below (previous / next run that is shale)
        above_k = k - 1 if k - 1 >= 0 and runs[k - 1][2] == 1.0 else None
        below_k = k + 1 if k + 1 < len(runs) and runs[k + 1][2] == 1.0 else None

        c_above = (mean_shmin[above_k] - res_shmin) if above_k is not None else np.nan
        c_below = (mean_shmin[below_k] - res_shmin) if below_k is not None else np.nan
        barrier_above = np.isfinite(c_above) and c_above >= t
        barrier_below = np.isfinite(c_below) and c_below >= t

        if thickness < float(min_zone_thickness) or not np.isfinite(res_shmin):
            grade = "Poor"
        elif barrier_above and barrier_below:
            grade = "Good"
        elif barrier_above or barrier_below:
            grade = "Moderate"
        else:
            grade = "Poor"

        quality[s:e + 1] = grade
        if barrier_above and above_k is not None:
            barrier_pairs.add(above_k)
        if barrier_below and below_k is not None:
            barrier_pairs.add(below_k)

        zone_rows.append(
            {
                "Top": float(depth.iloc[s]),
                "Base": float(depth.iloc[e]),
                "Thickness": round(thickness, 2),
                "Samples": e - s + 1,
                "Mean Shmin (psi)": round(res_shmin, 1) if np.isfinite(res_shmin) else np.nan,
                "Contrast above (psi)": round(c_above, 1) if np.isfinite(c_above) else np.nan,
                "Contrast below (psi)": round(c_below, 1) if np.isfinite(c_below) else np.nan,
                "Barrier above": "Yes" if barrier_above else "No",
                "Barrier below": "Yes" if barrier_below else "No",
                "Quality": grade,
            }
        )

    detail = pd.DataFrame(
        {
            "DEPTH": depth,
            "LITHO_CODE": codes,
            "SHMIN_PSI": shmin,
            "TREND_PSI": trend,
            "CONTRAST_PSI": contrast,
            "PERF_QUALITY": quality,
        }
    )

    zone_cols = ["Top", "Base", "Thickness", "Samples", "Mean Shmin (psi)",
                 "Contrast above (psi)", "Contrast below (psi)",
                 "Barrier above", "Barrier below", "Quality"]
    zones = (
        pd.DataFrame(zone_rows, columns=zone_cols).sort_values("Top").reset_index(drop=True)
        if zone_rows else pd.DataFrame(columns=zone_cols)
    )

    barrier_rows = [
        {
            "Top": float(depth.iloc[runs[k][0]]),
            "Base": float(depth.iloc[runs[k][1]]),
            "Thickness": round(float(depth.iloc[runs[k][1]] - depth.iloc[runs[k][0]]), 2),
            "Mean Shmin (psi)": round(mean_shmin[k], 1) if np.isfinite(mean_shmin[k]) else np.nan,
        }
        for k in sorted(barrier_pairs)
    ]
    barriers = (
        pd.DataFrame(barrier_rows).sort_values("Top").reset_index(drop=True)
        if barrier_rows else pd.DataFrame(columns=["Top", "Base", "Thickness", "Mean Shmin (psi)"])
    )

    return {"detail": detail, "zones": zones, "barriers": barriers}


# ---------------------------------------------------------------------------
# Display helpers & export
# ---------------------------------------------------------------------------

# Canonical column -> friendly display header (all stresses shown in psi).
DISPLAY_NAMES = {
    "DEPTH": "MD",
    "LITHO_CODE": "LITHO",
    "GR": "GR [gAPI]",
    "YME_MPSI": "YME [Mpsi]",
    "PR": "PR [-]",
    "SV_PSI": "Sv [psi]",
    "PP_PSI": "Pp [psi]",
    "SHMIN_PSI": "Shmin [psi]",
    "SHMAX_PSI": "SHmax [psi]",
    "Q_FACTOR": "q-factor [-]",
    "SH_RATIO": "SHmax/Shmin [-]",
    "TREND_PSI": "Shmin trend [psi]",
    "CONTRAST_PSI": "Stress contrast [psi]",
    "PERF_QUALITY": "Perf quality",
}


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename canonical columns to friendly headers for on-screen tables."""
    return df.rename(columns={c: DISPLAY_NAMES.get(c, c) for c in df.columns})


def lithology_label_column(df: pd.DataFrame) -> pd.Series:
    """Human-readable lithology names for a LITHO_CODE column."""
    codes = pd.to_numeric(df.get("LITHO_CODE"), errors="coerce")
    return codes.map(LITHO_NAME_BY_CODE).fillna("Undefined")


def results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a results frame for a Streamlit download button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, float_format="%.4f")
    return buffer.getvalue().encode("utf-8")


# Front-end code below uses the `sb.` namespace for the engine above. Snapshot
# the engine's public globals into a namespace so this works regardless of how
# the file is launched (streamlit run, python -m, importlib, ...).
import types as _types

sb = _types.SimpleNamespace(**{_k: _v for _k, _v in dict(globals()).items()
                               if not _k.startswith("_")})


# ===========================================================================
# STREAMLIT FRONT END
# ===========================================================================

st.set_page_config(
    page_title="Stress Barrier Analysis & Perforation Planner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in {
    "raw_df": None,        # uploaded / sample input data
    "results_df": None,    # stress workflow output (canonical units)
    "analysis": None,      # barrier / perforation analysis output
    "data_source": None,   # label shown in the sidebar
    "load_messages": [],   # info messages from the data cleaner
}.items():
    st.session_state.setdefault(key, default)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _add_lithology_column(fig: go.Figure, depth_series, code_series, col: int, row: int = 1) -> None:
    """Draw the lithology flag as a colored track (contiguous runs by code)."""
    depth = pd.to_numeric(depth_series, errors="coerce").to_numpy(dtype=float)
    codes = pd.to_numeric(code_series, errors="coerce").to_numpy(dtype=float)
    filled = np.where(np.isfinite(codes), codes, -1.0)
    n = len(filled)
    present: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and filled[j + 1] == filled[i]:
            j += 1
        c = int(filled[i])
        y0 = depth[i]
        y1 = depth[j + 1] if (j + 1) < n else depth[j]
        fig.add_shape(type="rect", x0=0.0, x1=1.0, y0=y0, y1=y1,
                      fillcolor=sb.LITHO_COLORS.get(c, "#ecf0f1"), line_width=0,
                      layer="below", row=row, col=col)
        present.append(c)
        i = j + 1
    for c in sorted(set(present)):
        name = sb.LITHO_NAME_BY_CODE.get(c, "Undefined")
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=11, color=sb.LITHO_COLORS.get(c, "#ecf0f1")),
                       name=name, legendgroup="litho"),
            row=row, col=col,
        )
    fig.update_xaxes(visible=False, range=[0.0, 1.0], row=row, col=col)


def _add_quality_column(fig: go.Figure, depth_series, quality_series, col: int, row: int = 1) -> None:
    """Draw the perforation-quality flag as a colored track (contiguous runs)."""
    depth = pd.to_numeric(depth_series, errors="coerce").to_numpy(dtype=float)
    qual = quality_series.astype(object).to_numpy()
    n = len(qual)
    present: list[str] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and qual[j + 1] == qual[i]:
            j += 1
        q = str(qual[i])
        y0 = depth[i]
        y1 = depth[j + 1] if (j + 1) < n else depth[j]
        fig.add_shape(type="rect", x0=0.0, x1=1.0, y0=y0, y1=y1,
                      fillcolor=sb.PERF_COLORS.get(q, "#ecf0f1"), line_width=0,
                      layer="below", row=row, col=col)
        present.append(q)
        i = j + 1
    for q in [g for g in sb.PERF_QUALITIES if g in set(present)]:
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=11, color=sb.PERF_COLORS.get(q, "#ecf0f1")),
                       name=f"{q} perf", legendgroup="perf"),
            row=row, col=col,
        )
    fig.update_xaxes(visible=False, range=[0.0, 1.0], row=row, col=col)


def composite_figure(results: pd.DataFrame, detail: pd.DataFrame, zones: pd.DataFrame,
                     height: int = 820) -> go.Figure:
    """Multi-track log plot: Litho | GR | stresses | stress contrast | perf quality."""
    depth = results["DEPTH"]
    titles = ["Litho", "GR (gAPI)", "Stresses (psi)", "Stress contrast (psi)", "Perf"]
    widths_raw = [0.6, 1.6, 2.6, 2.0, 0.6]
    total = sum(widths_raw)
    fig = make_subplots(
        rows=1, cols=5, shared_yaxes=True, horizontal_spacing=0.02,
        subplot_titles=titles, column_widths=[w / total for w in widths_raw],
    )

    # Col 1 — lithology flag
    _add_lithology_column(fig, depth, results["LITHO_CODE"], col=1)

    # Col 2 — GR with the cutoff for reference
    if "GR" in results.columns and results["GR"].notna().any():
        fig.add_trace(
            go.Scatter(x=results["GR"], y=depth, mode="lines", name="GR",
                       line=dict(color="#2c3e50"),
                       hovertemplate="GR: %{x:.1f}<br>Depth: %{y:.1f}<extra></extra>"),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="gAPI", row=1, col=2)

    # Col 3 — Pp, Sv, Shmin, SHmax
    stress_curves = [
        ("SV_PSI", "Sv", "#8e44ad"),
        ("SHMAX_PSI", "SHmax", "#c0392b"),
        ("SHMIN_PSI", "Shmin", "#2980b9"),
        ("PP_PSI", "Pp", "#7f8c8d"),
    ]
    for canonical, name, color in stress_curves:
        if canonical in results.columns:
            fig.add_trace(
                go.Scatter(x=results[canonical], y=depth, mode="lines", name=name,
                           line=dict(color=color),
                           hovertemplate=f"{name}: %{{x:.0f}} psi<br>Depth: %{{y:.1f}}<extra></extra>"),
                row=1, col=3,
            )
    # Shade the recommended perforation zones across the stress track.
    for _, z in zones.iterrows():
        fig.add_shape(type="rect", xref="x domain", x0=0.0, x1=1.0,
                      y0=z["Top"], y1=z["Base"],
                      fillcolor=sb.PERF_COLORS.get(z["Quality"], "#ecf0f1"),
                      opacity=0.18, line_width=0, layer="below", row=1, col=3)
    fig.update_xaxes(title_text="psi", row=1, col=3)

    # Col 4 — stress contrast (Shmin minus trend), zero reference line
    fig.add_trace(
        go.Scatter(x=detail["CONTRAST_PSI"], y=detail["DEPTH"], mode="lines",
                   name="Contrast", line=dict(color="#16a085"),
                   hovertemplate="Contrast: %{x:.0f} psi<br>Depth: %{y:.1f}<extra></extra>"),
        row=1, col=4,
    )
    fig.add_vline(x=0.0, line=dict(color="gray", dash="dot"), row=1, col=4)
    fig.update_xaxes(title_text="psi", row=1, col=4)

    # Col 5 — perforation quality flag
    _add_quality_column(fig, detail["DEPTH"], detail["PERF_QUALITY"], col=5)

    fig.update_yaxes(autorange="reversed", title_text="MD", col=1)
    fig.update_layout(
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.06),
        margin=dict(t=110, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar — inputs & parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎯 Perforation Planner")
    st.caption(
        "Stress barrier analysis & perforation planning powered by "
        "[geomechpy](https://github.com/0xsmolrun/GeomechPy_smolrun)."
    )

    with st.expander("1. Data input", expanded=True):
        uploaded = st.file_uploader(
            "Upload well log data (CSV / Excel / LAS)",
            type=["csv", "txt", "xls", "xlsx", "las"],
            help="One row per depth sample. Expected curves: DEPTH, GR, PP (pore "
            "pressure), SV/OVB (overburden), YME (Young's modulus), PR (Poisson's "
            "ratio). Unit rows and -999.25/-9999 nulls are handled automatically.",
        )
        if uploaded is not None:
            try:
                st.session_state.raw_df, st.session_state.load_messages = sb.load_data(uploaded)
                st.session_state.data_source = f"📄 {uploaded.name}"
                st.session_state.results_df = None
                st.session_state.analysis = None
            except ValueError as exc:
                st.error(str(exc))

        c_sample, c_dl = st.columns(2)
        if c_sample.button("🧪 Load Sample", use_container_width=True):
            st.session_state.raw_df = sb.generate_sample_data()
            st.session_state.data_source = "🧪 Synthetic sand/shale well (3000–3600 m)"
            st.session_state.load_messages = []
            st.session_state.results_df = None
            st.session_state.analysis = None
        c_dl.download_button(
            "⬇️ Template", data=sb.sample_csv_bytes(),
            file_name="perforation_planner_example.csv", mime="text/csv",
            use_container_width=True, help="Correctly formatted example CSV.",
        )

        if st.session_state.data_source:
            st.success(f"Loaded: {st.session_state.data_source}")
        for msg in st.session_state.load_messages:
            st.info(msg)

    with st.expander("2. Column mapping", expanded=True):
        column_map: dict[str, str] = {}
        if st.session_state.raw_df is not None:
            cols = list(st.session_state.raw_df.columns)
            options = ["-- not mapped --"] + cols
            for curve in sb.ALL_CURVES:
                required = curve in sb.REQUIRED_CURVES
                label = f"{sb.CURVE_LABELS[curve]} {'(required)' if required else '(optional)'}"
                choice = st.selectbox(
                    label, options, index=sb.guess_column(curve, cols), key=f"map_{curve}",
                )
                column_map[curve] = "" if choice == "-- not mapped --" else choice
        else:
            st.info("Load data first to map columns.")

    with st.expander("3. Input units", expanded=False):
        yme_unit = st.selectbox("Young's modulus unit", list(sb.YME_INPUT_UNITS.keys()), index=0,
                                help="Converted internally to Mpsi for the poroelastic equation.")
        pressure_unit = st.selectbox("Pressure unit (PP / Sv)", list(sb.PRESSURE_INPUT_UNITS.keys()),
                                     index=0, help="Applies to the PP and Sv log columns. Output is always psi.")
        depth_unit = st.radio("Depth (MD) unit", sb.DEPTH_UNITS, index=0, horizontal=True,
                              help="Used to convert MD to TVD (ft) for gradient-based Pp/Sv.")

    with st.expander("4. Pore pressure & overburden source", expanded=False):
        pp_source = sb.PP_SV_SOURCES[st.selectbox(
            "Pore pressure (PP)", list(sb.PP_SV_SOURCES.keys()), index=0,
            help="Use the mapped PP column, or derive Pp from a gradient with geomechpy.",
        )]
        pp_gradient = st.number_input("Pp gradient (psi/ft)", value=0.465, min_value=0.30,
                                      max_value=1.10, step=0.005, format="%.3f",
                                      disabled=(pp_source == "column"))
        sv_source = sb.PP_SV_SOURCES[st.selectbox(
            "Overburden (Sv)", list(sb.PP_SV_SOURCES.keys()), index=0,
            help="Use the mapped Sv/OVB column, or derive Sv from a lithostatic gradient with geomechpy.",
        )]
        ovb_gradient = st.number_input("Sv gradient (psi/ft)", value=1.02, min_value=0.60,
                                       max_value=1.30, step=0.005, format="%.3f",
                                       disabled=(sv_source == "column"))
        air_gap = st.number_input(f"Air gap / KB ({depth_unit})", value=0.0, min_value=0.0,
                                  format="%.1f", help="Only used when Pp/Sv are derived from a gradient.")

    with st.expander("5. Horizontal strains & stress model", expanded=True):
        st.caption("Manually define the horizontal strains used by the poroelastic equation.")
        eps_h = st.number_input("Minimum horizontal strain εh", value=0.0001, step=0.0001,
                                format="%.5f", help="Strain in the Shmin direction (EX in geomechpy).")
        eps_H = st.number_input("Maximum horizontal strain εH", value=0.0009, step=0.0001,
                                format="%.5f", help="Strain in the SHmax direction (EY). Keep εH ≥ εh.")
        if eps_H < eps_h:
            st.warning("εH < εh — SHmax may fall below Shmin. Set εH ≥ εh for a physical result.")
        biot = st.slider("Biot coefficient", 0.5, 1.0, 1.0, 0.05)
        shmax_method = sb.SHMAX_METHODS[st.selectbox("SHmax method", list(sb.SHMAX_METHODS.keys()), index=0)]
        shmax_multiplier = st.slider("SHmax / Shmin multiplier", 1.0, 2.0, 1.1, 0.05,
                                     disabled=(shmax_method != "multiplier"))

    with st.expander("6. Lithology (GR cutoff)", expanded=True):
        gr_cutoff = st.slider("GR cutoff (gAPI)", 0.0, 200.0, sb.DEFAULT_GR_CUTOFF, 1.0,
                              help="GR < cutoff → reservoir sand (0); GR ≥ cutoff → non-reservoir shale (1).")

    with st.expander("7. Barrier & perforation screening", expanded=True):
        contrast_threshold = st.slider(
            "Stress contrast threshold (psi)", 50.0, 2000.0, 300.0, 50.0,
            help="Minimum Shmin increase in the adjacent shale for it to count as a stress barrier.",
        )
        min_zone_thickness = st.number_input(
            f"Minimum reservoir thickness ({depth_unit})", value=5.0, min_value=0.0, step=1.0,
            format="%.1f", help="Reservoir intervals thinner than this are graded Poor.",
        )
        trend_window = st.slider("Contrast trend window (samples)", 5, 75, 25, 5,
                                 help="Window for the rolling-median Shmin trend used by the contrast curve.")

    st.divider()
    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True,
                            disabled=st.session_state.raw_df is None)

# Bundle the configuration once.
stress_config = dict(
    yme_unit=yme_unit if st.session_state.raw_df is not None else "Mpsi",
    pressure_unit=pressure_unit if st.session_state.raw_df is not None else "psi",
    depth_unit=depth_unit if st.session_state.raw_df is not None else "m",
    pp_source=pp_source if st.session_state.raw_df is not None else "column",
    sv_source=sv_source if st.session_state.raw_df is not None else "column",
    pp_gradient_psift=pp_gradient if st.session_state.raw_df is not None else 0.465,
    ovb_gradient_psift=ovb_gradient if st.session_state.raw_df is not None else 1.02,
    air_gap=air_gap if st.session_state.raw_df is not None else 0.0,
    eps_h=eps_h if st.session_state.raw_df is not None else 0.0001,
    eps_H=eps_H if st.session_state.raw_df is not None else 0.0009,
    biot=biot if st.session_state.raw_df is not None else 1.0,
    shmax_method=shmax_method if st.session_state.raw_df is not None else "poroelastic",
    shmax_multiplier=shmax_multiplier if st.session_state.raw_df is not None else 1.1,
    gr_cutoff=gr_cutoff if st.session_state.raw_df is not None else sb.DEFAULT_GR_CUTOFF,
)

# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------

if run_clicked:
    try:
        with st.spinner("Computing horizontal stresses and screening perforation zones..."):
            results = sb.run_stress_workflow(st.session_state.raw_df, column_map, stress_config)
            analysis = sb.analyze_stress_barriers(
                results,
                contrast_threshold_psi=contrast_threshold,
                trend_window=int(trend_window),
                min_zone_thickness=float(min_zone_thickness),
            )
        st.session_state.results_df = results
        st.session_state.analysis = analysis
        st.toast("Analysis complete ✅")
    except ValueError as exc:
        st.error(f"⚠️ {exc}")
    except Exception as exc:  # keep the app alive on unexpected input
        st.error(f"Unexpected error during analysis: {exc}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Stress Barrier Analysis & Perforation Planner")
st.markdown(
    "Compute horizontal stresses from rock properties with **geomechpy**, then locate "
    "**stress barriers** and recommend **perforation zones**. Configure inputs in the "
    "sidebar and click **🚀 Run Analysis**."
)

results = st.session_state.results_df
analysis = st.session_state.analysis

tab_input, tab_litho, tab_stress, tab_barrier, tab_perf = st.tabs(
    ["📥 Data Input", "🪨 Lithology", "↔️ Horizontal Stress", "🧱 Stress Barriers", "🎯 Perforation Zones"]
)

# --- Tab 1: Data input ------------------------------------------------------
with tab_input:
    st.subheader("Data input")
    ref = pd.DataFrame(
        [
            {"Curve": "DEPTH", "Description": sb.CURVE_LABELS["DEPTH"], "Requirement": "Required", "Example": "3000.0"},
            {"Curve": "YME", "Description": sb.CURVE_LABELS["YME"], "Requirement": "Required", "Example": "3.5 (Mpsi)"},
            {"Curve": "PR", "Description": sb.CURVE_LABELS["PR"], "Requirement": "Required", "Example": "0.25"},
            {"Curve": "GR", "Description": sb.CURVE_LABELS["GR"], "Requirement": "Optional (needed for lithology)", "Example": "75.0"},
            {"Curve": "PP", "Description": sb.CURVE_LABELS["PP"], "Requirement": "Optional (or from gradient)", "Example": "4600 (psi)"},
            {"Curve": "SV", "Description": sb.CURVE_LABELS["SV"], "Requirement": "Optional (or from gradient)", "Example": "10000 (psi)"},
        ]
    )
    st.markdown(
        "**Expected input columns** — one row per depth sample. Column *names* can be anything; "
        "map them to these curves in the sidebar (**2. Column mapping**). PP and Sv can either "
        "come from a log column or be derived from a gradient (**4. Pore pressure & overburden source**)."
    )
    st.dataframe(ref, use_container_width=True, hide_index=True)
    st.caption(
        "Tip: use **⬇️ Template** in the sidebar for a correctly formatted CSV, or **🧪 Load Sample** "
        "to try the app immediately. Horizontal strains εh / εH are defined manually in the sidebar."
    )

    if st.session_state.raw_df is None:
        st.info("👈 Upload a file or click **Load Sample** in the sidebar to get started.")
    else:
        df_in = st.session_state.raw_df
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_in):,}")
        c2.metric("Columns", df_in.shape[1])
        depth_col = column_map.get("DEPTH") if column_map else None
        if depth_col and depth_col in df_in.columns:
            c3.metric("Depth range", f"{df_in[depth_col].min():.0f} – {df_in[depth_col].max():.0f} {depth_unit}")
        st.subheader("Uploaded data preview")
        st.dataframe(df_in, use_container_width=True, height=360)
        with st.expander("Basic statistics"):
            st.dataframe(df_in.describe().T, use_container_width=True)

# --- Tab 2: Lithology -------------------------------------------------------
with tab_litho:
    st.subheader("Lithology flag (from GR)")
    st.markdown(
        f"Each depth is flagged from **GR** using the cutoff **{gr_cutoff:g} gAPI**: "
        "GR below the cutoff = **reservoir sand (0)**, at or above = **non-reservoir shale (1)**."
    )
    if results is None:
        st.info("Run the analysis from the sidebar to generate the lithology flag.")
    elif not results["LITHO_CODE"].notna().any():
        st.warning("No lithology flag was produced — map a **GR** column in the sidebar and re-run.")
    else:
        counts = sb.lithology_counts(results)
        cols = st.columns(len(sb.LITHO_NAME_BY_CODE))
        for col, (code, name) in zip(cols, sb.LITHO_NAME_BY_CODE.items()):
            row = counts[counts["Code"] == code]
            pct = float(row["Fraction %"].iloc[0]) if not row.empty else 0.0
            col.metric(name, f"{pct:.1f}%")
        st.dataframe(counts, use_container_width=True, hide_index=True)

        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=["Litho", "GR (gAPI)"], column_widths=[0.25, 0.75])
        _add_lithology_column(fig, results["DEPTH"], results["LITHO_CODE"], col=1)
        fig.add_trace(go.Scatter(x=results["GR"], y=results["DEPTH"], mode="lines", name="GR",
                                 line=dict(color="#2c3e50")), row=1, col=2)
        fig.add_vline(x=gr_cutoff, line=dict(color="#e74c3c", dash="dash"), row=1, col=2)
        fig.update_yaxes(autorange="reversed", title_text="MD", col=1)
        fig.update_xaxes(title_text="gAPI", row=1, col=2)
        fig.update_layout(height=620, legend=dict(orientation="h", yanchor="bottom", y=1.05),
                          margin=dict(t=90, b=40))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Horizontal stress -----------------------------------------------
with tab_stress:
    st.subheader("Horizontal stresses (psi)")
    if results is None:
        st.info("Run the analysis from the sidebar to compute horizontal stresses.")
    elif not results["SHMIN_PSI"].notna().any():
        st.warning(
            "No horizontal stresses were produced. Check that YME, PR, PP and Sv are mapped "
            "(or Pp/Sv gradients are enabled) and that PR is between 0 and 0.5."
        )
    else:
        method_txt = [k for k, v in sb.SHMAX_METHODS.items() if v == shmax_method]
        st.caption(
            f"Poroelastic equation (Thiercelin & Plumb, 1994) · εh {eps_h:g} · εH {eps_H:g} · "
            f"Biot {biot:.2f} · SHmax: {method_txt[0] if method_txt else '-'}"
            + (f" (×{shmax_multiplier:.2f})" if shmax_method == "multiplier" else "")
        )
        q_med = pd.to_numeric(results["Q_FACTOR"], errors="coerce").median()
        c1, c2, c3 = st.columns(3)
        c1.metric("Median Shmin (psi)", f"{results['SHMIN_PSI'].median():,.0f}")
        c2.metric("Median SHmax (psi)", f"{results['SHMAX_PSI'].median():,.0f}")
        if pd.notna(q_med):
            regime = "Normal" if q_med < 1 else ("Strike-slip" if q_med < 2 else "Reverse")
            c3.metric("Stress regime", regime, help=f"median q-factor = {q_med:.2f}")

        show = sb.display_frame(results[["DEPTH", "GR", "YME_MPSI", "PR", "SV_PSI", "PP_PSI",
                                         "SHMIN_PSI", "SHMAX_PSI", "Q_FACTOR", "SH_RATIO"]])
        st.dataframe(show.style.format(precision=2), use_container_width=True, height=340)

        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=["Litho", "Stresses (psi)"], column_widths=[0.15, 0.85])
        _add_lithology_column(fig, results["DEPTH"], results["LITHO_CODE"], col=1)
        for canonical, name, color in [("SV_PSI", "Sv", "#8e44ad"), ("SHMAX_PSI", "SHmax", "#c0392b"),
                                       ("SHMIN_PSI", "Shmin", "#2980b9"), ("PP_PSI", "Pp", "#7f8c8d")]:
            fig.add_trace(go.Scatter(x=results[canonical], y=results["DEPTH"], mode="lines",
                                     name=name, line=dict(color=color)), row=1, col=2)
        fig.update_yaxes(autorange="reversed", title_text="MD", col=1)
        fig.update_xaxes(title_text="psi", row=1, col=2)
        fig.update_layout(height=650, legend=dict(orientation="h", yanchor="bottom", y=1.05),
                          margin=dict(t=90, b=40))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Stress barriers -------------------------------------------------
with tab_barrier:
    st.subheader("Stress barrier analysis")
    st.markdown(
        "The **stress contrast** is the difference between each sample's Shmin and its depth "
        "trend. Reservoir intervals bounded by higher-stress shale (Shmin contrast ≥ the "
        "threshold set in the sidebar) are contained by **stress barriers**."
    )
    if analysis is None:
        st.info("Run the analysis from the sidebar to identify stress barriers.")
    else:
        detail = analysis["detail"]
        barriers = analysis["barriers"]
        c1, c2 = st.columns(2)
        c1.metric("Stress barrier intervals", f"{len(barriers)}")
        c2.metric("Contrast threshold (psi)", f"{contrast_threshold:,.0f}")

        st.markdown("**Identified stress barriers** (non-reservoir intervals containing a neighbouring reservoir):")
        if barriers.empty:
            st.warning("No stress barriers met the contrast threshold — try lowering it in the sidebar.")
        else:
            st.dataframe(barriers.style.format(precision=1), use_container_width=True, hide_index=True)

        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=["Litho", "Shmin & trend (psi)", "Contrast (psi)"],
                            column_widths=[0.15, 0.45, 0.40])
        _add_lithology_column(fig, results["DEPTH"], results["LITHO_CODE"], col=1)
        fig.add_trace(go.Scatter(x=detail["SHMIN_PSI"], y=detail["DEPTH"], mode="lines",
                                 name="Shmin", line=dict(color="#2980b9")), row=1, col=2)
        fig.add_trace(go.Scatter(x=detail["TREND_PSI"], y=detail["DEPTH"], mode="lines",
                                 name="Shmin trend", line=dict(color="#e67e22", dash="dash")), row=1, col=2)
        fig.add_trace(go.Scatter(x=detail["CONTRAST_PSI"], y=detail["DEPTH"], mode="lines",
                                 name="Contrast", line=dict(color="#16a085")), row=1, col=3)
        fig.add_vline(x=0.0, line=dict(color="gray", dash="dot"), row=1, col=3)
        for _, b in barriers.iterrows():
            fig.add_shape(type="rect", xref="x domain", x0=0.0, x1=1.0, y0=b["Top"], y1=b["Base"],
                          fillcolor="#7f8c8d", opacity=0.20, line_width=0, layer="below", row=1, col=2)
        fig.update_yaxes(autorange="reversed", title_text="MD", col=1)
        fig.update_xaxes(title_text="psi", row=1, col=2)
        fig.update_xaxes(title_text="psi", row=1, col=3)
        fig.update_layout(height=680, legend=dict(orientation="h", yanchor="bottom", y=1.05),
                          margin=dict(t=90, b=40))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Perforation zones -----------------------------------------------
with tab_perf:
    st.subheader("Recommended perforation zones")
    st.markdown(
        "Reservoir intervals are graded from their stress-barrier containment: "
        "**Good** = barrier above *and* below · **Moderate** = barrier on one side · "
        "**Poor** = uncontained or too thin."
    )
    if analysis is None:
        st.info("Run the analysis from the sidebar to generate perforation recommendations.")
    else:
        zones = analysis["zones"]
        detail = analysis["detail"]
        good = int((zones["Quality"] == "Good").sum()) if not zones.empty else 0
        mod = int((zones["Quality"] == "Moderate").sum()) if not zones.empty else 0
        poor = int((zones["Quality"] == "Poor").sum()) if not zones.empty else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Good zones", good)
        c2.metric("🟠 Moderate zones", mod)
        c3.metric("🔴 Poor zones", poor)

        if zones.empty:
            st.warning("No reservoir intervals were found — check the GR cutoff and lithology flag.")
        else:
            def _color_quality(val):
                return f"background-color: {sb.PERF_COLORS.get(val, '#ecf0f1')}; color: black"

            st.markdown("**Perforation zone recommendations** (reservoir intervals, ranked by depth):")
            st.dataframe(
                zones.style.map(_color_quality, subset=["Quality"]).format(precision=1),
                use_container_width=True, hide_index=True,
            )
            recommended = zones[zones["Quality"].isin(["Good", "Moderate"])]
            if not recommended.empty:
                st.success(
                    f"✅ {len(recommended)} recommended perforation interval(s) "
                    f"(Good/Moderate). Best candidates are the **Good** zones with the "
                    "highest stress contrast above and below."
                )

        st.plotly_chart(
            composite_figure(results, detail, zones, height=820),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Download results")
        dcol1, dcol2 = st.columns(2)
        full = results.copy()
        full["PERF_QUALITY"] = detail["PERF_QUALITY"].to_numpy()
        full["CONTRAST_PSI"] = detail["CONTRAST_PSI"].to_numpy()
        full["LITHOLOGY"] = sb.lithology_label_column(full).to_numpy()
        dcol1.download_button(
            "⬇️ Full results (per-depth) CSV",
            data=sb.results_to_csv_bytes(sb.display_frame(full)),
            file_name="perforation_planner_results.csv", mime="text/csv",
            type="primary", use_container_width=True,
        )
        dcol2.download_button(
            "⬇️ Perforation zones CSV",
            data=sb.results_to_csv_bytes(zones),
            file_name="perforation_zones.csv", mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Stress Barrier Analysis & Perforation Planner · built with "
    "[Streamlit](https://streamlit.io) + [geomechpy](https://github.com/0xsmolrun/GeomechPy_smolrun) · "
    "horizontal stresses: Thiercelin & Plumb (1994)."
)

