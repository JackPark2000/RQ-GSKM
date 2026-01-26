
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Clustering Comparison

This script generates plots comparing Standard K-Means (KM) against Gain-Shape K-Means (GSKM) by sweeping across dimensions (D).

Each plot overlays K=256 and K=1024 in the same axis.
Metrics: MSE, gain error, cosine error
  - avg_mse
  - avg_gain_err
  - avg_cos_sim_err

Input: multiple JSON files like:
  clustering_comparison_report_value_dim_128_K1024.json
  clustering_comparison_report_key_dim_8_K256.json
  clustering_comparison_report_key_dim_8_K256_residual.json
etc.

Usage:
  python plot_clustering.py \
    --input_dir clustering_results \
    --output_dir output_figures \
    --Ks 256 1024 \
    --include_residual 1

If you want to also plot residual variants separately:
  python plot_clustering_icml.py --include_residual 1

Author: Minjae Park, Soosung Kim
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------
# Helper Function for Data Parsing
# ----------------------------------
def _as_list(obj: Any) -> List[Dict[str, Any]]:
    """
    Standardized input to a list of dictionaries.
    Handles cases where JSON contains a single dict or a list.
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _try_float(x: Any) -> Optional[float]:
    """Attempts to convert a value to a float"""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _bool_from_filename(path: str) -> bool:
    """Detects if the file represents a residual quantization experiment."""
    base = os.path.basename(path).lower()
    return ("residual" in base) or ("_res" in base)


def _extract_from_filename(path: str) -> Dict[str, Any]:
    """
    Parses metadata (Key/Value, Dimension, K) directly from the filename.
    Example:
      clustering_comparison_report_value_dim_128_K1024.json
      clustering_comparison_report_key_dim_8_K256_residual.json
    """
    base = os.path.basename(path)
    out: Dict[str, Any] = {}

    # Extract key vs value
    m_kv = re.search(r"report_(key|value)_", base, flags=re.IGNORECASE)
    if m_kv:
        out["key_or_value_fn"] = m_kv.group(1).lower()

    # Extract Dimension (D)
    m_dim = re.search(r"_dim_(\d+)", base, flags=re.IGNORECASE)
    if m_dim:
        out["dim_fn"] = int(m_dim.group(1))

    # Extract Number of Clusters (K)
    m_k = re.search(r"_K(\d+)", base, flags=re.IGNORECASE)
    if m_k:
        out["K_fn"] = int(m_k.group(1))

    out["residual"] = _bool_from_filename(path)
    return out


# --------------------------------------------
# Configuration & Constants
# --------------------------------------------

@dataclass
class AlgoSpec:
    json_name: str   # key in averages dict
    label: str       # label in plots


# Algorithm definitions
ALGORITHMS = [
    AlgoSpec(json_name="K-Means", label="KM"),
    AlgoSpec(json_name="ABC",     label="GSKM"),
]

# Metrics to plot
METRICS = {
    "avg_mse": "MSE",
    "avg_gain_err": "Gain error",
    "avg_cos_sim_err": "Cosine similarity",  # (1 - error)로 그릴 예정
}


# --------------------------------------------
# Data Loading Logic
# --------------------------------------------
def load_reports(paths: Iterable[str]) -> pd.DataFrame:
    """
    Reads all JSON reports and consolidates them into Pandas DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fn_meta = _extract_from_filename(path)

        for obj in _as_list(data):
            dim = obj.get("dim", fn_meta.get("dim_fn"))
            kv = obj.get("key_or_value", fn_meta.get("key_or_value_fn"))
            K = obj.get("num_clusters", fn_meta.get("K_fn"))
            trim = obj.get("trim_ratio", None)

            if dim is None or kv is None or K is None:
                # Skip malformed entries
                continue

            averages = obj.get("averages", {}) or {}

            for algo in ALGORITHMS:
                a = averages.get(algo.json_name, {}) or {}
                row = {
                    "file": os.path.basename(path),
                    "path": path,
                    "residual": bool(fn_meta.get("residual", False)),
                    "trim_ratio": _try_float(trim),
                    "key_or_value": str(kv).lower(),
                    "dim": int(dim),
                    "K": int(K),
                    "algo": algo.label,  # KM / GSKM
                }
                print(row)
                # Pull metrics (if missing -> NaN)
                for m in METRICS.keys():
                    row[m] = _try_float(a.get(m, None))
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure numeric dtypes
    df["dim"] = df["dim"].astype(int)
    df["K"] = df["K"].astype(int)
    df["residual"] = df["residual"].astype(bool)
    for m in METRICS.keys():
        df[m] = pd.to_numeric(df[m], errors="coerce")
    print(df)

    return df


# -------------------------------------------
# Plotting Functions
# -------------------------------------------
def set_icml_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })



def plot_metric_sweep(
    df: pd.DataFrame,
    out_path: str,
    key_or_value: str,
    metric_key: str,
    Ks: List[int],
    residual_mode: Optional[bool] = None,
    title_suffix: str = "",
):
    """
    Generates and saves a single plot sweeping across dimensions.
    """
    d = df[df["key_or_value"] == key_or_value].copy()
    d = d[d["K"].isin(Ks)].copy()
    if residual_mode is not None:
        d = d[d["residual"] == residual_mode].copy()

    print(df)
    if d.empty:
        print(f"[WARN] No data for {key_or_value=} {metric_key=} {Ks=} residual={residual_mode}")
        return

    dims_sorted = sorted(d["dim"].unique().tolist())

    algo_to_color = {
        "KM":   "#4d4d4d",
        "GSKM": "#d62728",
    }
    algo_to_marker = {"KM": "o", "GSKM": "s"}

    K_to_style = {}
    styles = {min(Ks): "--", max(Ks): "-"} if len(Ks) >= 2 else {Ks[0]: "-"}
    for K in Ks:
        K_to_style[K] = styles.get(K, "-.")

    algo_to_marker = {"KM": "o", "GSKM": "s"}

    fig = plt.figure(figsize=(3.4, 2.4))
    ax = fig.add_subplot(111)

    # ----- plot lines -----
    for K in sorted(Ks):
        for algo in ["KM", "GSKM"]:
            dd = d[(d["K"] == K) & (d["algo"] == algo)].copy()
            if dd.empty:
                continue
            dd = dd.sort_values("dim")

            y = dd[metric_key].to_numpy()
            if metric_key == "avg_cos_sim_err":
                y = 1.0 - y  # cosine similarity

            ax.plot(
                dd["dim"].to_numpy(),
                y,
                linestyle=K_to_style[K],
                marker=algo_to_marker.get(algo, "o"),
                color=algo_to_color.get(algo, None),  # 핵심: algo별 색 고정
                label=f"{algo} (K={K})",
            )

    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        ax.set_xscale("log", basex=2)

    ax.set_xticks(dims_sorted)
    ax.set_xticklabels([str(x) for x in dims_sorted])
    ax.minorticks_off()

    # Labels
    ax.set_xlabel("D")                   # dim -> D
    ax.set_ylabel(METRICS[metric_key])   # cosine similarity / MSE / gain error

    # ----- Title rules (no metric name in title) -----
    kv_title = "Key" if key_or_value.lower() == "key" else "Value"
    if residual_mode is True:
        # mathtext for 1^st style
        title = f"{kv_title} (1$^{{st}}$ Residual)"
    else:
        # non-residual OR mixed(None) -> just Key/Value
        title = kv_title

    if title_suffix:
        title += f" · {title_suffix}"
    ax.set_title(title)

    #ax.legend(loc="best", frameon=True)
    ax.legend(loc="best", frameon=True, handlelength=3.2)


    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")

# ---------------------------------------------
# Main Execution
# ---------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing clustering_comparison_report_*.json")
    ap.add_argument("--output_dir", type=str, required=True, help="Where to save figures (pdf/png)")
    ap.add_argument("--pattern", type=str, default="clustering_comparison_report_*.json")
    ap.add_argument("--Ks", type=int, nargs="+", default=[256, 1024], help="Cluster counts to overlay in one plot (e.g., 256 1024)")
    ap.add_argument("--trim_ratio", type=float, default=None, help="Optional: filter exact trim_ratio (e.g., 0.01). If None, do not filter.")
    ap.add_argument("--include_residual", type=int, default=0,
                    help="0: ignore residual flag (mix). 1: plot both residual and non-residual separately.")
    ap.add_argument("--ext", type=str, default="pdf", choices=["pdf", "png"], help="Output file extension")
    args = ap.parse_args()

    set_icml_style()

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No JSON files found: {os.path.join(args.input_dir, args.pattern)}")

    df = load_reports(paths)
    if df.empty:
        raise RuntimeError("Loaded dataframe is empty. Check JSON formats / patterns.")

    # Optional trim filter (exact match with tolerance)
    if args.trim_ratio is not None:
        tol = 1e-12
        df = df[df["trim_ratio"].notna() & (np.abs(df["trim_ratio"] - args.trim_ratio) <= tol)].copy()

    # Keep only K of interest
    df = df[df["K"].isin(args.Ks)].copy()

    # Sanity print
    print(df.groupby(["key_or_value", "K", "algo", "residual"])["dim"].nunique().reset_index(name="num_dims"))

    # Plot
    key_or_values = sorted(df["key_or_value"].unique().tolist())
    for kv in key_or_values:
        for metric_key in METRICS.keys():
            if args.include_residual == 1:
                for residual_mode in [False, True]:
                    out_name = f"{kv}_{metric_key}_K{'_'.join(map(str, sorted(args.Ks)))}_{'res' if residual_mode else 'nonres'}.{args.ext}"
                    out_path = os.path.join(args.output_dir, out_name)
                    plot_metric_sweep(
                        df=df,
                        out_path=out_path,
                        key_or_value=kv,
                        metric_key=metric_key,
                        Ks=sorted(args.Ks),
                        residual_mode=residual_mode,
                    )
            else:
                out_name = f"{kv}_{metric_key}_K{'_'.join(map(str, sorted(args.Ks)))}.{args.ext}"
                out_path = os.path.join(args.output_dir, out_name)
                plot_metric_sweep(
                    df=df,
                    out_path=out_path,
                    key_or_value=kv,
                    metric_key=metric_key,
                    Ks=sorted(args.Ks),
                    residual_mode=None,
                )


if __name__ == "__main__":
    main()

