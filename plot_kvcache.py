#!/usr/bin/env python3
"""
Figure 4: Llama-3-8B KV-Cache Reconstruction (KM vs GSKM).

Reads JSON reports produced by ``run_clustering.py`` and generates comparison
plots sweeping across head dimensions D for each (Key/Value) x (Original/Residual)
combination.

Usage:
    python plot_kvcache.py \
        --input_dir clustering_results \
        --output_dir figures/kvcache \
        --Ks 256 1024 \
        --include_residual 1
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AlgoSpec:
    json_name: str   # key inside the ``averages`` dict of each JSON report
    label: str       # display label in plots


ALGORITHMS = [
    AlgoSpec(json_name="K-Means", label="KM"),
    AlgoSpec(json_name="ABC",     label="GSKM"),
]

METRICS = {
    "avg_mse":         "MSE",
    "avg_gain_err":    "Gain error",
    "avg_cos_sim_err": "Cosine similarity",
}

ALGO_COLOR  = {"KM": "#4d4d4d", "GSKM": "#d62728"}
ALGO_MARKER = {"KM": "o", "GSKM": "s"}


# ---------------------------------------------------------------------------
# Data parsing helpers
# ---------------------------------------------------------------------------

def _as_list(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _try_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _extract_from_filename(path: str) -> Dict[str, Any]:
    """Parse metadata (key/value, D, K, residual) from the filename."""
    base = os.path.basename(path)
    out: Dict[str, Any] = {}

    m_kv = re.search(r"report_(key|value)_", base, flags=re.IGNORECASE)
    if m_kv:
        out["key_or_value_fn"] = m_kv.group(1).lower()

    m_dim = re.search(r"_dim_(\d+)", base, flags=re.IGNORECASE)
    if m_dim:
        out["dim_fn"] = int(m_dim.group(1))

    m_k = re.search(r"_K(\d+)", base, flags=re.IGNORECASE)
    if m_k:
        out["K_fn"] = int(m_k.group(1))

    out["residual"] = "residual" in base.lower()
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reports(paths: Iterable[str]) -> pd.DataFrame:
    """Read all JSON reports and consolidate into a DataFrame."""
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
                continue

            averages = obj.get("averages", {}) or {}

            for algo in ALGORITHMS:
                a = averages.get(algo.json_name, {}) or {}
                row = {
                    "file": os.path.basename(path),
                    "residual": bool(fn_meta.get("residual", False)),
                    "trim_ratio": _try_float(trim),
                    "key_or_value": str(kv).lower(),
                    "dim": int(dim),
                    "K": int(K),
                    "algo": algo.label,
                }
                for m in METRICS:
                    row[m] = _try_float(a.get(m, None))
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["dim"] = df["dim"].astype(int)
    df["K"] = df["K"].astype(int)
    df["residual"] = df["residual"].astype(bool)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_metric_sweep(df: pd.DataFrame, out_path: str,
                      key_or_value: str, metric_key: str,
                      Ks: List[int],
                      residual_mode: Optional[bool] = None):
    """Generate and save a single D-sweep plot."""
    d = df[df["key_or_value"] == key_or_value].copy()
    d = d[d["K"].isin(Ks)].copy()
    if residual_mode is not None:
        d = d[d["residual"] == residual_mode].copy()

    if d.empty:
        print(f"  [WARN] No data for {key_or_value=} {metric_key=} {Ks=} "
              f"residual={residual_mode}")
        return

    dims_sorted = sorted(d["dim"].unique().tolist())

    # Linestyle by K (dashed for smaller K, solid for larger K)
    K_to_style = {}
    styles = {min(Ks): "--", max(Ks): "-"} if len(Ks) >= 2 else {Ks[0]: "-"}
    for K in Ks:
        K_to_style[K] = styles.get(K, "-.")

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    for K in sorted(Ks):
        for algo in ["KM", "GSKM"]:
            dd = d[(d["K"] == K) & (d["algo"] == algo)].sort_values("dim")
            if dd.empty:
                continue

            y = dd[metric_key].to_numpy()
            if metric_key == "avg_cos_sim_err":
                y = 1.0 - y  # plot as cosine similarity

            ax.plot(
                dd["dim"].to_numpy(), y,
                linestyle=K_to_style[K],
                marker=ALGO_MARKER.get(algo, "o"),
                color=ALGO_COLOR.get(algo),
                label=f"{algo} (K={K})",
            )

    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        ax.set_xscale("log", basex=2)
    ax.set_xticks(dims_sorted)
    ax.set_xticklabels([str(x) for x in dims_sorted])
    ax.minorticks_off()

    ax.set_xlabel("D")
    ax.set_ylabel(METRICS[metric_key])

    kv_title = "Key" if key_or_value == "key" else "Value"
    if residual_mode is True:
        title = f"{kv_title} (1$^{{st}}$ Residual)"
    else:
        title = kv_title
    ax.set_title(title)

    ax.legend(loc="best", frameon=True, handlelength=3.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Figure 4: KV-Cache Reconstruction Comparison")
    ap.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing clustering_comparison_report_*.json")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Directory to save output figures")
    ap.add_argument("--pattern", type=str,
                    default="clustering_comparison_report_*.json")
    ap.add_argument("--Ks", type=int, nargs="+", default=[256, 1024],
                    help="Codebook sizes to overlay in each plot")
    ap.add_argument("--trim_ratio", type=float, default=None,
                    help="Filter by exact trim_ratio (optional)")
    ap.add_argument("--include_residual", type=int, default=0,
                    help="1: plot original and residual separately")
    ap.add_argument("--ext", type=str, default="pdf", choices=["pdf", "png"])
    args = ap.parse_args()

    set_plot_style()

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No JSON files found: {os.path.join(args.input_dir, args.pattern)}"
        )

    df = load_reports(paths)
    if df.empty:
        raise RuntimeError("Loaded DataFrame is empty. Check JSON formats.")

    if args.trim_ratio is not None:
        df = df[df["trim_ratio"].notna()
                & (np.abs(df["trim_ratio"] - args.trim_ratio) <= 1e-12)].copy()

    df = df[df["K"].isin(args.Ks)].copy()

    print(df.groupby(["key_or_value", "K", "algo", "residual"])
            ["dim"].nunique().reset_index(name="num_dims"))

    for kv in sorted(df["key_or_value"].unique()):
        for metric_key in METRICS:
            if args.include_residual == 1:
                for residual_mode in [False, True]:
                    tag = "res" if residual_mode else "nonres"
                    K_tag = "_".join(map(str, sorted(args.Ks)))
                    out_name = f"{kv}_{metric_key}_K{K_tag}_{tag}.{args.ext}"
                    plot_metric_sweep(
                        df, os.path.join(args.output_dir, out_name),
                        kv, metric_key, sorted(args.Ks), residual_mode,
                    )
            else:
                K_tag = "_".join(map(str, sorted(args.Ks)))
                out_name = f"{kv}_{metric_key}_K{K_tag}.{args.ext}"
                plot_metric_sweep(
                    df, os.path.join(args.output_dir, out_name),
                    kv, metric_key, sorted(args.Ks), None,
                )


if __name__ == "__main__":
    main()
