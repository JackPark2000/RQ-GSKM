#!/usr/bin/env python3
"""
Figure 3: Synthetic Gaussian Experiments (KM vs GSKM).

Generates D-sweep and K-sweep comparison plots on random Gaussian data.
Produces 6 PDF figures: {D, K}-sweep x {MSE, Gain Error, Cosine Similarity}.

Usage:
    python plot_synthetic.py --out_dir figures/synthetic \
        --dims 4 8 16 32 64 128 256 512 1024 \
        --Ks 64 128 256 512 1024 2048 4096 8192 16384
"""

import os
import argparse
import time
from typing import Dict, List, Tuple

import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

from algorithms import run_comparison

# ---------------------------------------------------------------------------
# Style & constants
# ---------------------------------------------------------------------------

ALGO_ORDER = ["KM", "GSKM"]
ALGO_COLOR = {"KM": "#4d4d4d", "GSKM": "#d62728"}
ALGO_MARKER = {"KM": "o", "GSKM": "s"}


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 1.8, "lines.markersize": 5,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _extract_km_gskm(
    results: Dict[str, Tuple[float, float, float, float, float]],
) -> Dict[str, Dict[str, float]]:
    """Convert run_comparison() output to a {algo: {metric: value}} dict."""
    out = {}
    for src_key, algo_label in [("K-Means", "KM"), ("ABC", "GSKM")]:
        mse, _, gain_err, cos_err, _ = results[src_key]
        out[algo_label] = {
            "MSE": float(mse),
            "GainError": float(gain_err),
            "CosineSimilarity": 1.0 - float(cos_err),
        }
    return out


def run_d_sweep(N: int, dims: List[int], K: int,
                trim_ratio: float, seed: int) -> pd.DataFrame:
    """Sweep across dimensions D with fixed K."""
    rows = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for D in dims:
        print(f"  [D-sweep] D={D}, K={K}")
        data = torch.randn(N, D, generator=gen)
        metrics = _extract_km_gskm(
            run_comparison(data, num_clusters=K, trim_ratio=trim_ratio)
        )
        for algo in ALGO_ORDER:
            rows.append({"D": D, "Algorithm": algo, **metrics[algo]})
    return pd.DataFrame(rows)


def run_k_sweep(N: int, D: int, Ks: List[int],
                trim_ratio: float, seed: int) -> pd.DataFrame:
    """Sweep across cluster counts K with fixed D."""
    rows = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    data = torch.randn(N, D, generator=gen)
    for K in Ks:
        print(f"  [K-sweep] K={K}, D={D}")
        metrics = _extract_km_gskm(
            run_comparison(data, num_clusters=K, trim_ratio=trim_ratio)
        )
        for algo in ALGO_ORDER:
            rows.append({"K": K, "Algorithm": algo, **metrics[algo]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _set_log2_x(ax, xticks: List[int]):
    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        ax.set_xscale("log", basex=2)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.minorticks_off()


def plot_metric(df: pd.DataFrame, x_col: str, metric_col: str,
                x_label: str, y_label: str, title: str,
                xticks: List[int], out_path: str):
    """Generate a single comparison plot for one metric."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    for algo in ALGO_ORDER:
        dd = df[df["Algorithm"] == algo].sort_values(x_col)
        ax.plot(
            dd[x_col].to_numpy(), dd[metric_col].to_numpy(),
            color=ALGO_COLOR[algo], marker=ALGO_MARKER[algo],
            linestyle="-", label=algo,
        )

    _set_log2_x(ax, xticks)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Legend without markers for cleaner appearance
    handles, labels = ax.get_legend_handles_labels()
    clean = [Line2D([0], [0], color=h.get_color(), linestyle=h.get_linestyle(),
                    linewidth=h.get_linewidth()) for h in handles]
    ax.legend(clean, labels, loc="best", frameon=True, handlelength=3.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRIC_SPECS = [
    ("MSE",              "MSE"),
    ("GainError",        "Gain error"),
    ("CosineSimilarity", "Cosine similarity"),
]


def main():
    ap = argparse.ArgumentParser(description="Figure 3: Synthetic Gaussian Experiments")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory to save output PDFs")
    ap.add_argument("--N", type=int, default=10000,
                    help="Number of random samples")
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
                    help="Dimensions for D-sweep")
    ap.add_argument("--K_fixed", type=int, default=2048,
                    help="Fixed K for D-sweep")
    ap.add_argument("--D_fixed", type=int, default=256,
                    help="Fixed D for K-sweep")
    ap.add_argument("--Ks", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                    help="Cluster counts for K-sweep")
    ap.add_argument("--trim_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_plot_style()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # D-sweep
    print("Running D-sweep ...")
    df_d = run_d_sweep(args.N, args.dims, args.K_fixed, args.trim_ratio, args.seed)
    for col, ylabel in METRIC_SPECS:
        plot_metric(
            df_d, "D", col, "D", ylabel,
            f"D-sweep (K={args.K_fixed})", args.dims,
            os.path.join(args.out_dir, f"dsweep_{col}_K{args.K_fixed}.pdf"),
        )

    # K-sweep
    print("Running K-sweep ...")
    df_k = run_k_sweep(args.N, args.D_fixed, args.Ks, args.trim_ratio, args.seed)
    for col, ylabel in METRIC_SPECS:
        plot_metric(
            df_k, "K", col, "K", ylabel,
            f"K-sweep (D={args.D_fixed})", args.Ks,
            os.path.join(args.out_dir, f"ksweep_{col}_D{args.D_fixed}.pdf"),
        )

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
