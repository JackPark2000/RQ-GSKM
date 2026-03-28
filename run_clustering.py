#!/usr/bin/env python3
"""
Run KM vs GSKM clustering on dumped KV-cache activations and save JSON reports.

For each (dim, key/value, K) configuration, this script:
  1. Loads pre-extracted .pt files from ``dumped_data_dim_{D}/``.
  2. Runs Standard K-Means (KM) and Gain-Shape K-Means (GSKM).
  3. Computes the KM residual (x - KM(x)) and re-runs both algorithms on it.
  4. Saves two JSON reports (original + residual) to ``clustering_results/``.

Usage:
    python run_clustering.py <dim> <key|value> <num_clusters> [--data_dir DIR]

Example:
    python run_clustering.py 128 key 1024
    python run_clustering.py 128 key 1024 --data_dir /data/home/jyc
"""

import sys
import os
import json
import glob
import argparse

import torch

from algorithms import run_comparison

METHODS = ["K-Means", "ABC"]


def _accumulate(totals, method, metrics):
    """Accumulate per-file metrics into running totals."""
    mse, tmse, gain_err, cos_sim_err, sil = metrics
    totals[method]["mse"] += mse
    totals[method]["tmse"] += tmse
    totals[method]["gain_err"] += gain_err
    totals[method]["cos_sim_err"] += cos_sim_err
    if sil != -1.0:
        totals[method]["sil"] += sil
        totals[method]["count_sil"] += 1


def _compute_averages(totals, n_files):
    """Compute per-method average metrics across all files."""
    averages = {}
    for m in METHODS:
        n_sil = totals[m]["count_sil"]
        averages[m] = {
            "avg_mse": totals[m]["mse"] / n_files,
            "avg_tmse_1pct": totals[m]["tmse"] / n_files,
            "avg_gain_err": totals[m]["gain_err"] / n_files,
            "avg_cos_sim_err": totals[m]["cos_sim_err"] / n_files,
            "avg_sil": totals[m]["sil"] / n_sil if n_sil > 0 else -1.0,
        }
    return averages


def _new_totals():
    return {m: {"mse": 0.0, "tmse": 0.0, "gain_err": 0.0,
                "cos_sim_err": 0.0, "sil": 0.0, "count_sil": 0}
            for m in METHODS}


def _print_metrics(tag, method, metrics):
    mse, tmse, gain_err, cos_sim_err, sil = metrics
    print(f"  [{tag}] {method:10s}: MSE={mse:.6f}  GainErr={gain_err:.6f}  "
          f"CosSimErr={cos_sim_err:.6f}  Sil={sil:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Run KM vs GSKM on KV-cache data")
    ap.add_argument("dim", type=int, help="Head dimension D")
    ap.add_argument("key_or_value", choices=["key", "value"])
    ap.add_argument("num_clusters", type=int, help="Number of clusters K")
    ap.add_argument("--data_dir", type=str, default=".",
                    help="Root directory containing dumped_data_dim_*/ (default: .)")
    ap.add_argument("--trim_ratio", type=float, default=0.01)
    args = ap.parse_args()

    dim = args.dim
    key_or_value = args.key_or_value
    num_clusters = args.num_clusters
    trim_ratio = args.trim_ratio

    file_pattern = os.path.join(
        args.data_dir, f"dumped_data_dim_{dim}/raw_data_layer_*_{key_or_value}_cb*.pt"
    )
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found matching: {file_pattern}")
        sys.exit(1)

    orig_totals = _new_totals()
    resid_totals = _new_totals()
    orig_details = {}
    resid_details = {}

    for f in files:
        fname = os.path.basename(f)
        payload = torch.load(f)
        data = payload["data"].float()
        weights = payload.get("weights", None)

        # -- Original data --
        results, recon_km = run_comparison(
            data, weights, num_clusters=num_clusters,
            trim_ratio=trim_ratio, return_km_recon=True,
        )
        orig_details[fname] = results
        print(f"\n{fname}:")
        for m in METHODS:
            _print_metrics("orig", m, results[m])
            _accumulate(orig_totals, m, results[m])

        # -- Residual: x - KM(x) --
        residual = (data - recon_km).contiguous()
        resid_results = run_comparison(
            residual, weights, num_clusters=num_clusters,
            trim_ratio=trim_ratio,
        )
        resid_details[fname] = resid_results
        for m in METHODS:
            _print_metrics("resid", m, resid_results[m])
            _accumulate(resid_totals, m, resid_results[m])

    n_files = len(files)
    orig_averages = _compute_averages(orig_totals, n_files)
    resid_averages = _compute_averages(resid_totals, n_files)

    # Print summary
    print(f"\n{'Averages (Original)':^50s}")
    print("-" * 50)
    for m in METHODS:
        a = orig_averages[m]
        print(f"  {m:10s}: MSE={a['avg_mse']:.6f}  "
              f"GainErr={a['avg_gain_err']:.6f}  CosSimErr={a['avg_cos_sim_err']:.6f}")

    print(f"\n{'Averages (Residual)':^50s}")
    print("-" * 50)
    for m in METHODS:
        a = resid_averages[m]
        print(f"  {m:10s}: MSE={a['avg_mse']:.6f}  "
              f"GainErr={a['avg_gain_err']:.6f}  CosSimErr={a['avg_cos_sim_err']:.6f}")

    # Save reports
    os.makedirs("clustering_results", exist_ok=True)
    base = f"clustering_results/clustering_comparison_report_{key_or_value}_dim_{dim}_K{num_clusters}"

    for suffix, averages, details in [
        ("", orig_averages, orig_details),
        ("_residual", resid_averages, resid_details),
    ]:
        report = {
            "dim": dim,
            "key_or_value": key_or_value,
            "num_clusters": num_clusters,
            "trim_ratio": trim_ratio,
            "averages": averages,
            "details": details,
        }
        out_path = f"{base}{suffix}.json"
        with open(out_path, "w") as jf:
            json.dump(report, jf, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
