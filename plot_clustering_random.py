#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Normal data sweeps (ICML-ish style) and save 6 PDF figures:

1) D sweep (x = D, log2 scale):  MSE, Gain Error, Cosine Similarity
2) K sweep (x = K, log2 scale):  MSE, Gain Error, Cosine Similarity

- Compare KM vs GSKM
  * KM   := "K-Means" in run_comparison() results
  * GSKM := "ABC"     in run_comparison() results
- Cosine Similarity = 1 - AvgCosineSimilarityError
- Save 6 PDFs into --out_dir

IMPORTANT:
This file assumes you already have `run_comparison(data_cpu, num_clusters=..., trim_ratio=...)`
defined (exactly like in your provided code), and it returns:
  results["K-Means"] = (mse, tmse, gain_err, cos_err, sil)
  results["ABC"]     = (mse, tmse, gain_err, cos_err, sil)

Usage:
  python sweep_random_plots.py --out_dir ./figs
"""

import os
import argparse
import math
import time
import torch
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np
import cupy as cp
import torch.utils.dlpack as dlpack
from cuml.cluster import KMeans as cuKMeans
from cuml.metrics.cluster import silhouette_score
import json
import os
import glob
import math


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

def torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    """CUDA torch tensor -> CuPy (DLPack, zero-copy)"""
    if not x.is_cuda:
        raise ValueError("torch_to_cupy expects a CUDA tensor.")
    return cp.from_dlpack(dlpack.to_dlpack(x.contiguous()))


def _centroids_mean_from_labels(data_gpu: torch.Tensor, labels_gpu: torch.Tensor, K: int):
    """원래 공간 data에서 label 기준 cluster mean centroid 생성 (GPU)"""
    N, D = data_gpu.shape
    cent = torch.zeros((K, D), device=data_gpu.device, dtype=data_gpu.dtype)
    cnt = torch.zeros((K,), device=data_gpu.device, dtype=data_gpu.dtype)

    cent.scatter_add_(0, labels_gpu[:, None].expand(-1, D), data_gpu)
    cnt.scatter_add_(0, labels_gpu, torch.ones((N,), device=data_gpu.device, dtype=data_gpu.dtype))
    cent = cent / cnt.clamp_min(1.0).unsqueeze(1)
    return cent


def _compute_avg_gain_norm(data_gpu, label_gpu, num_entries):
    """cluster별 gain = ||x|| 의 평균"""
    sum_gain = torch.zeros(num_entries, device=data_gpu.device)
    counts = torch.zeros(num_entries, device=data_gpu.device)
    sample_gain = torch.norm(data_gpu, p=2, dim=-1)
    sum_gain.scatter_add_(0, label_gpu, sample_gain)
    counts.scatter_add_(0, label_gpu, torch.ones_like(sample_gain))
    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    return sum_gain / safe_counts

def _compute_avg_gain_proj(data_gpu, label_gpu, center_dir, num_entries):
    """cluster별 gain = (x · ĉ_k) projection 평균"""
    sum_gain = torch.zeros(num_entries, device=data_gpu.device)
    counts = torch.zeros(num_entries, device=data_gpu.device)

    dir_assigned = center_dir[label_gpu]
    proj = torch.sum(data_gpu * dir_assigned, dim=-1)
    sum_gain.scatter_add_(0, label_gpu, proj)
    counts.scatter_add_(0, label_gpu, torch.ones_like(proj))

    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    return (sum_gain / safe_counts).clamp(min=1e-4)

def _centroids_mean_from_labels(data_gpu: torch.Tensor, labels_gpu: torch.Tensor, K: int):
    """원래 공간 data에서 label 기준 cluster mean centroid 생성 (GPU)"""
    N, D = data_gpu.shape
    cent = torch.zeros((K, D), device=data_gpu.device, dtype=data_gpu.dtype)
    cnt = torch.zeros((K,), device=data_gpu.device, dtype=data_gpu.dtype)

    cent.scatter_add_(0, labels_gpu[:, None].expand(-1, D), data_gpu)
    cnt.scatter_add_(0, labels_gpu, torch.ones((N,), device=data_gpu.device, dtype=data_gpu.dtype))
    cent = cent / cnt.clamp_min(1.0).unsqueeze(1)
    return cent

# ---
# --------------------------------------------------------------------------------
# 1. 공통 유틸리티 및 메트릭 함수
# --------------------------------------------------------------------------------

def _trimmed_mse(data_cpu: torch.Tensor, recon_cpu: torch.Tensor, trim_ratio: float = 0.01) -> float:
    """
    Trimmed MSE: per-sample MSE를 구한 뒤 상위 trim_ratio 비율을 제거하고 평균.
    - data_cpu, recon_cpu: [N, D] on CPU float
    """
    with torch.no_grad():
        err = (recon_cpu - data_cpu).float()
        per_sample = (err * err).mean(dim=1)  # [N]
        n = per_sample.numel()
        if n == 0:
            return float("nan")
        k = int(math.floor(n * (1.0 - trim_ratio)))
        k = max(1, k)  # 최소 1개는 남김
        vals, _ = torch.topk(per_sample, k=k, largest=False)  # 작은 쪽 k개
        return vals.mean().item()

def trimmed_mse_by_gain(data_cpu, recon_cpu, trim_ratio=0.03, eps=1e-12):
    # data_cpu, recon_cpu: [N,D] float CPU
    g = torch.norm(data_cpu, dim=1)                 # [N]
    thr = torch.quantile(g, 1.0 - trim_ratio)       # q_(1-trim)
    keep = g <= thr
    if keep.sum() == 0:
        return float("nan")
    err = (recon_cpu[keep] - data_cpu[keep]).pow(2).mean()
    return err.item()

def avg_gain_error(data_cpu: torch.Tensor, recon_cpu: torch.Tensor, eps: float = 1e-12) -> float:
    """
    AvgGainErr = E[ | ||x|| - ||xhat|| | ]
    """
    with torch.no_grad():
        g = torch.norm(data_cpu.float(), dim=1)
        gh = torch.norm(recon_cpu.float(), dim=1)
        return (g - gh).abs().mean().item()

def avg_cosine_similarity_error(data_cpu: torch.Tensor, recon_cpu: torch.Tensor, eps: float = 1e-12) -> float:
    """
    AvgCosSimErr = E[ 1 - cos(x, xhat) ]
    """
    with torch.no_grad():
        x = data_cpu.float()
        y = recon_cpu.float()
        x_n = x / (torch.norm(x, dim=1, keepdim=True) + eps)
        y_n = y / (torch.norm(y, dim=1, keepdim=True) + eps)
        cos = (x_n * y_n).sum(dim=1).clamp(-1.0, 1.0)  # [N]
        return (1.0 - cos).mean().item()

def avg_angular_error(data_cpu: torch.Tensor, recon_cpu: torch.Tensor, eps: float = 1e-12, use_degrees: bool = True) -> float:
    """
    Angular Error = E[ arccos(cos(x, xhat)) ]
    Cosine Similarity Error보다 미세한 각도 차이에 선형적으로 민감하게 반응합니다.
    """
    with torch.no_grad():
        x = data_cpu.float()
        y = recon_cpu.float()
        
        # 벡터 정규화 (L2 Norm)
        x_n = x / (torch.norm(x, dim=1, keepdim=True) + eps)
        y_n = y / (torch.norm(y, dim=1, keepdim=True) + eps)
        
        # 내적을 통한 Cosine Similarity 계산 및 수치적 안정성을 위한 클램핑
        # 1.0을 아주 미세하게 초과할 경우 acos에서 NaN이 발생할 수 있으므로 주의가 필요합니다.
        cos = (x_n * y_n).sum(dim=1).clamp(-1.0 + eps, 1.0 - eps)
        
        # 아크코사인으로 각도(Radians) 추출
        theta = torch.acos(cos)
        
        if use_degrees:
            # Degree로 변환하여 가독성 증대 (0 ~ 180도)
            theta = theta * (180.0 / torch.pi)
            
        return theta.mean().item()


def evaluate_metrics(data, recon, labels, sil_features=None, max_samples=10000, trim_ratio=0.01):
    """
    - MSE: (recon vs original data)
    - Trimmed MSE: (여기서는 gain 기준 trim)  -> tMSE_by_gain
    - AvgGainErr: E[ | ||x|| - ||xhat|| | ]
    - AvgCosSimErr: E[ 1 - cos(x, xhat) ]
    - Silhouette: cosine metric, sil_features 가 주어지면 그것 사용 (보통 normalized)
    반환: (mse, tmse, avg_gain_err, avg_cos_sim_err, sil)
    """
    # --- CPU metrics ---
    data_cpu = data.detach().cpu().float()
    recon_cpu = recon.detach().cpu().float()

    mse = F.mse_loss(recon_cpu, data_cpu).item()
    #sse_per_vector = (recon_cpu - data_cpu).pow(2).sum(dim=1)
    #mse = sse_per_vector.mean().item()

    # tmse = _trimmed_mse(data_cpu, recon_cpu, trim_ratio=trim_ratio)          # (오차 기준 trim)
    tmse = trimmed_mse_by_gain(data_cpu, recon_cpu, trim_ratio=trim_ratio)     # (gain 기준 trim)

    gain_err = avg_gain_error(data_cpu, recon_cpu)
    cos_sim_err = avg_cosine_similarity_error(data_cpu, recon_cpu)
    #cos_sim_err = avg_angular_error(data_cpu, recon_cpu)

    # --- Silhouette ---
    feats = sil_features if sil_features is not None else data
    feats = feats.detach()
    labels_t = labels.detach() if torch.is_tensor(labels) else torch.tensor(labels)

    # labels를 feats와 같은 device로 맞춤
    labels_t = labels_t.to(feats.device, non_blocking=True)

    n = feats.shape[0]
    if n > max_samples:
        idx = torch.randperm(n, device=feats.device)[:max_samples]
        feats_s = feats.index_select(0, idx)
        labels_s = labels_t.index_select(0, idx)
    else:
        feats_s = feats
        labels_s = labels_t

    feats_s = feats_s.to("cuda", non_blocking=True).float()
    labels_s = labels_s.to("cuda", non_blocking=True).int()

    try:
        sil = silhouette_score(torch_to_cupy(feats_s), torch_to_cupy(labels_s), metric="cosine")
        sil = float(sil)
    except Exception:
        sil = -1.0

    return mse, tmse, gain_err, cos_sim_err, sil


# ------------------------------------------------------------
# You must have run_comparison() available in this namespace.
# Either paste your run_comparison() above this script,
# or import it from your module:
#
# from your_module import run_comparison
# ------------------------------------------------------------
def run_comparison(data_cpu, weights_cpu=None, num_clusters=2**10, tag="Target", trim_ratio=0.03):
    device = torch.device("cuda")
    data_gpu = data_cpu.to(device).float()

    weight_cp = None
    if weights_cpu is not None:
        w_gpu = weights_cpu.to(device).float()
        weight_cp = torch_to_cupy(w_gpu)

    results = {}

    data_norm_gpu = F.normalize(data_gpu, p=2, dim=-1)

    # [1] Standard K-Means
    km_std = cuKMeans(n_clusters=num_clusters, max_iter=20, random_state=42)
    km_std.fit(torch_to_cupy(data_gpu))
    labels_std = torch.from_numpy(km_std.labels_.get()).to(device).long()
    centroids_std = torch.from_numpy(km_std.cluster_centers_.get()).to(device).float()
    recon_std = centroids_std[labels_std]
    results["K-Means"] = evaluate_metrics(
        data_cpu, recon_std, labels_std, sil_features=None, trim_ratio=trim_ratio
    )

    # [1-b] Baseline: LogGain-Assign KMeans
    r = torch.norm(data_gpu, p=2, dim=-1)
    tau = r.median().clamp_min(1e-12)
    rp = torch.log1p(r / tau)                       # [N]
    xprime = data_norm_gpu * rp.unsqueeze(1)        # [N,D]

    km_log = cuKMeans(n_clusters=num_clusters, max_iter=20, random_state=42)
    km_log.fit(torch_to_cupy(xprime))
    labels_log = torch.from_numpy(km_log.labels_.get()).to(device).long()

    centroids_log = _centroids_mean_from_labels(data_gpu, labels_log, num_clusters)
    recon_log = centroids_log[labels_log]
    results["LogGain-Assign KMeans"] = evaluate_metrics(
        data_cpu, recon_log, labels_log, sil_features=data_norm_gpu, trim_ratio=trim_ratio
    )

    # [2] Spherical K-Means: on normalized vectors
    km_sph = cuKMeans(n_clusters=num_clusters, max_iter=20, random_state=42)
    km_sph.fit(torch_to_cupy(data_norm_gpu), sample_weight=weight_cp)

    labels_sph_gpu = torch.from_numpy(km_sph.labels_.get()).to(device).long()
    centroids_sph_gpu = torch.from_numpy(km_sph.cluster_centers_.get()).to(device).float()
    center_dir = F.normalize(centroids_sph_gpu, p=2, dim=-1)

    # [2-a] Spherical-Only
    recon_sph_only = center_dir[labels_sph_gpu]
    results["Spherical-Only"] = evaluate_metrics(
        data_cpu, recon_sph_only, labels_sph_gpu, sil_features=data_norm_gpu, trim_ratio=trim_ratio
    )

    # [2-b] Spherical + AvgNormGain
    gain_norm = _compute_avg_gain_norm(data_gpu, labels_sph_gpu, num_clusters).clamp(min=1e-4)
    recon_sph_normgain = center_dir[labels_sph_gpu] * gain_norm[labels_sph_gpu].unsqueeze(1)
    results["Spherical"] = evaluate_metrics(
        data_cpu, recon_sph_normgain, labels_sph_gpu, sil_features=data_norm_gpu, trim_ratio=trim_ratio
    )

    # [2-c] Spherical + AvgProjGain
    gain_proj = _compute_avg_gain_proj(data_gpu, labels_sph_gpu, center_dir, num_clusters)
    recon_sph_projgain = center_dir[labels_sph_gpu] * gain_proj[labels_sph_gpu].unsqueeze(1)
    results["Spherical-ProjGain"] = evaluate_metrics(
        data_cpu, recon_sph_projgain, labels_sph_gpu, sil_features=data_norm_gpu, trim_ratio=trim_ratio
    )

    # [3] S-Init K-Means
    centroids_sph_np = centroids_sph_gpu.detach().cpu().numpy()
    km_sinit = cuKMeans(n_clusters=num_clusters, max_iter=20, init=centroids_sph_np, n_init=1)
    km_sinit.fit(torch_to_cupy(data_gpu))
    labels_sinit = torch.from_numpy(km_sinit.labels_.get()).to(device).long()
    centroids_sinit = torch.from_numpy(km_sinit.cluster_centers_.get()).to(device).float()
    recon_sinit = centroids_sinit[labels_sinit]
    results["S-Init K-Means"] = evaluate_metrics(
        data_cpu, recon_sinit, labels_sinit, sil_features=None, trim_ratio=trim_ratio
    )

    # [4] ABC Method
    shape = center_dir.clone()
    raw_gain_abc = gain_norm.detach().cpu().clone()

    for _ in range(10):
        gain_q = raw_gain_abc.to(device).to(torch.bfloat16).float()
        projections = torch.matmul(data_gpu, shape.t())
        scores = (2 * projections * gain_q.unsqueeze(0)) - gain_q.unsqueeze(0).pow(2)
        labels_abc = torch.argmax(scores, dim=1)

        new_shape_acc = torch.zeros_like(shape)
        new_shape_acc.scatter_add_(
            0,
            labels_abc.unsqueeze(1).expand(-1, shape.shape[1]),
            data_gpu * gain_q[labels_abc].unsqueeze(1)
        )
        shape = F.normalize(new_shape_acc, p=2, dim=-1)

        proj_assigned = projections.gather(1, labels_abc.unsqueeze(1)).squeeze(1)
        new_gain = torch.zeros_like(gain_q)
        cnts = torch.zeros_like(gain_q)
        new_gain.scatter_add_(0, labels_abc, proj_assigned)
        cnts.scatter_add_(0, labels_abc, torch.ones_like(proj_assigned))
        updated = torch.where(cnts > 0, new_gain / cnts, gain_q).clamp(min=1e-4)
        raw_gain_abc.data.copy_(updated.detach().cpu())

    recon_abc = shape[labels_abc] * raw_gain_abc.to(device)[labels_abc].unsqueeze(1)
    results["ABC"] = evaluate_metrics(
        data_cpu, recon_abc, labels_abc, sil_features=data_norm_gpu, trim_ratio=trim_ratio
    )

    return results

#
# -----------------------------
# Style
# -----------------------------
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
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


ALGO_ORDER = ["KM", "GSKM"]
ALGO_COLOR = {
    "KM":   "#4d4d4d",  # neutral gray
    "GSKM": "#d62728",  # standout red (your proposed)
}
ALGO_MARKER = {"KM": "o", "GSKM": "s"}


# -----------------------------
# Data collection
# -----------------------------
def _extract_km_gskm(metrics_dict: Dict[str, Tuple[float, float, float, float, float]]):
    """
    metrics_dict: results from run_comparison()
      - "K-Means": (mse, tmse, gain_err, cos_err, sil)
      - "ABC":     (mse, tmse, gain_err, cos_err, sil)
    returns:
      {"KM": {...}, "GSKM": {...}}
    """
    out = {}
    for k_src, k_dst in [("K-Means", "KM"), ("ABC", "GSKM")]:
        if k_src not in metrics_dict:
            raise KeyError(f"run_comparison() result missing key: '{k_src}'. keys={list(metrics_dict.keys())}")
        mse, tmse, gain_err, cos_err, sil = metrics_dict[k_src]
        out[k_dst] = {
            "MSE": float(mse),
            "GainError": float(gain_err),
            "CosineSimilarity": float(1.0 - float(cos_err)),  # 핵심 변환
        }
    return out


def run_dimension_sweep_normal(
    N: int,
    dims: List[int],
    K_fixed: int,
    trim_ratio: float,
    seed: int,
) -> pd.DataFrame:
    import torch

    rows = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    for D in dims:
        print(f"[D-sweep] D={D}, K={K_fixed}")
        data = torch.randn(N, D, generator=gen)  # Normal only (lognormal removed)
        res = run_comparison(data, num_clusters=K_fixed, trim_ratio=trim_ratio)
        m = _extract_km_gskm(res)

        for algo in ALGO_ORDER:
            rows.append({
                "D": D,
                "Algorithm": algo,
                **m[algo],
            })

    return pd.DataFrame(rows)


def run_cluster_sweep_normal(
    N: int,
    D_fixed: int,
    Ks: List[int],
    trim_ratio: float,
    seed: int,
) -> pd.DataFrame:
    import torch

    rows = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    print(f"[K-sweep] generate once: Normal N={N}, D={D_fixed}")
    data = torch.randn(N, D_fixed, generator=gen)

    for K in Ks:
        print(f"[K-sweep] K={K}, D={D_fixed}")
        res = run_comparison(data, num_clusters=K, trim_ratio=trim_ratio)
        m = _extract_km_gskm(res)

        for algo in ALGO_ORDER:
            rows.append({
                "K": K,
                "Algorithm": algo,
                **m[algo],
            })

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------
def _set_log2_x(ax, xticks: List[int]):
    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        ax.set_xscale("log", basex=2)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.minorticks_off()


def _legend_lines_only(ax):
    """Legend에서 marker를 제거해 선이 더 잘 보이게."""
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        try:
            new_handles.append(Line2D(
                [0], [0],
                color=h.get_color(),
                linestyle=h.get_linestyle(),
                linewidth=h.get_linewidth(),
                marker=None,
            ))
        except Exception:
            new_handles.append(h)
    ax.legend(
        new_handles, labels,
        loc="best",
        frameon=True,
        handlelength=3.0,
        handletextpad=0.6,
        borderpad=0.4,
    )


def plot_metric(
    df: pd.DataFrame,
    x_col: str,
    metric_col: str,
    x_label: str,
    y_label: str,
    title: str,
    xticks: List[int],
    out_pdf: str,
):
    fig = plt.figure(figsize=(3.4, 2.4))
    ax = fig.add_subplot(111)

    for algo in ALGO_ORDER:
        dd = df[df["Algorithm"] == algo].copy()
        dd = dd.sort_values(x_col)

        ax.plot(
            dd[x_col].to_numpy(),
            dd[metric_col].to_numpy(),
            color=ALGO_COLOR[algo],
            marker=ALGO_MARKER[algo],
            linestyle="-",
            label=algo,
        )

    _set_log2_x(ax, xticks=xticks)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    _legend_lines_only(ax)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--N", type=int, default=10000)

    # D-sweep
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--K_fixed", type=int, default=2048)

    # K-sweep
    ap.add_argument("--D_fixed", type=int, default=256)
    ap.add_argument("--Ks", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    ap.add_argument("--trim_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_icml_style()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()

    # ------------------ D sweep (Normal only) ------------------
    dfD = run_dimension_sweep_normal(
        N=args.N,
        dims=args.dims,
        K_fixed=args.K_fixed,
        trim_ratio=args.trim_ratio,
        seed=args.seed,
    )

    # Save 3 PDFs: D-sweep
    plot_metric(
        df=dfD, x_col="D", metric_col="MSE",
        x_label="D", y_label="MSE",
        title=f"Normal · D-sweep (K={args.K_fixed})",
        xticks=args.dims,
        out_pdf=os.path.join(args.out_dir, f"dsweep_MSE_K{args.K_fixed}.pdf"),
    )
    plot_metric(
        df=dfD, x_col="D", metric_col="GainError",
        x_label="D", y_label="Gain error",
        title=f"Normal · D-sweep (K={args.K_fixed})",
        xticks=args.dims,
        out_pdf=os.path.join(args.out_dir, f"dsweep_GainError_K{args.K_fixed}.pdf"),
    )
    plot_metric(
        df=dfD, x_col="D", metric_col="CosineSimilarity",
        x_label="D", y_label="Cosine similarity",
        title=f"Normal · D-sweep (K={args.K_fixed})",
        xticks=args.dims,
        out_pdf=os.path.join(args.out_dir, f"dsweep_CosineSimilarity_K{args.K_fixed}.pdf"),
    )

    # ------------------ K sweep (Normal only) ------------------
    dfK = run_cluster_sweep_normal(
        N=args.N,
        D_fixed=args.D_fixed,
        Ks=args.Ks,
        trim_ratio=args.trim_ratio,
        seed=args.seed,
    )

    # Save 3 PDFs: K-sweep
    plot_metric(
        df=dfK, x_col="K", metric_col="MSE",
        x_label="K", y_label="MSE",
        title=f"Normal · K-sweep (D={args.D_fixed})",
        xticks=args.Ks,
        out_pdf=os.path.join(args.out_dir, f"ksweep_MSE_D{args.D_fixed}.pdf"),
    )
    plot_metric(
        df=dfK, x_col="K", metric_col="GainError",
        x_label="K", y_label="Gain error",
        title=f"Normal · K-sweep (D={args.D_fixed})",
        xticks=args.Ks,
        out_pdf=os.path.join(args.out_dir, f"ksweep_GainError_D{args.D_fixed}.pdf"),
    )
    plot_metric(
        df=dfK, x_col="K", metric_col="CosineSimilarity",
        x_label="K", y_label="Cosine similarity",
        title=f"Normal · K-sweep (D={args.D_fixed})",
        xticks=args.Ks,
        out_pdf=os.path.join(args.out_dir, f"ksweep_CosineSimilarity_D{args.D_fixed}.pdf"),
    )

    print(f"[DONE] total time: {time.time() - t0:.2f}s")
    print("\n[D-sweep head]\n", dfD.head())
    print("\n[K-sweep head]\n", dfK.head())


if __name__ == "__main__":
    main()

