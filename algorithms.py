#!/usr/bin/env python3
"""
Core algorithms for comparing Standard K-Means (KM) vs Gain-Shape K-Means (GSKM).

This module implements:
  - Standard K-Means (KM): Lloyd's algorithm via cuML GPU acceleration
  - Gain-Shape K-Means (GSKM): Alternating gain-shape optimization with
    spherical K-Means initialization

Paper: GSRQ: Gain-Shape Residual Quantization for Sub-1-bit KV Cache
"""

import math

import cupy as cp
import torch
import torch.nn.functional as F
import torch.utils.dlpack as dlpack
from cuml.cluster import KMeans as cuKMeans
from cuml.metrics.cluster import silhouette_score


# ---------------------------------------------------------------------------
# Torch <-> CuPy conversion
# ---------------------------------------------------------------------------

def torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    """Convert a CUDA torch tensor to a CuPy array via DLPack (zero-copy)."""
    if not x.is_cuda:
        raise ValueError("torch_to_cupy expects a CUDA tensor.")
    return cp.from_dlpack(dlpack.to_dlpack(x.contiguous()))


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def trimmed_mse_by_gain(data_cpu, recon_cpu, trim_ratio=0.03):
    """Trimmed MSE: discard top-gain samples before computing MSE."""
    g = torch.norm(data_cpu, dim=1)
    thr = torch.quantile(g, 1.0 - trim_ratio)
    keep = g <= thr
    if keep.sum() == 0:
        return float("nan")
    return (recon_cpu[keep] - data_cpu[keep]).pow(2).mean().item()


def avg_gain_error(data_cpu, recon_cpu):
    r"""Average gain error: :math:`\mathbb{E}[\,|\;\|x\| - \|\hat{x}\|\;|\,]`."""
    with torch.no_grad():
        g = torch.norm(data_cpu.float(), dim=1)
        gh = torch.norm(recon_cpu.float(), dim=1)
        return (g - gh).abs().mean().item()


def avg_cosine_similarity_error(data_cpu, recon_cpu, eps=1e-12):
    r"""Average cosine similarity error: :math:`\mathbb{E}[1 - \cos(x, \hat{x})]`."""
    with torch.no_grad():
        x = data_cpu.float()
        y = recon_cpu.float()
        x_n = x / (torch.norm(x, dim=1, keepdim=True) + eps)
        y_n = y / (torch.norm(y, dim=1, keepdim=True) + eps)
        cos = (x_n * y_n).sum(dim=1).clamp(-1.0, 1.0)
        return (1.0 - cos).mean().item()


def evaluate_metrics(data, recon, labels, sil_features=None,
                     max_samples=10000, trim_ratio=0.01):
    """
    Compute reconstruction quality metrics.

    Returns:
        (mse, trimmed_mse, avg_gain_error, avg_cosine_sim_error, silhouette)
    """
    data_cpu = data.detach().cpu().float()
    recon_cpu = recon.detach().cpu().float()

    mse = F.mse_loss(recon_cpu, data_cpu).item()
    tmse = trimmed_mse_by_gain(data_cpu, recon_cpu, trim_ratio=trim_ratio)
    gain_err = avg_gain_error(data_cpu, recon_cpu)
    cos_sim_err = avg_cosine_similarity_error(data_cpu, recon_cpu)

    # Silhouette score (cosine metric)
    feats = sil_features if sil_features is not None else data
    feats = feats.detach()
    labels_t = labels.detach() if torch.is_tensor(labels) else torch.tensor(labels)
    labels_t = labels_t.to(feats.device, non_blocking=True)

    n = feats.shape[0]
    if n > max_samples:
        idx = torch.randperm(n, device=feats.device)[:max_samples]
        feats_s = feats.index_select(0, idx)
        labels_s = labels_t.index_select(0, idx)
    else:
        feats_s, labels_s = feats, labels_t

    feats_s = feats_s.to("cuda", non_blocking=True).float()
    labels_s = labels_s.to("cuda", non_blocking=True).int()

    try:
        sil = float(silhouette_score(
            torch_to_cupy(feats_s), torch_to_cupy(labels_s), metric="cosine"
        ))
    except Exception:
        sil = -1.0

    return mse, tmse, gain_err, cos_sim_err, sil


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def _compute_avg_gain_norm(data_gpu, label_gpu, num_entries):
    """Per-cluster average gain: E[||x||] for each cluster k."""
    sum_gain = torch.zeros(num_entries, device=data_gpu.device)
    counts = torch.zeros(num_entries, device=data_gpu.device)
    sample_gain = torch.norm(data_gpu, p=2, dim=-1)
    sum_gain.scatter_add_(0, label_gpu, sample_gain)
    counts.scatter_add_(0, label_gpu, torch.ones_like(sample_gain))
    return sum_gain / torch.where(counts > 0, counts, torch.ones_like(counts))


# ---------------------------------------------------------------------------
# KM vs GSKM comparison
# ---------------------------------------------------------------------------

def run_comparison(data_cpu, weights_cpu=None, num_clusters=1024,
                   trim_ratio=0.03, abc_iters=10, return_km_recon=False):
    """
    Run Standard K-Means (KM) and Gain-Shape K-Means (GSKM) on the input data.

    Args:
        data_cpu:         [N, D] tensor on CPU.
        weights_cpu:      Optional [N] sample weights on CPU.
        num_clusters:     Number of clusters K.
        trim_ratio:       Outlier trimming ratio for trimmed MSE.
        abc_iters:        Number of GSKM alternating optimization iterations.
        return_km_recon:  If True, also return KM reconstruction for residual
                          computation.

    Returns:
        results: dict mapping algorithm name to metric tuple
            ``{"K-Means": (mse, tmse, gain_err, cos_err, sil),
               "ABC":     (mse, tmse, gain_err, cos_err, sil)}``
        km_recon: (only when *return_km_recon=True*) KM reconstruction [N, D]
                  on CPU for computing residual vectors.
    """
    device = torch.device("cuda")
    data_gpu = data_cpu.to(device).float()
    data_norm_gpu = F.normalize(data_gpu, p=2, dim=-1)

    weight_cp = None
    if weights_cpu is not None:
        weight_cp = torch_to_cupy(weights_cpu.to(device).float())

    results = {}

    # ---- Standard K-Means (KM) ----
    km = cuKMeans(n_clusters=num_clusters, max_iter=20, random_state=42)
    km.fit(torch_to_cupy(data_gpu))
    labels_km = torch.from_numpy(km.labels_.get()).to(device).long()
    centroids_km = torch.from_numpy(km.cluster_centers_.get()).to(device).float()
    recon_km = centroids_km[labels_km]
    results["K-Means"] = evaluate_metrics(
        data_cpu, recon_km, labels_km, trim_ratio=trim_ratio,
    )

    # ---- Gain-Shape K-Means (GSKM) ----
    # Step 1: Spherical K-Means for initial shape directions
    km_sph = cuKMeans(n_clusters=num_clusters, max_iter=20, random_state=42)
    km_sph.fit(torch_to_cupy(data_norm_gpu), sample_weight=weight_cp)
    labels_sph = torch.from_numpy(km_sph.labels_.get()).to(device).long()
    centroids_sph = torch.from_numpy(km_sph.cluster_centers_.get()).to(device).float()
    shape = F.normalize(centroids_sph, p=2, dim=-1)

    # Step 2: Initialize gain as per-cluster average norm
    raw_gain = _compute_avg_gain_norm(
        data_gpu, labels_sph, num_clusters,
    ).clamp(min=1e-4).cpu()

    # Step 3: Alternating optimization
    labels_abc = None
    for _ in range(abc_iters):
        gain_q = raw_gain.to(device).to(torch.bfloat16).float()
        projections = torch.matmul(data_gpu, shape.t())
        scores = 2 * projections * gain_q.unsqueeze(0) - gain_q.unsqueeze(0).pow(2)
        labels_abc = torch.argmax(scores, dim=1)

        # Update shape directions
        new_shape = torch.zeros_like(shape)
        new_shape.scatter_add_(
            0,
            labels_abc.unsqueeze(1).expand(-1, shape.shape[1]),
            data_gpu * gain_q[labels_abc].unsqueeze(1),
        )
        shape = F.normalize(new_shape, p=2, dim=-1)

        # Update gains
        proj_assigned = projections.gather(1, labels_abc.unsqueeze(1)).squeeze(1)
        new_gain = torch.zeros_like(gain_q)
        cnts = torch.zeros_like(gain_q)
        new_gain.scatter_add_(0, labels_abc, proj_assigned)
        cnts.scatter_add_(0, labels_abc, torch.ones_like(proj_assigned))
        raw_gain.data.copy_(
            torch.where(cnts > 0, new_gain / cnts, gain_q).clamp(min=1e-4).cpu()
        )

    recon_abc = shape[labels_abc] * raw_gain.to(device)[labels_abc].unsqueeze(1)
    results["ABC"] = evaluate_metrics(
        data_cpu, recon_abc, labels_abc,
        sil_features=data_norm_gpu, trim_ratio=trim_ratio,
    )

    if return_km_recon:
        return results, recon_km.detach().cpu().float()
    return results
