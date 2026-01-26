import sys
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

# --------------------------------------------------------------------------------
# 0. Torch <-> CuPy 유틸
# --------------------------------------------------------------------------------

def torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    """CUDA torch tensor -> CuPy (DLPack, zero-copy)"""
    if not x.is_cuda:
        raise ValueError("torch_to_cupy expects a CUDA tensor.")
    return cp.from_dlpack(dlpack.to_dlpack(x.contiguous()))

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

def _robust_z(x: torch.Tensor, eps: float = 1e-12):
    """robust z-score based on median & MAD"""
    med = x.median()
    mad = (x - med).abs().median().clamp_min(eps)
    z = 0.6745 * (x - med) / mad
    return z, med, mad

def _quantiles(x: torch.Tensor, qs=(0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)):
    qv = torch.quantile(x, torch.tensor(qs, device=x.device))
    return {str(q): float(v) for q, v in zip(qs, qv)}

def _sample_indices(n: int, m: int, device="cpu"):
    """큰 N에서 randperm 비용 줄이기용 샘플링"""
    if m >= n:
        return torch.arange(n, device=device)
    if n > 5 * m:
        return torch.randint(low=0, high=n, size=(m,), device=device)
    else:
        return torch.randperm(n, device=device)[:m]

def analyze_data_outliers(
    data_cpu: torch.Tensor,
    tag: str,
    max_samples: int = 200_000,
    z_thresh: float = 6.0,
    topk: int = 8,
):
    """gain/direction/spiky outlier 요약"""
    with torch.no_grad():
        n, d = data_cpu.shape
        idx = _sample_indices(n, min(max_samples, n), device="cpu")
        x = data_cpu.index_select(0, idx).to("cuda", non_blocking=True).float()
        m = x.shape[0]

        gain = torch.norm(x, p=2, dim=-1)
        u = F.normalize(x, p=2, dim=-1)

        zg, g_med, g_mad = _robust_z(gain)
        gain_out = (zg.abs() > z_thresh)

        mean_dir = F.normalize(u.mean(dim=0, keepdim=True), p=2, dim=-1)
        cos = (u * mean_dir).sum(dim=-1).clamp(-1.0, 1.0)
        dist = 1.0 - cos
        zd, d_med, d_mad = _robust_z(dist)
        dir_out = (zd > z_thresh)

        max_abs = u.abs().amax(dim=-1)
        zs, s_med, s_mad = _robust_z(max_abs)
        spiky_out = (zs > z_thresh)

        a, b, c = gain_out, dir_out, spiky_out
        only_gain = a & ~b & ~c
        only_dir = ~a & b & ~c
        only_spiky = ~a & ~b & c
        gain_and_dir = a & b
        dir_and_spiky = b & c
        gain_and_spiky = a & c
        all_three = a & b & c

        def pct(mask): return 100.0 * mask.float().mean().item()

        def topk_info(score, name):
            k = min(topk, m)
            vals, ii = torch.topk(score, k=k, largest=True)
            orig_idx = idx[ii.cpu()].tolist()
            return {
                "name": name,
                "topk_score": [float(v) for v in vals.detach().cpu()],
                "topk_sample_idx_in_original": orig_idx,
            }

        report = {
            "tag": tag,
            "N_total": int(n),
            "D": int(d),
            "sampled_M": int(m),

            "gain_stats": {
                "mean": float(gain.mean().item()),
                "std": float(gain.std().item()),
                "median": float(g_med.item()),
                "mad": float(g_mad.item()),
                "quantiles": _quantiles(gain),
            },
            "dir_stats": {
                "dist_mean": float(dist.mean().item()),
                "dist_std": float(dist.std().item()),
                "dist_median": float(d_med.item()),
                "dist_mad": float(d_mad.item()),
                "dist_quantiles": _quantiles(dist),
                "cos_quantiles": _quantiles(cos),
            },
            "spiky_stats": {
                "maxabs_mean": float(max_abs.mean().item()),
                "maxabs_std": float(max_abs.std().item()),
                "maxabs_median": float(s_med.item()),
                "maxabs_mad": float(s_mad.item()),
                "maxabs_quantiles": _quantiles(max_abs),
            },

            "outlier_rates_percent": {
                "gain_out": pct(a),
                "dir_out": pct(b),
                "spiky_out": pct(c),
                "only_gain": pct(only_gain),
                "only_dir": pct(only_dir),
                "only_spiky": pct(only_spiky),
                "gain_and_dir": pct(gain_and_dir),
                "dir_and_spiky": pct(dir_and_spiky),
                "gain_and_spiky": pct(gain_and_spiky),
                "all_three": pct(all_three),
            },

            "topk_examples": {
                "gain": topk_info(zg.abs(), "robust_z(|x|)"),
                "direction": topk_info(dist, "1 - cos(u, mean_dir)"),
                "spiky": topk_info(max_abs, "max(|u_i|)"),
            }
        }

        print("\n" + "=" * 80)
        print(f"[Outlier Analysis] {tag}  (sampled {m}/{n}, z_thresh={z_thresh})")
        r = report["outlier_rates_percent"]
        print(f"- Outlier rates (%): gain={r['gain_out']:.3f}, dir={r['dir_out']:.3f}, spiky={r['spiky_out']:.3f}")
        print(f"  * only_gain={r['only_gain']:.3f}, only_dir={r['only_dir']:.3f}, only_spiky={r['only_spiky']:.3f}")
        print("=" * 80 + "\n")
        return report

# --------------------------------------------------------------------------------
# 2. Spherical gain 계산 variants
# --------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------
# 3. 통합 비교 실행 함수
# --------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------
# 4. Long-tail random generator (LogNormal gain)
# --------------------------------------------------------------------------------

def sample_unit_vectors(N, D, device="cpu", seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    u = torch.randn(N, D, generator=g, device=device)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-12)
    return u

def gen_longtail_gain_lognormal(N, D, mu=math.log(0.5), sigma=0.6, device="cpu", seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed + 123)
    u = sample_unit_vectors(N, D, device=device, seed=seed)
    r = torch.exp(mu + sigma * torch.randn(N, generator=g, device=device))
    x = u * r.unsqueeze(1)
    return x, r

# --------------------------------------------------------------------------------
# 5. 메인 로직
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    dim = int(sys.argv[1])
    key_or_value = sys.argv[2]
    file_pattern = "dumped_data_dim_%d/raw_data_layer_*_%s_cb*.pt" % (dim, key_or_value)
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        exit()

    trim_ratio = 0.01  # gain 기준 trim 비율

    methods = [
        "K-Means",
        "LogGain-Assign KMeans",
        "Spherical-Only",
        "Spherical",
        "Spherical-ProjGain",
        "S-Init K-Means",
        "ABC",
    ]

    summary = {}
    outlier_summary = {}
    method_totals = {m: {"mse": 0.0, "tmse": 0.0, "gain_err": 0.0, "cos_sim_err": 0.0, "sil": 0.0, "count_sil": 0} for m in methods}

    # -------------------------------
    # [Part 1] 실제 데이터 평가
    # -------------------------------
    for f in files:
        fname = os.path.basename(f)
        payload = torch.load(f)
        data = payload["data"].float()

        outlier_summary[fname] = analyze_data_outliers(
            data_cpu=data,
            tag=f"Target::{fname}",
            max_samples=200_000,
            z_thresh=6.0,
            topk=8,
        )

        res = run_comparison(data, payload.get("weights", None), trim_ratio=trim_ratio)
        summary[fname] = res

        print(f"\nResults for {fname}:")
        for m in methods:
            mse, tmse, gain_err, cos_sim_err, sil = res[m]
            print(f"  - {m:20s}: MSE={mse:.6f}, tMSE@1%={tmse:.6f}, "
                  f"AvgGainErr={gain_err:.6f}, AvgCosSimErr={cos_sim_err:.6f}, Sil={sil:.4f}")
            method_totals[m]["mse"] += mse
            method_totals[m]["tmse"] += tmse
            method_totals[m]["gain_err"] += gain_err
            method_totals[m]["cos_sim_err"] += cos_sim_err
            if sil != -1.0:
                method_totals[m]["sil"] += sil
                method_totals[m]["count_sil"] += 1

    # -------------------------------
    # [Part 2] Random 컨트롤 2종
    # -------------------------------
    print("\n" + "=" * 60)
    print(f"{'RANDOM DATASET EVALUATION (Normal Dist)':^60s}")
    print("=" * 60)

    sample_data = payload["data"].float()
    random_data = torch.randn_like(sample_data)

    outlier_summary["__random_normal__"] = analyze_data_outliers(
        data_cpu=random_data,
        tag="Random::Normal",
        max_samples=200_000,
        z_thresh=6.0,
        topk=8,
    )

    rand_res = run_comparison(random_data, tag="Random-Normal", trim_ratio=trim_ratio)
    for m in methods:
        mse, tmse, gain_err, cos_sim_err, sil = rand_res[m]
        print(f"  - {m:20s}: MSE={mse:.6f}, tMSE@1%={tmse:.6f}, "
              f"AvgGainErr={gain_err:.6f}, AvgCosSimErr={cos_sim_err:.6f}, Sil={sil:.4f}")

    # (신규) Long-tail random (LogNormal gain)
    print("\n" + "=" * 60)
    print(f"{'RANDOM DATASET EVALUATION (LogNormal Gain Long-tail)':^60s}")
    print("=" * 60)

    random_out, _ = gen_longtail_gain_lognormal(
        random_data.shape[0], random_data.shape[1],
        mu=math.log(0.5), sigma=0.6, device="cpu", seed=0
    )


    outlier_summary["__random_longtail__"] = analyze_data_outliers(
        data_cpu=random_out,
        tag="Random::LongTail(LogNormal)",
        max_samples=200_000,
        z_thresh=6.0,
        topk=8,
    )

    rand_out_res = run_comparison(random_out, tag="Random-LongTail", trim_ratio=trim_ratio)
    for m in methods:
        mse, tmse, gain_err, cos_sim_err, sil = rand_out_res[m]
        print(f"  - {m:20s}: MSE={mse:.6f}, tMSE@1%={tmse:.6f}, "
              f"AvgGainErr={gain_err:.6f}, AvgCosSimErr={cos_sim_err:.6f}, Sil={sil:.4f}")

    # -------------------------------
    # [Part 3] 전체 평균 출력
    # -------------------------------
    print("\n" + "=" * 60)
    print(f"{'OVERALL AVERAGES (Target Data)':^60s}")
    print("=" * 60)

    averages = {}
    for m in methods:
        n_files = len(files)
        avg_mse = method_totals[m]["mse"] / n_files
        avg_tmse = method_totals[m]["tmse"] / n_files
        avg_gain_err = method_totals[m]["gain_err"] / n_files
        avg_cos_sim_err = method_totals[m]["cos_sim_err"] / n_files

        n_sil = method_totals[m]["count_sil"]
        avg_sil = method_totals[m]["sil"] / n_sil if n_sil > 0 else -1.0

        averages[m] = {
            "avg_mse": avg_mse,
            "avg_tmse_1pct": avg_tmse,
            "avg_gain_err": avg_gain_err,
            "avg_cos_sim_err": avg_cos_sim_err,
            "avg_sil": avg_sil
        }

        print(f"  - {m:20s}: Avg MSE={avg_mse:.6f}, Avg tMSE@1%={avg_tmse:.6f}, "
              f"AvgGainErr={avg_gain_err:.6f}, AvgCosSimErr={avg_cos_sim_err:.6f}, Avg Sil={avg_sil:.4f}")

    report = {
        "trim_ratio": trim_ratio,
        "averages": averages,
        "random_baseline": {
            "normal": rand_res,
            "longtail": rand_out_res,
        },
        "details": summary,
        "outlier_analysis": outlier_summary,
    }

    with open("clustering_comparison_report_%s_dim_%d.json" % (key_or_value, dim), "w") as jf:
        json.dump(report, jf, indent=4)

    print("\nSaved: clustering_comparison_report.json")

