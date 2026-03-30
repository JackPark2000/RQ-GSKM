# GSRQ: Gain-Shape Residual Quantization for Sub-1-bit KV Cache

This repository provides the scripts to reproduce **Figure 4** (synthetic Gaussian experiments) and **Figure 5** (Llama-3-8B KV-cache reconstruction) from the paper:

> **GSRQ: Gain-Shape Residual Quantization for Sub-1-bit KV Cache**

Both figures compare **Standard K-Means (KM)** against **Gain-Shape K-Means (GSKM)**, demonstrating the centroid shrinkage phenomenon in high-dimensional clustering and the effectiveness of gain-shape reparameterization.

<br>

## Repository Structure

```
.
├── algorithms.py          # Core KM & GSKM implementations + evaluation metrics
├── dump_kvcache.py        # Extract KV-cache activations from Llama-3-8B (Wikitext-2)
├── dump_kvcache.sh        # Shell script to dump all (dim, key/value) configurations
├── run_clustering.py      # Run KM vs GSKM on dumped KV-cache data → JSON reports
├── run_clustering.sh      # Shell script to batch-run clustering experiments
├── plot_synthetic.py      # Figure 4: Synthetic Gaussian D-sweep & K-sweep
├── plot_kvcache.py        # Figure 5: Llama-3-8B KV-cache reconstruction
├── clustering_results/    # Pre-computed clustering results (JSON) for Figure 5
├── requirements.txt       # Python dependencies
└── README.md
```

<br>

## Prerequisites

```bash
conda create -n gsrq python=3.11 -y
conda activate gsrq
```

This is sufficient to generate **Figure 5** (KV-cache) from the pre-computed results included in this repository. For **Figure 4** (synthetic) and re-running clustering from scratch, see the additional setup in the sections below.

<br>

## Reproducing Figures

### Figure 4: Synthetic Gaussian Sweeps

Visualize the centroid shrinkage phenomenon on synthetic data. Requires a **CUDA GPU** and additional dependencies:

```bash
# Additional setup (CUDA GPU required)
conda install -y -c pytorch -c nvidia -c rapidsai -c conda-forge pytorch torchvision torchaudio pytorch-cuda=12.1 cuml=24.04 cupy pandas numpy matplotlib cuda-version=12.1
```

Then run:

```bash
python plot_synthetic.py \
    --out_dir figures/synthetic \
    --dims 4 8 16 32 64 128 256 512 1024 \
    --Ks 64 128 256 512 1024 2048 4096 8192 16384
```

**Output:** `figures/synthetic/dsweep_*.pdf` and `figures/synthetic/ksweep_*.pdf` (Figure 4a-f)


### Figure 5: Llama-3-8B KV-Cache Reconstruction

Pre-computed clustering results are included in `clustering_results/`, so you can generate the plots immediately after cloning:

```bash
python plot_kvcache.py \
    --input_dir clustering_results \
    --output_dir figures/kvcache \
    --Ks 256 1024 \
    --include_residual 1
```

| Argument | Description |
|---|---|
| `--input_dir` | Directory containing JSON reports |
| `--output_dir` | Directory to save PDF/PNG figures |
| `--Ks` | Codebook sizes to overlay (default: 256, 1024) |
| `--include_residual` | Set to `1` to plot original and 1st residual separately |

**Output:** Plots for Key/Value Original and Key/Value 1st Residual (Figure 5a-l).

<details>
<summary><b>Re-running from scratch (optional)</b></summary>

If you want to reproduce the clustering results yourself, install the additional dependencies first (if not already done for Figure 4):

```bash
conda install -c rapidsai -c conda-forge -c nvidia cuml cupy rapids-dask-dependency -y
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install transformers datasets accelerate
```

> **Note:** Llama-3-8B requires access approval on [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B). Run `huggingface-cli login` before dumping KV-cache data.

**Step 1.** Dump KV-cache activations for all required dimensions:

```bash
bash dump_kvcache.sh
```

This runs `dump_kvcache.py` for $D \in \{8, 32, 128, 512\}$ and both key/value caches. Each run loads the model, performs a forward pass on Wikitext-2, and saves the D-dimensional sub-vectors to `dumped_data_dim_{D}/`. You can also run individual configurations:

```bash
python dump_kvcache.py --dim 128 --kv key
python dump_kvcache.py --dim 128 --kv value --max_length 4096
```

**Step 2.** Run clustering experiments across all configurations:

```bash
bash run_clustering.sh
```

This executes `run_clustering.py` for all $D \in \{8, 32, 128, 512\}$, $K \in \{256, 1024\}$, and both key/value caches. Results (including the 1st residual stage) are saved as JSON files in `clustering_results/`.

**Step 3.** Generate the plots using the command above.

</details>

<br>

## Metrics

The following reconstruction metrics are reported:

| Metric | Formula |
|---|---|
| **MSE** | $\frac{1}{N}\sum \|x - \hat{x}\|_2^2$ |
| **Gain Error** | $\mathbb{E}\bigl[\|\|x\|_2 - \|\hat{x}\|_2\|\bigr]$ |
| **Cosine Similarity** | $\mathbb{E}\bigl[\cos(x, \hat{x})\bigr]$ |
