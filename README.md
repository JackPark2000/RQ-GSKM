# Gain-Shape Reparameterized K-Means: Toward Sub 1-bit KV-Cache

This repository contains the analysis scripts and experimental logs required to reproduce the key figures from the ICML 2026 paper: **"Gain-Shape Reparameterized K-Means: Toward Sub-1-bit KV-Cache"**

By running the provided scripts, you can visualize the centroid shrinkage phenomenon and the effectiveness of GSKM compared to standard K-Means.

<br>

## 📂 Repository Structure

The repository is organized as follows:

```
├── clustering.py                 # Core logic for K-Means/GSKM algorithms on real data
├── plot_clustering.py            # Visualization script for Figure 4 (Real KV Cache)
├── plot_clustering_random.py     # Simulation & Visualization for Figure 3 (Synthetic Data)
├── run_clustering.sh             # Shell script to run clustering experiments
├── run_duming.sh                 # Shell script to dump calibration set of Llama3-8b with Wikitext-2s
└── README.md
```

<br>

## 🛠️ Prerequisites

The code relies on PyTorch, CuPy, and RAPIDS cuML for GPU-accelerated clustering. To run the scripts, ensure you have a CUDA-enabled environment.

Core Dependencies:
* Python 3.8+
* PyTorch (GPU support)
* CuPy
* RAPIDS cuML (for cuml.cluster.KMeans)
* Matplotlib, NumPy, Pandas

<br>

## 📊 Reproducing Figures

### 1. Figure 3: Random Gaussian Sweeps

Visualize the impact of high-dimensional averaging cancellation on standard K-Means vs. GSKM using synthetic data. This corresponds to the dimension ($D$) and capacity ($K$) sweeps discussed in the paper.

The script plot_clustering_random.py generates the synthetic data internally, runs the algorithms, and produces the plots in one go.

```Bash
python plot_clustering_random.py --out_dir output_figures_random --dims 4 8 16 32 64 128 256 512 1024 --Ks 64 128 256 512 1024 2048 4096 8192 16384
```
* Output: Creates PDF files in output_figures_random/ (e.g., dsweep_CosineSimilarity_K2048.pdf).


### 2. Figure 4: Llama-3-8B KV-Cache Reconstruction

Evaluate reconstruction performance on real KV-cache activations (Wikitext-2) for both the original signal and the first residual stage.

**Step 1: Generate Experimental Data**
First, you need to extract clustering metrics from the Llama-3-8B model. We provide a shell script that automatically runs `llama_wiki_dump.py` across all required dimensions ($D \in \{8, 32, 128, 512\}$).

```bash
bash run_dumping.sh
```

**Step 2: Run Clustering Analysis Next, run the clustering algorithms (Standard K-Means vs. GSKM) on the extracted data. This script executes clustering.py for various configurations and generates the JSON metric logs in the clustering_results/ directory.**

```Bash
bash run_clustering.sh
```

**Step 3: Visualize Results Finally, generate the figure using the computed logs:**

```Bash
python plot_clustering.py --input_dir clustering_results --output_dir output_figures --include_residual 1
```

* ```--input_dir```: Directory containing the JSON logs.

* ```--output_dir```: Directory to save the resulting PDF/PNG figures.

* ```--include_residual```: Set to 1 to plot original and residual curves separately.

* ```--Ks```: Codebook sizes to plot (default: 256 1024).

* Output: This will generate the plots for Key/Value Original and Key/Value 1st Residual (Figure 4 a-l).

<br>

## 📄 Data Format

The ```clustering_results/``` directory contains JSON files following this naming convention: ```clustering_comparison_report_{type}_dim_{D}_K{K}_```{residual}.json

Each file records the following metrics used in the paper:

* MSE: Mean Squared Error ($\frac{1}{N}\sum \|x - \hat{x}\|_2^2$).
* Gain Error: Magnitude difference ($\mathbb{E}[|\|x\|_2 - \|\hat{x}\|_2 |]$).
* Cosine Similarity: Directional alignment ($\mathbb{E}[\cos(x, \hat{x})]$).

