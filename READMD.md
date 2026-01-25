# Gain-Shape K-Means (GSKM) Analysis & Visualization

[cite_start]This repository contains the analysis scripts and experimental logs required to reproduce the key figures from the ICML 2026 paper: **"Gain-Shape Reparameterized K-Means: Toward Sub-1-bit KV-Cache"**

[cite_start]By running the provided scripts, you can visualize the centroid shrinkage phenomenon and the effectiveness of GSKM compared to standard K-Means.

## 📂 Repository Structure

The repository is organized as follows:

```
text
.
├── clustering_results/           # Directory containing pre-computed JSON logs
│   ├── clustering_comparison_report_value_dim_8_K1024.json
│   ├── clustering_comparison_report_value_dim_8_K1024_residual.json
│   ├── clustering_comparison_report_value_dim_8_K256.json
│   ├── clustering_comparison_report_value_dim_8_K256_residual.json
│   └── ...
├── plot_clustering.py            # Script to reproduce Figure 4 (KV-Cache Reconstruction)
├── plot_clustering_random.py     # Script to reproduce Figure 3 (Random Gaussian Sweeps)
└── README.md
```


## 🛠️ Prerequisites

The scripts are written in Python. To run the visualization code, you will need the following dependencies:
Python 3.x
Matplotlib
NumPy
You can install the necessary packages using pip:

```Bash
pip install matplotlib numpy
```

## 📊 Reproducing Figures

### 1. Figure 3: Random Gaussian Sweeps

Goal: Visualize the impact of high-dimensional averaging cancellation on standard K-Means vs. GSKM using synthetic data. This corresponds to the dimension ($D$) and capacity ($K$) sweeps discussed in the paper.

To generate the plots for MSE, Gain Error, and Cosine Similarity (Figure 3a-f):

```Bash
python plot_clustering_random.py
```

### 2. Figure 4: Llama-3-8B KV-Cache Reconstruction

Goal: Evaluate reconstruction performance on real KV-cache activations (Wikitext-2) for both the original signal and the first residual stage.

To generate the plots for Key/Value Original and Key/Value 1st Residual (Figure 4a-l), run the script specifying the input data directory and the output directory for figures:

```Bash
python plot_clustering.py --input_dir clustering_results --output_dir output_figures
```

This script reads the JSON logs from the ```clustering_results/``` directory and saves the generated plots to ```output_figures/```.

* Output: This script processes the JSON files in clustering_results/ and generates the plots for Key/Value Original and Key/Value 1st Residual (Figure 4a-l).

## 📄 Data Format

The ```clustering_results/``` directory contains JSON files following this naming convention: ```clustering_comparison_report_{type}_dim_{D}_K{K}_```{residual}.json

Each file records the following metrics used in the paper:

* MSE: Mean Squared Error ($\frac{1}{N}\sum \|x - \hat{x}\|_2^2$).
* Gain Error: Magnitude difference ($\mathbb{E}[|\|x\|_2 - \|\hat{x}\|_2 |]$).
* Cosine Similarity: Directional alignment ($\mathbb{E}[\cos(x, \hat{x})]$).

## 📝 Citation

If you find this code or our paper useful for your research, please cite.