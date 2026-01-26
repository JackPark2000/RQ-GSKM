# Gain-Shape Reparameterized K-Means: Toward Sub 1-bit KV-Cache

This repository contains the analysis scripts and experimental logs required to reproduce the key figures from the ICML 2026 paper: **"Gain-Shape Reparameterized K-Means: Toward Sub-1-bit KV-Cache"**

By running the provided scripts, you can visualize the centroid shrinkage phenomenon and the effectiveness of GSKM compared to standard K-Means.

<br>

## 📂 Repository Structure

The repository is organized as follows:

```
├── clustering_results/ 
│   ├── clustering_comparison_report_value_dim_8_K1024.json
│   ├── clustering_comparison_report_value_dim_8_K1024_residual.json
│   ├── clustering_comparison_report_value_dim_8_K256.json
│   ├── clustering_comparison_report_value_dim_8_K256_residual.json
│   └── ...
├── plot_clustering.py 
├── plot_clustering_random.py
└── README.md
```

<br>

## 🛠️ Prerequisites

The code is implemented in Python. To run the visualization scripts, install the required dependences:

```Bash
pip install matplotlib numpy
```

<br>

## 📊 Reproducing Figures

### 1. Figure 3: Random Gaussian Sweeps

Visualize the impact of high-dimensional averaging cancellation on standard K-Means vs. GSKM using synthetic data. This corresponds to the dimension ($D$) and capacity ($K$) sweeps discussed in the paper.

To generate the plots for MSE, Gain Error, and Cosine Similarity (Figure 3a-f):

```Bash
python plot_clustering_random.py
```


### 2. Figure 4: Llama-3-8B KV-Cache Reconstruction

Evaluate reconstruction performance on real KV-cache activations (Wikitext-2) for both the original signal and the first residual stage.

To generate the plots for Key/Value Original and Key/Value 1st Residual (Figure 4a-l), run the script specifying the input data directory and the output directory for figures:

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

<br>

## 📝 Citation

If you find this code or our paper useful for your research, please cite.
