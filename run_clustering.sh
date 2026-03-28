#!/bin/bash
# Run KM vs GSKM clustering for all (dim, key/value, K) configurations.
# Outputs JSON reports to clustering_results/.
#
# Usage:
#   bash run_clustering.sh [DATA_DIR]
#
# DATA_DIR: root directory containing dumped_data_dim_*/ (default: current dir)

DATA_DIR=${1:-.}

for K in 1024 256; do
    for dim in 512 128 32 8; do
        for kv in key value; do
            echo ">>> dim=${dim}, kv=${kv}, K=${K}"
            python3 run_clustering.py ${dim} ${kv} ${K} --data_dir "${DATA_DIR}"
        done
    done
done
