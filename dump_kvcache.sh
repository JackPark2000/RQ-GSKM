#!/bin/bash
# Dump KV-cache activations from Llama-3-8B on Wikitext-2
# for all required (dim, key/value) configurations.

for dim in 8 32 128 512; do
    for kv in key value; do
        echo ">>> Dumping dim=${dim}, kv=${kv}"
        python3 dump_kvcache.py --dim ${dim} --kv ${kv}
    done
done
