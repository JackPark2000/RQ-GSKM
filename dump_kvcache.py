#!/usr/bin/env python3
"""
Extract KV-cache activations from Llama-3-8B on Wikitext-2.

For each transformer layer, the key (or value) cache is a
[num_tokens, num_kv_heads * head_dim] matrix.  This script reshapes it
into contiguous D-dimensional sub-vectors and saves each chunk as a
separate .pt file.

Output layout::

    dumped_data_dim_{D}/
        raw_data_layer_{layer}_{key|value}_cb{chunk}.pt   # {"data": [N, D]}

where ``N = total_tokens`` and ``chunk ∈ [0, kv_dim / D)``.

Usage:
    python dump_kvcache.py --dim 128 --kv key
    python dump_kvcache.py --dim 8 --kv value --max_length 4096

Dependencies (in addition to requirements.txt):
    pip install transformers datasets accelerate
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(
        description="Dump KV-cache activations from Llama-3-8B (Wikitext-2)")
    ap.add_argument("--dim", type=int, required=True,
                    help="Sub-vector dimension D (e.g. 8, 32, 128, 512)")
    ap.add_argument("--kv", type=str, required=True, choices=["key", "value"],
                    help="Extract key or value cache")
    ap.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                    help="HuggingFace model name (default: Meta-Llama-3-8B)")
    ap.add_argument("--max_length", type=int, default=2048,
                    help="Maximum sequence length for tokenisation")
    ap.add_argument("--output_dir", type=str, default=".",
                    help="Root output directory (default: current dir)")
    args = ap.parse_args()

    device = torch.device("cuda")
    kv_idx = 0 if args.kv == "key" else 1

    # ---- Load model & tokenizer ----
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    # ---- Load & tokenize Wikitext-2 ----
    print("Loading Wikitext-2 ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    input_ids = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=args.max_length,
    ).input_ids.to(device)
    print(f"Tokenized: {input_ids.shape[1]} tokens")

    # ---- Forward pass (collect KV cache) ----
    print("Running forward pass ...")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    past_kv = outputs.past_key_values
    n_layers = len(past_kv)

    # Infer geometry
    # past_kv[layer][0|1] shape: [batch=1, num_kv_heads, seq_len, head_dim]
    num_kv_heads = past_kv[0][kv_idx].shape[1]
    head_dim = past_kv[0][kv_idx].shape[3]
    kv_dim = num_kv_heads * head_dim
    seq_len = past_kv[0][kv_idx].shape[2]

    assert kv_dim % args.dim == 0, (
        f"dim={args.dim} does not divide kv_dim={kv_dim} "
        f"(num_kv_heads={num_kv_heads}, head_dim={head_dim})"
    )
    n_chunks = kv_dim // args.dim
    print(f"Geometry: {n_layers} layers, {num_kv_heads} KV heads, "
          f"head_dim={head_dim}, kv_dim={kv_dim} -> {n_chunks} chunks of D={args.dim}")

    # ---- Extract & save ----
    out_dir = os.path.join(args.output_dir, f"dumped_data_dim_{args.dim}")
    os.makedirs(out_dir, exist_ok=True)

    for layer_idx in range(n_layers):
        # [1, num_kv_heads, seq_len, head_dim] -> [seq_len, kv_dim]
        cache = (past_kv[layer_idx][kv_idx]
                 .squeeze(0)              # [num_kv_heads, seq_len, head_dim]
                 .permute(1, 0, 2)        # [seq_len, num_kv_heads, head_dim]
                 .reshape(seq_len, kv_dim) # [seq_len, kv_dim]
                 .cpu().float())

        # Split into D-dimensional chunks
        # cache shape: [seq_len, n_chunks * D]  ->  n_chunks tensors of [seq_len, D]
        chunks = cache.reshape(seq_len, n_chunks, args.dim)

        for cb in range(n_chunks):
            data = chunks[:, cb, :].contiguous()  # [seq_len, D]
            path = os.path.join(
                out_dir, f"raw_data_layer_{layer_idx}_{args.kv}_cb{cb}.pt"
            )
            torch.save({"data": data}, path)

        print(f"  Layer {layer_idx:2d}/{n_layers - 1} saved ({n_chunks} chunks)")

    print(f"\nDone. Output: {out_dir}/")


if __name__ == "__main__":
    main()
