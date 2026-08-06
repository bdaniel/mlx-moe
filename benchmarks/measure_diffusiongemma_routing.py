"""Measure DiffusionGemma MoE routing sparsity with mlx-vlm.

Run with an mlx-vlm-compatible MLX stack, e.g.:

    uv run --with mlx-vlm --with mlx==0.31.2 --with mlx-lm==0.31.3 \
      python benchmarks/measure_diffusiongemma_routing.py \
      /path/to/diffusiongemma-26B-A4B-it-4bit --steps 4 --tokens 16
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time

import mlx.core as mx
import numpy as np


def _attach_router_trace(model):
    from mlx_vlm.models.diffusion_gemma import language as dg_language

    stats = defaultdict(lambda: {
        "calls": 0,
        "positions": 0,
        "unique": set(),
        "hist": defaultdict(int),
    })
    original_call = dg_language.Router.__call__

    for idx, layer in enumerate(model.layers):
        layer.router._trace_layer_idx = idx

    def traced_call(self, x):
        indices, weights = original_call(self, x)
        layer_idx = getattr(self, "_trace_layer_idx", None)
        if layer_idx is not None:
            mx.eval(indices)
            arr = np.asarray(indices).reshape(-1)
            s = stats[layer_idx]
            s["calls"] += 1
            s["positions"] += int(indices.shape[0])
            for eid in np.unique(arr):
                s["unique"].add(int(eid))
            vals, counts = np.unique(arr, return_counts=True)
            for eid, count in zip(vals, counts):
                s["hist"][int(eid)] += int(count)
        return indices, weights

    dg_language.Router.__call__ = traced_call
    return stats, lambda: setattr(dg_language.Router, "__call__", original_call)


def _patch_router_only_no_experts():
    from mlx_vlm.models.diffusion_gemma import diffusion_gemma, language

    original_sanitize = diffusion_gemma.Model.sanitize
    original_experts_call = language.Experts.__call__

    def sanitize_without_experts(self, weights):
        weights = {
            key: value
            for key, value in weights.items()
            if ".experts." not in key
        }
        return original_sanitize(self, weights)

    def zero_experts(self, x, top_k_indices, top_k_weights):
        return mx.zeros_like(x)

    diffusion_gemma.Model.sanitize = sanitize_without_experts
    language.Experts.__call__ = zero_experts

    def restore():
        diffusion_gemma.Model.sanitize = original_sanitize
        language.Experts.__call__ = original_experts_call

    return restore


def _summarize(stats, num_experts: int):
    layers = {}
    unique_counts = []
    top1_shares = []
    for layer_idx in sorted(stats):
        s = stats[layer_idx]
        hist = dict(s["hist"])
        top = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[:16]
        total_routes = sum(hist.values())
        top1 = (top[0][1] / total_routes) if top and total_routes else 0.0
        unique_count = len(s["unique"])
        unique_counts.append(unique_count)
        top1_shares.append(top1)
        layers[str(layer_idx)] = {
            "calls": s["calls"],
            "positions": s["positions"],
            "unique_experts": unique_count,
            "unique_fraction": unique_count / num_experts if num_experts else None,
            "top1_share": top1,
            "top_experts": [{"id": eid, "routes": count} for eid, count in top],
        }

    return {
        "layers": layers,
        "summary": {
            "layers": len(layers),
            "num_experts": num_experts,
            "unique_min": min(unique_counts) if unique_counts else 0,
            "unique_median": float(np.median(unique_counts)) if unique_counts else 0.0,
            "unique_max": max(unique_counts) if unique_counts else 0,
            "top1_share_median": float(np.median(top1_shares)) if top1_shares else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Measure DiffusionGemma expert routing")
    parser.add_argument("model", help="Local path or HF repo")
    parser.add_argument("--prompt", default="Describe this image in one short sentence.")
    parser.add_argument("--image", default=None)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--min-canvas", type=int, default=16)
    parser.add_argument("--max-canvas", type=int, default=16)
    parser.add_argument(
        "--router-only",
        action="store_true",
        help="Skip expert weights and zero expert outputs to measure router sparsity under low memory.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    t0 = time.perf_counter()
    restore_router_only = _patch_router_only_no_experts() if args.router_only else None
    model, processor = load(args.model, lazy=True, strict=not args.router_only)
    config = load_config(args.model)
    formatted = apply_chat_template(
        processor,
        config,
        args.prompt,
        num_images=1 if args.image else 0,
    )

    num_experts = int(model.config.text_config.num_experts)
    stats, restore = _attach_router_trace(model)
    try:
        result = generate(
            model,
            processor,
            formatted,
            image=args.image,
            max_tokens=args.tokens,
            max_denoising_steps=args.steps,
            diffusion_min_canvas_length=args.min_canvas,
            diffusion_max_canvas_length=args.max_canvas,
            verbose=False,
        )
    finally:
        restore()
        if restore_router_only is not None:
            restore_router_only()

    payload = _summarize(stats, num_experts)
    payload["model"] = args.model
    payload["prompt"] = args.prompt
    payload["image"] = args.image
    payload["tokens"] = args.tokens
    payload["steps"] = args.steps
    payload["router_only"] = args.router_only
    payload["elapsed_s"] = time.perf_counter() - t0
    payload["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    payload["output"] = result.text

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
