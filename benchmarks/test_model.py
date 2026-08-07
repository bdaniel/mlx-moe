"""Test a single MoE model with mlx-moe.

Usage:
    uv run python benchmarks/test_model.py MODEL_NAME [--capacity N] [--baseline] [--tokens N]

Examples:
    uv run python benchmarks/test_model.py mlx-community/Qwen3-30B-A3B-4bit --baseline
    uv run python benchmarks/test_model.py mlx-community/Qwen3-30B-A3B-4bit --capacity 37
    uv run python benchmarks/test_model.py mlx-community/Qwen2-57B-A14B-Instruct-4bit
"""

import argparse
import gc
import json
import time

import mlx.core as mx

TEST_PROMPTS = [
    "Write a Python function that implements binary search on a sorted array.",
    "Explain the difference between TCP and UDP in simple terms.",
    "What causes the seasons on Earth?",
]


def run_baseline(model_name, max_tokens=200):
    """Run stock mlx-lm generation (no mlx-moe) as baseline."""
    import mlx_lm

    print(f"\n{'='*60}")
    print(f"BASELINE: {model_name}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_name)
    mx.eval(model.parameters())
    load_time = time.perf_counter() - t0
    mem_gb = mx.get_active_memory() / 1e9
    print(f"Load: {load_time:.1f}s, Memory: {mem_gb:.1f} GB")

    prompt = TEST_PROMPTS[0]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            pass

    print(f"\nGenerating {max_tokens} tokens...")
    t0 = time.perf_counter()
    text = mlx_lm.generate(model, tokenizer, prompt=prompt,
                            max_tokens=max_tokens, verbose=True)
    gen_time = time.perf_counter() - t0
    mem_after = mx.get_active_memory() / 1e9

    print(f"\nBaseline results:")
    print(f"  Memory: {mem_gb:.1f} -> {mem_after:.1f} GB")
    print(f"  Time: {gen_time:.1f}s")
    print(f"  Output preview: {text[:200]}...")

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    return {"memory_gb": mem_gb, "memory_after_gb": mem_after,
            "load_time": load_time, "gen_time": gen_time}


def run_mlx_moe(
    model_name,
    capacity=None,
    max_tokens=200,
    expert_policy="approximate",
):
    """Run mlx-moe generation with diagnostics."""
    from mlx_moe.lazy_experts.generate import _startup, _stream_with_dynamic_updates
    from mlx_moe.lazy_experts.core import (
        adaptive_cache_growth,
        dynamic_cache_update,
        get_faithful_stats,
        get_fallback_stats,
    )
    from mlx_moe.lazy_experts.loading import _find_switch_mlp, _detect_num_experts
    import mlx_lm

    print(f"\n{'='*60}")
    print(f"MLX-MOE: {model_name} (capacity={capacity or 'auto'})")
    print(f"{'='*60}")

    prompt_text = TEST_PROMPTS[0]

    t0 = time.perf_counter()
    model, tokenizer, model_path = _startup(
        model_name,
        prompt_text,
        capacity=capacity,
        warmup="none" if expert_policy == "faithful" else "hybrid",
        expert_policy=expert_policy,
    )
    startup_time = time.perf_counter() - t0
    mem_gb = mx.get_active_memory() / 1e9

    # Report architecture
    moe_layers = 0
    num_experts = 0
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is not None:
            moe_layers += 1
            if num_experts == 0:
                num_experts = _detect_num_experts(switch)
    print(f"\nArchitecture: {moe_layers} MoE layers, {num_experts} experts/layer")
    print(f"Startup: {startup_time:.1f}s, Memory: {mem_gb:.1f} GB")

    # Generate on each test prompt
    results = []
    for i, prompt_text in enumerate(TEST_PROMPTS):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                kwargs = {"add_generation_prompt": True, "tokenize": False}
                if getattr(tokenizer, "has_thinking", False):
                    kwargs["enable_thinking"] = False
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    **kwargs)
            except Exception:
                prompt = prompt_text
        else:
            prompt = prompt_text

        print(f"\n--- Prompt {i+1}: {prompt_text[:50]}... ---")

        fb_before = get_fallback_stats(model)
        faithful_before = get_faithful_stats(model)
        t0 = time.perf_counter()
        tokens = 0
        text = ""
        dcu_calls = 0
        dcu_swaps = 0
        dcu_requests = 0
        dcu_fallbacks = 0
        grown_experts = 0
        growth_events = 0
        last_headroom_gb = None

        def track_update(stats, growth=None):
            nonlocal dcu_calls, dcu_swaps, dcu_requests, dcu_fallbacks
            nonlocal grown_experts, growth_events, last_headroom_gb
            if stats:
                dcu_calls += 1
                dcu_swaps += sum(s.get("swaps", 0) for s in stats)
                dcu_requests += sum(s.get("requests", 0) for s in stats)
                dcu_fallbacks += sum(s.get("fallbacks", 0) for s in stats)
            if growth:
                headroom_gb = growth.get("headroom_gb")
                if headroom_gb is not None:
                    last_headroom_gb = headroom_gb
                grown = growth.get("grown", 0)
                if grown:
                    grown_experts += grown
                    growth_events += 1

        responses = mlx_lm.stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens
        )
        for resp in _stream_with_dynamic_updates(
            model, responses, on_update=track_update, adaptive_growth=True
        ):
            text += resp.text
            tokens = resp.generation_tokens

        for _ in range(4 if expert_policy == "approximate" else 0):
            stats = dynamic_cache_update(model, max_layer_updates=32)
            requests = sum(s.get("requests", 0) for s in stats)
            fallbacks = sum(s.get("fallbacks", 0) for s in stats)
            growth = (
                adaptive_cache_growth(model, fallbacks / requests)
                if requests
                else None
            )
            track_update(stats, growth)
            if not any(s.get("swaps", 0) or s.get("requests", 0) for s in stats):
                break

        gen_time = time.perf_counter() - t0
        tps = tokens / gen_time if gen_time > 0 else 0.0
        mem_after = mx.get_active_memory() / 1e9

        fb_after = get_fallback_stats(model)
        faithful_after = get_faithful_stats(model)
        fb_requests = fb_after["total_requests"] - fb_before["total_requests"]
        fb_fallbacks = fb_after["total_fallbacks"] - fb_before["total_fallbacks"]
        fallback_rate = fb_fallbacks / fb_requests if fb_requests else 0.0
        faithful_requests = faithful_after["requests"] - faithful_before["requests"]
        faithful_hits = faithful_after["hits"] - faithful_before["hits"]
        faithful_stats = {
            "requests": faithful_requests,
            "hits": faithful_hits,
            "misses": faithful_after["misses"] - faithful_before["misses"],
            "hit_rate": faithful_hits / faithful_requests if faithful_requests else 0.0,
            "load_calls": faithful_after["load_calls"] - faithful_before["load_calls"],
            "experts_loaded": (
                faithful_after["experts_loaded"] - faithful_before["experts_loaded"]
            ),
            "bytes_loaded": (
                faithful_after["bytes_loaded"] - faithful_before["bytes_loaded"]
            ),
            "load_seconds": (
                faithful_after["load_seconds"] - faithful_before["load_seconds"]
            ),
        }

        print(f"  {tokens} tokens, {tps:.1f} tok/s, {gen_time:.1f}s")
        print(f"  Memory: {mem_after:.1f} GB")
        print(f"  Fallback: {fallback_rate*100:.1f}% ({fb_fallbacks}/{fb_requests})")
        if expert_policy == "faithful":
            print(
                f"  Exact loads: {faithful_stats['bytes_loaded'] / 1e6:.1f} MB, "
                f"{faithful_stats['load_seconds']:.1f}s stall, "
                f"{faithful_stats['load_calls']} calls, "
                f"{faithful_stats['hit_rate'] * 100:.1f}% hits"
            )
        headroom = (
            f", headroom={last_headroom_gb:.1f} GB"
            if last_headroom_gb is not None
            else ""
        )
        print(
            f"  Updates: calls={dcu_calls}, swaps={dcu_swaps}, "
            f"grown={grown_experts} ({growth_events} events){headroom}"
        )
        print(f"  Output: {text[:150]}...")

        results.append({
            "prompt": prompt_text[:50],
            "tokens": tokens,
            "tps": tps,
            "gen_time": gen_time,
            "memory_gb": mem_after,
            "fallback_rate": fallback_rate,
            "faithful": faithful_stats,
            "dcu_calls": dcu_calls,
            "dcu_swaps": dcu_swaps,
            "dcu_fallback_rate": dcu_fallbacks / dcu_requests if dcu_requests else 0.0,
            "grown_experts": grown_experts,
            "growth_events": growth_events,
        })

    # Cleanup
    st_map = getattr(model, "_st_map", None)
    if st_map is not None:
        st_map.close()
    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    return {
        "startup_time": startup_time,
        "memory_gb": mem_gb,
        "moe_layers": moe_layers,
        "num_experts": num_experts,
        "prompts": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test a MoE model with mlx-moe")
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument("--capacity", "-c", type=int, default=None,
                        help="Expert capacity per layer (default: auto)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run stock mlx-lm baseline")
    parser.add_argument("--tokens", "-t", type=int, default=200,
                        help="Max tokens to generate (default: 200)")
    parser.add_argument(
        "--expert-policy",
        choices=("approximate", "faithful"),
        default="approximate",
    )
    args = parser.parse_args()

    all_results = {
        "model": args.model,
        "capacity": args.capacity,
        "expert_policy": args.expert_policy,
    }

    if args.baseline:
        baseline = run_baseline(args.model, max_tokens=args.tokens)
        all_results["baseline"] = baseline
        # Need a fresh process for mlx-moe after baseline since Metal buffers leak
        print("\n[NOTE: For accurate mlx-moe results after baseline, run in separate process]")

    flash = run_mlx_moe(
        args.model,
        capacity=args.capacity,
        max_tokens=args.tokens,
        expert_policy=args.expert_policy,
    )
    all_results["mlx_moe"] = flash

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
