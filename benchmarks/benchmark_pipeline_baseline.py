"""Compare single-host faithful expert streaming with two-host MLX-LM."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import socket
import time


PROMPT = (
    "Write a Python function that merges overlapping half-open intervals. "
    "Explain the algorithm and its time complexity."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("faithful", "pipeline"))
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prefill-step-size", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if min(args.tokens, args.prefill_step_size, args.capacity, args.repeat) < 1:
        parser.error("tokens, prefill step size, capacity and repeat must be positive")
    model_path = args.model.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        parser.error(f"missing checkpoint config under {model_path}")
    prompt = args.prompt_file.read_text() if args.prompt_file else PROMPT

    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    rank = 0
    mx.reset_peak_memory()
    startup = time.perf_counter()
    if args.mode == "pipeline":
        from mlx_lm.utils import sharded_load

        group = mx.distributed.init(backend="ring", strict=True)
        if group.size() != 2:
            parser.error("pipeline mode requires two MLX ring ranks")
        rank = group.rank()
        model, tokenizer = sharded_load(str(model_path), pipeline_group=group)
    else:
        from mlx_moe.lazy_experts.core import get_faithful_stats
        from mlx_moe.lazy_experts.generate import _startup

        model, tokenizer, _ = _startup(
            str(model_path),
            prompt,
            capacity=args.capacity,
            warmup="none",
            expert_policy="faithful",
        )

    context = {
        "mode": args.mode,
        "host": socket.gethostname(),
        "rank": rank,
        "model": str(model_path),
        "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm")},
        "prefill_step_size": args.prefill_step_size,
        "max_tokens": args.tokens,
        "capacity": args.capacity if args.mode == "faithful" else None,
    }
    loaded = {
        **context,
        "event": "loaded",
        "seconds": time.perf_counter() - startup,
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }
    print(json.dumps(loaded), flush=True)

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    sampler = make_sampler(temp=0.0)
    for trial in range(1, args.repeat + 1):
        mx.reset_peak_memory()
        before = get_faithful_stats(model) if args.mode == "faithful" else None
        token_ids = []
        text = []
        start = time.perf_counter()
        for response in stream_generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=args.tokens,
            sampler=sampler,
            prefill_step_size=args.prefill_step_size,
        ):
            if not token_ids:
                first = time.perf_counter()
            token_ids.append(int(response.token))
            text.append(response.text)
        elapsed = time.perf_counter() - start
        result = {
            **context,
            "event": "trial",
            "trial": trial,
            "ttft_s": first - start,
            "elapsed_s": elapsed,
            "prompt_tokens": response.prompt_tokens,
            "generation_tokens": response.generation_tokens,
            "prefill_tps": response.prompt_tps,
            "decode_tps": response.generation_tps,
            "peak_bytes": mx.get_peak_memory(),
            "finish_reason": response.finish_reason,
            "token_ids": token_ids,
            "text": "".join(text),
        }
        if before is not None:
            after = get_faithful_stats(model)
            experts = {
                key: after[key] - before[key]
                for key in ("hits", "misses", "load_calls", "bytes_loaded", "load_seconds")
            }
            experts["bytes_loaded_per_generated_token"] = (
                experts["bytes_loaded"] / response.generation_tokens
            )
            experts["load_time_fraction"] = experts["load_seconds"] / elapsed
            result["experts"] = experts
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
