"""Compare single-host SSD streaming, resident pipelining, and SSD pipelining."""

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


def encode_prompt(tokenizer, prompt):
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)


def load_prompts(args):
    if args.prompt_suite is not None:
        prompts = json.loads(args.prompt_suite.read_text())
        if not isinstance(prompts, dict) or len(prompts) < 2 or any(
            not name.strip() or not isinstance(prompt, str) or not prompt.strip()
            for name, prompt in prompts.items()
        ):
            raise ValueError("prompt suite must map at least two unique names to nonempty prompts")
        return prompts
    if args.prompt_file is not None:
        return {args.prompt_file.stem: args.prompt_file.read_text()}
    return {"default": PROMPT}


def run_local_pipeline_stage(mx, model, prompt_ids):
    stage = model.model
    # The loaded slice and its cache indices are independent of these flags.
    stage.pipeline_rank = 0
    stage.pipeline_size = 1
    cache = model.make_cache()
    start = time.perf_counter()
    logits = model(mx.array([prompt_ids]), cache=cache)[:, -1, :]
    mx.eval(logits, [entry.state for entry in cache])
    return {
        "layers_start": stage.start_idx,
        "layers_end_exclusive": stage.end_idx,
        "prompt_tokens": len(prompt_ids),
        "forward_s": time.perf_counter() - start,
        "peak_bytes": mx.get_peak_memory(),
    }


def host_synchronized_step(mx, model, group):
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    stage = model.model
    rank = group.rank()
    stage.pipeline_rank = 0
    stage.pipeline_size = 1

    def step(inputs, cache, sample):
        hidden = stage.embed_tokens(inputs)
        if rank == 1:
            fa_mask = (
                create_attention_mask(hidden, cache[stage.fa_idx])
                if stage.fa_idx is not None else None
            )
            ssm_mask = (
                create_ssm_mask(hidden, cache[stage.ssm_idx])
                if stage.ssm_idx is not None else None
            )
            for layer, entry in zip(stage.pipeline_layers, cache):
                hidden = layer(hidden, mask=ssm_mask if layer.is_linear else fa_mask, cache=entry)
            # Materialize the GPU result before scheduling the TCP send.
            mx.eval(hidden, [entry.state for entry in cache])
            mx.eval(mx.distributed.send(hidden, 0, group=group, stream=mx.cpu))
            token_id = 0
        else:
            hidden = mx.distributed.recv_like(hidden, 1, group=group, stream=mx.cpu)
            # Wait on the host before submitting the downstream GPU commands.
            mx.eval(hidden)
            logits = model(inputs, cache=cache, input_embeddings=hidden)[:, -1, :]
            output = mx.argmax(logits, axis=-1) if sample else logits
            mx.eval(output, [entry.state for entry in cache])
            token_id = output.item() if sample else 0

        if sample:
            return mx.distributed.all_sum(mx.array(token_id), group=group, stream=mx.cpu).item()

    return step


def run_synchronous_pipeline(mx, model, tokenizer, prompt, step_size, max_tokens, step=None):
    from mlx_lm.models.cache import make_prompt_cache

    if step is None:
        def step(inputs, cache, sample):
            logits = model(inputs, cache=cache)[:, -1, :]
            output = mx.argmax(logits, axis=-1) if sample else logits
            # Evaluating the output keeps the ring collectives in the prefill graph.
            mx.eval(output, [entry.state for entry in cache])
            return output.item() if sample else None

    prompt_ids = encode_prompt(tokenizer, prompt)
    cache = make_prompt_cache(model)
    start = time.perf_counter()
    for offset in range(0, len(prompt_ids), step_size):
        chunk = prompt_ids[offset:offset + step_size]
        print(f"  Synchronous prefill: {offset + len(chunk)}/{len(prompt_ids)} tokens", flush=True)
        token_id = step(mx.array([chunk]), cache, offset + len(chunk) == len(prompt_ids))

    token_ids = [token_id]
    first = time.perf_counter()
    while len(token_ids) < max_tokens and token_ids[-1] not in tokenizer.eos_token_ids:
        token_ids.append(step(mx.array([[token_ids[-1]]]), cache, True))

    end = time.perf_counter()
    stopped = token_ids[-1] in tokenizer.eos_token_ids
    return {
        "ttft_s": first - start,
        "elapsed_s": end - start,
        "prompt_tokens": len(prompt_ids),
        "generation_tokens": len(token_ids),
        "prefill_tps": len(prompt_ids) / (first - start),
        "decode_tps": (len(token_ids) - 1) / (end - first) if len(token_ids) > 1 else 0.0,
        "peak_bytes": mx.get_peak_memory(),
        "finish_reason": "stop" if stopped else "length",
        "token_ids": token_ids,
        "text": tokenizer.decode(token_ids[:-1] if stopped else token_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("faithful", "pipeline", "pipeline-ssd"))
    parser.add_argument("model", type=Path)
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompt-file", type=Path)
    prompt_source.add_argument("--prompt-suite", type=Path,
                               help="JSON object mapping prompt names to text, run in file order.")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prefill-step-size", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--pipeline-synchronous", action="store_true",
                             help="Evaluate model outputs and cache state together for each pipeline step.")
    diagnostics.add_argument("--pipeline-stage-only", action="store_true",
                             help="Time one local forward pass on each Qwen 3.5 pipeline shard.")
    diagnostics.add_argument("--pipeline-host-sync", action="store_true",
                             help="Finish GPU work before each TCP handoff in a Qwen 3.5 pipeline.")
    args = parser.parse_args()
    if (args.pipeline_synchronous or args.pipeline_stage_only or args.pipeline_host_sync) and args.mode != "pipeline":
        parser.error("pipeline diagnostics require pipeline mode")
    if args.pipeline_stage_only and args.prompt_suite is not None:
        parser.error("--pipeline-stage-only takes a single prompt")
    if min(args.tokens, args.prefill_step_size, args.capacity, args.repeat) < 1:
        parser.error("tokens, prefill step size, capacity and repeat must be positive")
    model_path = args.model.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        parser.error(f"missing checkpoint config under {model_path}")
    try:
        prompts = load_prompts(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    pipeline = args.mode != "faithful"
    expert_streaming = args.mode != "pipeline"
    host_sync = args.pipeline_host_sync or args.mode == "pipeline-ssd"
    if expert_streaming:
        from mlx_moe.lazy_experts.core import get_faithful_stats

    rank = 0
    mx.reset_peak_memory()
    startup = time.perf_counter()
    if pipeline:
        group = mx.distributed.init(backend="ring", strict=True)
        if group.size() != 2:
            parser.error("pipeline mode requires two MLX ring ranks")
        rank = group.rank()
        if expert_streaming:
            from mlx_moe.lazy_experts.pipeline import load_faithful_pipeline_stage

            model, tokenizer = load_faithful_pipeline_stage(model_path, group, args.capacity)
        else:
            from mlx_lm.utils import sharded_load

            model, tokenizer = sharded_load(str(model_path), pipeline_group=group)
    else:
        from mlx_moe.lazy_experts.generate import _startup

        model, tokenizer, _ = _startup(
            str(model_path),
            next(iter(prompts.values())),
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
        "capacity": args.capacity if expert_streaming else None,
    }
    if args.pipeline_synchronous:
        context["pipeline_synchronous"] = True
    if host_sync:
        context["pipeline_host_sync"] = True
    if args.mode == "pipeline-ssd":
        context["layers_start"] = model.model.start_idx
        context["layers_end_exclusive"] = model.model.end_idx
    loaded = {
        **context,
        "event": "loaded",
        "seconds": time.perf_counter() - startup,
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }
    print(json.dumps(loaded), flush=True)

    formatted_prompts = [
        (name, tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
        for name, prompt in prompts.items()
    ]
    step = None
    if args.pipeline_stage_only or host_sync:
        if model.model_type not in {"qwen3_5", "qwen3_5_moe"}:
            parser.error("this diagnostic requires a Qwen 3.5 model")
    if host_sync:
        step = host_synchronized_step(mx, model, group)
    if args.pipeline_stage_only:
        _, formatted = formatted_prompts[0]
        prompt_ids = encode_prompt(tokenizer, formatted)[:args.prefill_step_size]
        mx.reset_peak_memory()
        result = run_local_pipeline_stage(mx, model, prompt_ids)
        print(json.dumps({**context, "event": "stage_probe", **result}), flush=True)
        return

    sampler = make_sampler(temp=0.0)
    for trial, (name, formatted) in enumerate(formatted_prompts * args.repeat, start=1):
        mx.reset_peak_memory()
        trial_context = {**context, "event": "trial", "trial": trial, "prompt_name": name}
        if args.prompt_suite is not None:
            print(f"  Trial {trial}: {name}", flush=True)
        before = get_faithful_stats(model) if expert_streaming else None
        if args.pipeline_synchronous or host_sync:
            result = run_synchronous_pipeline(
                mx, model, tokenizer, formatted, args.prefill_step_size, args.tokens, step
            )
        else:
            result = run_streaming_trial(
                mx, model, tokenizer, formatted, args, sampler, stream_generate
            )
        if before is not None:
            after = get_faithful_stats(model)
            experts = {
                key: after[key] - before[key]
                for key in ("hits", "misses", "load_calls", "bytes_loaded", "load_seconds")
            }
            experts["bytes_loaded_per_generated_token"] = (
                experts["bytes_loaded"] / result["generation_tokens"]
            )
            experts["load_time_fraction"] = experts["load_seconds"] / result["elapsed_s"]
            result["experts"] = experts
        print(json.dumps({**trial_context, **result}), flush=True)


def run_streaming_trial(mx, model, tokenizer, prompt, args, sampler, stream_generate):
    token_ids = []
    text = []
    start = time.perf_counter()
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.tokens,
        sampler=sampler,
        prefill_step_size=args.prefill_step_size,
    ):
        if not token_ids:
            first = time.perf_counter()
        token_ids.append(int(response.token))
        text.append(response.text)
    elapsed = time.perf_counter() - start
    return {
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


if __name__ == "__main__":
    main()
