"""Compare single-host, pipeline-parallel, and tensor-parallel generation."""

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


def host_synchronized_step(mx, model, group, timings):
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    stage = model.model
    rank = group.rank()
    stage.pipeline_rank = 0
    stage.pipeline_size = 1

    def step(inputs, cache, sample):
        started = time.perf_counter()
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
            forward_done = time.perf_counter()
            mx.eval(mx.distributed.send(hidden, 0, group=group, stream=mx.cpu))
            timings["local_forward_s"] += forward_done - started
            timings["activation_handoff_s"] += time.perf_counter() - forward_done
            token_id = 0
        else:
            hidden = mx.distributed.recv_like(hidden, 1, group=group, stream=mx.cpu)
            # Wait on the host before submitting the downstream GPU commands.
            mx.eval(hidden)
            activation_ready = time.perf_counter()
            logits = model(inputs, cache=cache, input_embeddings=hidden)[:, -1, :]
            output = mx.argmax(logits, axis=-1) if sample else logits
            mx.eval(output, [entry.state for entry in cache])
            token_id = output.item() if sample else 0
            timings["activation_handoff_s"] += activation_ready - started
            timings["local_forward_s"] += time.perf_counter() - activation_ready

        if sample:
            started = time.perf_counter()
            token_id = mx.distributed.all_sum(mx.array(token_id), group=group, stream=mx.cpu).item()
            timings["token_sync_s"] += time.perf_counter() - started
            return token_id

    return step


def run_synchronous_pipeline(mx, model, tokenizer, prompt, step_size, max_tokens,
                             step=None, timings=None):
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
    prefill_timings = timings.copy() if timings is not None else None
    while len(token_ids) < max_tokens and token_ids[-1] not in tokenizer.eos_token_ids:
        token_ids.append(step(mx.array([[token_ids[-1]]]), cache, True))

    end = time.perf_counter()
    stopped = token_ids[-1] in tokenizer.eos_token_ids
    result = {
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
    if prefill_timings is not None:
        result["pipeline_timings"] = {
            "prefill": prefill_timings,
            "decode": {key: value - prefill_timings[key] for key, value in timings.items()},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("resident", "faithful", "pipeline", "pipeline-ssd", "tensor"))
    parser.add_argument("model", type=Path)
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompt-file", type=Path)
    prompt_source.add_argument("--prompt-suite", type=Path,
                               help="JSON object mapping prompt names to text, run in file order.")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prefill-step-size", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument(
        "--pipeline-split", type=int, nargs=2, metavar=("RANK0", "RANK1"),
        help="Number of layers per rank in pipeline-ssd mode. Rank 0 gets the last layers.",
    )
    parser.add_argument(
        "--pipeline-hot-experts", type=int, metavar="N",
        help="Pack up to N exact experts per layer after the first pipeline-ssd pass.",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"),
                        help="Set GPT-OSS reasoning effort in the chat template.")
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
    if args.pipeline_split is not None and args.mode != "pipeline-ssd":
        parser.error("--pipeline-split requires pipeline-ssd mode")
    if args.pipeline_hot_experts is not None:
        if args.mode != "pipeline-ssd" or args.repeat < 2:
            parser.error("--pipeline-hot-experts requires pipeline-ssd mode and at least two passes")
        if args.pipeline_hot_experts < 1:
            parser.error("--pipeline-hot-experts must be positive")
    if args.pipeline_stage_only and args.prompt_suite is not None:
        parser.error("--pipeline-stage-only takes a single prompt")
    if min(args.tokens, args.prefill_step_size, args.capacity, args.repeat) < 1:
        parser.error("tokens, prefill step size, capacity and repeat must be positive")
    model_path = args.model.expanduser().resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        parser.error(f"missing checkpoint config under {model_path}")
    try:
        prompts = load_prompts(args)
        model_config = json.loads(config_path.read_text())
        model_type = model_config.get("model_type")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.reasoning_effort is not None and model_type != "gpt_oss":
        parser.error("--reasoning-effort requires a GPT-OSS checkpoint")
    if args.pipeline_hot_experts is not None:
        if model_type != "qwen3_5_moe":
            parser.error("--pipeline-hot-experts requires a Qwen 3.5 MoE checkpoint")
        top_k = model_config.get("text_config", model_config)["num_experts_per_tok"]
        # Leave enough cache slots for every routed expert in a full prefill chunk.
        if args.pipeline_hot_experts + top_k * args.prefill_step_size > args.capacity:
            parser.error("--capacity must cover packed experts plus prefill-step-size × experts per token")

    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    pipeline = args.mode in ("pipeline", "pipeline-ssd")
    expert_streaming = args.mode in ("faithful", "pipeline-ssd")
    host_sync = args.pipeline_host_sync or args.mode == "pipeline-ssd"
    if expert_streaming:
        from mlx_moe.lazy_experts.core import get_faithful_stats

    rank = 0
    mx.reset_peak_memory()
    startup = time.perf_counter()
    if pipeline or args.mode == "tensor":
        group = mx.distributed.init(backend="ring", strict=True)
        if group.size() != 2:
            parser.error("distributed mode requires two MLX ring ranks")
        rank = group.rank()
        if expert_streaming:
            from mlx_moe.lazy_experts.pipeline import (
                load_faithful_pipeline_stage,
                pack_pipeline_hot_experts,
            )

            model, tokenizer = load_faithful_pipeline_stage(
                model_path, group, args.capacity, split=args.pipeline_split
            )
        else:
            from mlx_lm.utils import sharded_load

            if pipeline:
                model, tokenizer = sharded_load(str(model_path), pipeline_group=group)
            else:
                model, tokenizer = sharded_load(str(model_path), tensor_group=group)
    elif expert_streaming:
        from mlx_moe.lazy_experts.generate import _startup

        model, tokenizer, _ = _startup(
            str(model_path),
            next(iter(prompts.values())),
            capacity=args.capacity,
            warmup="none",
            expert_policy="faithful",
        )
    else:
        from mlx_moe.lazy_experts.generate import _load_model_and_tokenizer

        model, tokenizer, _ = _load_model_and_tokenizer(str(model_path), lazy=True)
        mx.eval(model.parameters())

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
    chat_options = {}
    if model_type == "gpt_oss":
        context["reasoning_effort"] = args.reasoning_effort or "medium"
        if args.reasoning_effort is not None:
            chat_options["reasoning_effort"] = args.reasoning_effort
    if args.pipeline_synchronous:
        context["pipeline_synchronous"] = True
    if host_sync:
        context["pipeline_host_sync"] = True
    if args.mode == "pipeline-ssd":
        context["layers_start"] = model.model.start_idx
        context["layers_end_exclusive"] = model.model.end_idx
    if args.pipeline_hot_experts is not None:
        context["pipeline_hot_experts"] = args.pipeline_hot_experts
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
            **chat_options,
        ))
        for name, prompt in prompts.items()
    ]
    if args.pipeline_stage_only or host_sync:
        if model.model_type not in {"qwen3_5", "qwen3_5_moe"}:
            parser.error("this diagnostic requires a Qwen 3.5 model")
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
        if args.pipeline_hot_experts is not None and trial == len(formatted_prompts) + 1:
            pack_start = time.perf_counter()
            packed = pack_pipeline_hot_experts(model, args.pipeline_hot_experts)
            pack_seconds = time.perf_counter() - pack_start
            mx.eval(mx.distributed.all_sum(mx.array(1), group=group, stream=mx.cpu))
            print(json.dumps({
                **context, "event": "packed", "experts": packed, "seconds": pack_seconds,
                "peak_bytes": mx.get_peak_memory(),
            }), flush=True)
            mx.reset_peak_memory()
        trial_context = {**context, "event": "trial", "trial": trial, "prompt_name": name}
        if args.prompt_suite is not None:
            print(f"  Trial {trial}: {name}", flush=True)
        before = get_faithful_stats(model) if expert_streaming else None
        if args.pipeline_synchronous or host_sync:
            timings = step = None
            if host_sync:
                timings = dict.fromkeys(("local_forward_s", "activation_handoff_s", "token_sync_s"), 0.0)
                step = host_synchronized_step(mx, model, group, timings)
            result = run_synchronous_pipeline(
                mx, model, tokenizer, formatted, args.prefill_step_size, args.tokens, step, timings
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
            if args.pipeline_hot_experts is not None:
                experts["packed_dispatches"] = after["packed_dispatches"] - before["packed_dispatches"]
            result["experts"] = experts
        print(json.dumps({**trial_context, **result}), flush=True)


def run_streaming_trial(mx, model, tokenizer, prompt, args, sampler, stream_generate):
    token_ids = []
    text = []
    final_header = []
    if getattr(model, "model_type", None) == "gpt_oss":
        final_header = tokenizer.encode("<|channel|>final<|message|>", add_special_tokens=False)
        control_tokens = set(tokenizer.eos_token_ids)
        for marker in ("<|start|>", "<|end|>", "<|return|>", "<|call|>", "<|channel|>", "<|message|>"):
            control_tokens.update(tokenizer.encode(marker, add_special_tokens=False))
    in_final = False
    first_final = None
    final_tokens = 0

    start = time.perf_counter()
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.tokens,
        sampler=sampler,
        prefill_step_size=args.prefill_step_size,
    ):
        token = int(response.token)
        if not token_ids:
            first = time.perf_counter()
        token_ids.append(token)
        text.append(response.text)
        if final_header:
            # Follow token IDs because the detokenizer can buffer text across yields.
            if token in control_tokens:
                in_final = token_ids[-len(final_header):] == final_header
            elif in_final:
                if first_final is None:
                    first_final = time.perf_counter()
                final_tokens += 1
    elapsed = time.perf_counter() - start
    result = {
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
    if final_header:
        result["ttft_final_s"] = first_final - start if first_final is not None else None
        result["final_answer_tokens"] = final_tokens
    return result


if __name__ == "__main__":
    main()
