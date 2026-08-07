#!/usr/bin/env python3
"""Execution-graded local model benchmark for coding and reverse engineering."""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx_lm

from mlx_moe.lazy_experts.core import get_faithful_stats, get_fallback_stats
from mlx_moe.lazy_experts.generate import (
    _load_model_and_tokenizer,
    _startup,
    _stream_with_dynamic_updates,
)
from mlx_moe.lazy_experts.warmup import fast_delta_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "logs" / "agent-quality"


def _chat_prompt(tokenizer, text: str) -> str:
    if not tokenizer.has_chat_template:
        return text
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if tokenizer.has_thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], **kwargs)


def _json_value(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON value found")


def _python_source(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def grade_binary_search(text: str) -> tuple[float, str]:
    source = _python_source(text)
    tree = ast.parse(source)
    forbidden = (ast.Import, ast.ImportFrom, ast.Attribute, ast.Global, ast.Nonlocal)
    if any(isinstance(node, forbidden) for node in ast.walk(tree)):
        return 0.0, "unsafe syntax"
    namespace: dict = {}
    exec(compile(tree, "<model-output>", "exec"), {"__builtins__": {"len": len}}, namespace)
    fn = namespace.get("binary_search")
    if not callable(fn):
        return 0.0, "binary_search missing"
    cases = [
        ([], 1, -1),
        ([1], 1, 0),
        ([1], 2, -1),
        ([1, 3, 5, 7], 1, 0),
        ([1, 3, 5, 7], 7, 3),
        ([1, 3, 5, 7], 4, -1),
    ]
    passed = sum(fn(xs, target) == expected for xs, target, expected in cases)
    return passed / len(cases), f"{passed}/{len(cases)} cases"


def grade_tool_call(text: str) -> tuple[float, str]:
    value = _json_value(text)
    args = value.get("arguments", {}) if isinstance(value, dict) else {}
    checks = [
        value.get("name") == "read_file",
        args.get("path") == "mlx_moe/server.py",
        isinstance(args.get("line_start"), int) and args["line_start"] <= 200,
        isinstance(args.get("line_end"), int) and args["line_end"] >= 220,
    ]
    return sum(checks) / len(checks), f"{sum(checks)}/{len(checks)} fields"


def grade_grounding(text: str) -> tuple[float, str]:
    value = _json_value(text)
    expected = {"a": True, "b": False, "c": True, "d": False, "e": True}
    passed = sum(value.get(key) is answer for key, answer in expected.items())
    return passed / len(expected), f"{passed}/{len(expected)} claims"


def grade_reverse_engineering(text: str) -> tuple[float, str]:
    value = _json_value(text)
    if not isinstance(value, list):
        return 0.0, "expected JSON array"
    joined = json.dumps(value).lower()
    checks = [
        len(value) == 5,
        "endian" in joined,
        "length" in joined,
        "crc" in joined or "checksum" in joined,
        "payload" in joined and ("byte" in joined or "bit" in joined),
        "header" in joined or "trailer" in joined or "footer" in joined,
    ]
    return sum(checks) / len(checks), f"{sum(checks)}/{len(checks)} rubric items"


TASKS = [
    (
        "executable_patch",
        "Return only Python code defining binary_search(xs, target). It must return the "
        "index of target in a sorted list or -1 when absent and terminate for every input.",
        grade_binary_search,
    ),
    (
        "tool_call",
        "Available tool: read_file(path, line_start, line_end). Inspect the Server startup "
        "and load implementation around lines 180-240 of mlx_moe/server.py. Return exactly "
        "one JSON object with keys name and arguments. No prose.",
        grade_tool_call,
    ),
    (
        "grounded_comprehension",
        "Read this exact function:\n"
        "```python\ndef choose(rate):\n    if rate >= 0.35:\n        return 1, 48\n"
        "    if rate >= 0.20:\n        return 1, 32\n    return 4, 12\n```\n"
        "Return only JSON booleans for: a=choose(.4)==[1,48], "
        "b=choose(.3)==[4,12], c=choose(.2)==[1,32], "
        "d=choose(.1)==[1,48], e=rate .35 takes first branch. "
        "Format: {\"a\":true,...}.",
        grade_grounding,
    ),
    (
        "reverse_engineering",
        "A binary frame has a changing four-byte header, payload, and changing four-byte "
        "trailer. Header correlates with payload length; one payload bit changes trailer. "
        "Return exactly five next experiments as a JSON array. Each item must contain "
        "experiment and expected_evidence. No prose outside JSON.",
        grade_reverse_engineering,
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument(
        "--backend",
        choices=("mlx-lm", "mlx-moe"),
        default="mlx-lm",
    )
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--pin-top-k", type=int)
    parser.add_argument("--warmup", choices=("hybrid", "full", "none"), default="none")
    parser.add_argument(
        "--expert-policy",
        choices=("approximate", "faithful"),
        default="faithful",
    )
    parser.add_argument("--adaptive-growth", action="store_true")
    parser.add_argument("--prefill-step-size", type=int)
    parser.add_argument(
        "--delta-warmup",
        choices=("predictive", "router-only"),
        help="Specialize approximate cache before each task",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=tuple(task[0] for task in TASKS),
        help="Run only selected task; repeat for multiple tasks",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()

    run_dir = args.results_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    selected_tasks = [task for task in TASKS if not args.task or task[0] in args.task]

    mx.reset_peak_memory()
    started = time.perf_counter()
    if args.backend == "mlx-moe":
        model, tokenizer, model_path = _startup(
            args.model,
            selected_tasks[0][1],
            capacity=args.capacity,
            profile_path=str(args.profile) if args.profile else None,
            pin_top_k=args.pin_top_k,
            warmup=args.warmup,
            expert_policy=args.expert_policy,
        )
    else:
        model, tokenizer, model_path = _load_model_and_tokenizer(args.model, lazy=True)
        mx.eval(model.parameters())
    load_time = time.perf_counter() - started

    rows = []
    for name, prompt_text, grader in selected_tasks:
        prompt = _chat_prompt(tokenizer, prompt_text)
        started = time.perf_counter()
        delta_warmup_seconds = 0.0
        delta_swaps = 0
        if args.backend == "mlx-moe" and args.delta_warmup:
            delta = fast_delta_warmup(
                model,
                tokenizer,
                model_path,
                prompt,
                discovery_tokens=10,
                discovery_method=args.delta_warmup,
            )
            delta_warmup_seconds = delta["total_time"]
            delta_swaps = delta["total_swaps"]
        text = ""
        last = None
        faithful_before = get_faithful_stats(model)
        fallback_before = get_fallback_stats(model)
        grown = 0

        def track_growth(_stats, growth=None):
            nonlocal grown
            if growth:
                grown += growth.get("grown", 0)

        generate_kwargs = {}
        if args.prefill_step_size is not None:
            generate_kwargs["prefill_step_size"] = args.prefill_step_size
        elif args.backend == "mlx-moe" and args.expert_policy == "faithful":
            generate_kwargs["prefill_step_size"] = 16
        responses = mlx_lm.stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.tokens,
            **generate_kwargs,
        )
        if args.backend == "mlx-moe":
            responses = _stream_with_dynamic_updates(
                model,
                responses,
                on_update=track_growth,
                adaptive_growth=args.adaptive_growth,
            )
        for response in responses:
            text += response.text
            last = response
        elapsed = time.perf_counter() - started
        if last is None:
            raise RuntimeError(f"{name} produced no generation responses")

        faithful_after = get_faithful_stats(model)
        fallback_after = get_fallback_stats(model)
        faithful_requests = faithful_after["requests"] - faithful_before["requests"]
        faithful_hits = faithful_after["hits"] - faithful_before["hits"]
        fallback_requests = (
            fallback_after["total_requests"] - fallback_before["total_requests"]
        )
        fallback_misses = (
            fallback_after["total_fallbacks"] - fallback_before["total_fallbacks"]
        )
        try:
            score, detail = grader(text)
        except Exception as exc:
            score, detail = 0.0, f"grader error: {exc}"
        row = {
            "task": name,
            "score": score,
            "detail": detail,
            "prompt_tokens": last.prompt_tokens,
            "generation_tokens": last.generation_tokens,
            "prompt_tps": last.prompt_tps,
            "generation_tps": last.generation_tps,
            "seconds": elapsed,
            "fallback_rate": (
                fallback_misses / fallback_requests if fallback_requests else 0.0
            ),
            "expert_hit_rate": (
                faithful_hits / faithful_requests if faithful_requests else 0.0
            ),
            "expert_load_gb": (
                faithful_after["bytes_loaded"] - faithful_before["bytes_loaded"]
            ) / 1e9,
            "expert_load_seconds": (
                faithful_after["load_seconds"] - faithful_before["load_seconds"]
            ),
            "grown_experts": grown,
            "delta_warmup_seconds": delta_warmup_seconds,
            "delta_swaps": delta_swaps,
            "output": text,
        }
        rows.append(row)
        checkpoint = {
            "status": "running",
            "model": args.model,
            "backend": args.backend,
            "capacity": args.capacity,
            "profile": str(args.profile) if args.profile else None,
            "pin_top_k": args.pin_top_k,
            "expert_policy": (
                args.expert_policy if args.backend == "mlx-moe" else None
            ),
            "adaptive_growth": args.adaptive_growth,
            "delta_warmup": args.delta_warmup,
            "prefill_step_size": generate_kwargs.get("prefill_step_size"),
            "load_time": load_time,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "score": sum(item["score"] for item in rows) / len(rows),
            "tasks": rows,
        }
        (run_dir / "benchmark.json").write_text(json.dumps(checkpoint, indent=2))
        print(
            f"{name}: {score:.0%}, {last.generation_tps:.2f} tok/s, {detail}, "
            f"load={row['expert_load_gb']:.2f} GB/{row['expert_load_seconds']:.1f}s",
            flush=True,
        )

    result = {
        "status": "complete",
        "model": args.model,
        "backend": args.backend,
        "capacity": args.capacity,
        "profile": str(args.profile) if args.profile else None,
        "pin_top_k": args.pin_top_k,
        "expert_policy": args.expert_policy if args.backend == "mlx-moe" else None,
        "adaptive_growth": args.adaptive_growth,
        "delta_warmup": args.delta_warmup,
        "prefill_step_size": generate_kwargs.get("prefill_step_size"),
        "load_time": load_time,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "score": sum(row["score"] for row in rows) / len(rows),
        "tasks": rows,
    }
    (run_dir / "benchmark.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Agent quality benchmark",
        "",
        f"- Model: `{args.model}`",
        f"- Backend: `{args.backend}`",
        f"- Capacity: `{args.capacity or 'auto'}`",
        f"- Expert policy: `{result['expert_policy'] or 'n/a'}`",
        f"- Score: {result['score']:.1%}",
        f"- Peak memory: {result['peak_memory_gb']:.2f} GB",
        "",
        "| Task | Score | Decode tok/s | Expert load GB | Detail |",
        "|---|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {row['task']} | {row['score']:.0%} | {row['generation_tps']:.2f} | "
        f"{row['expert_load_gb']:.2f} | {row['detail']} |"
        for row in rows
    )
    (run_dir / "benchmark.md").write_text("\n".join(lines) + "\n")
    print(f"Score: {result['score']:.1%}; results: {run_dir / 'benchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
