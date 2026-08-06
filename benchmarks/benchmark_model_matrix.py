#!/usr/bin/env python3
"""Compare local MLX MoE and dense models in isolated processes.

By default, discovers every complete MoE under the LM Studio model directory.
Dense models are opt-in with repeated ``--dense MODEL`` arguments. Each model
runs in a fresh process so Metal allocations and wired limits cannot contaminate
the next result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = Path.home() / ".lmstudio" / "models"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "logs" / "model-matrix"

PROMPTS = [
    (
        "debugging",
        "Fix this Python function and explain the bug briefly: "
        "`def binary_search(xs, target):\n    lo, hi = 0, len(xs)\n    "
        "while lo <= hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target: "
        "return mid\n        if xs[mid] < target: lo = mid\n        else: hi = mid\n    return -1`",
    ),
    (
        "code_reasoning",
        "A Python service becomes slower after many requests, but RSS stays flat. "
        "Give a concise debugging plan that distinguishes lock contention, event-loop "
        "blocking, and an exhausted connection pool.",
    ),
    (
        "reverse_engineering",
        "You are reverse engineering an undocumented binary protocol. Captures show "
        "the first four bytes change with payload length and the final four bytes change "
        "when any payload byte changes. Describe the next five experiments you would run.",
    ),
]

EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
MLX_MOE_MODEL_TYPES = {"deepseek_v2", "qwen3_5_moe", "qwen3_next"}


@dataclass
class ModelInfo:
    path: str
    name: str
    model_type: str
    architecture: str
    backend: str
    num_experts: int | None
    quant_bits: int | None
    weight_gb: float
    complete: bool
    missing_shards: list[str]


def _config_sections(config: dict[str, Any]) -> list[dict[str, Any]]:
    sections = [config]
    for key in ("text_config", "language_config"):
        value = config.get(key)
        if isinstance(value, dict):
            sections.append(value)
    return sections


def _first_value(sections: list[dict[str, Any]], keys: tuple[str, ...]) -> Any:
    for section in sections:
        for key in keys:
            value = section.get(key)
            if value is not None:
                return value
    return None


def _referenced_shards(model_path: Path) -> tuple[list[Path], list[str]]:
    indexes = sorted(model_path.glob("*.safetensors.index.json"))
    if indexes:
        names: set[str] = set()
        for index in indexes:
            try:
                data = json.loads(index.read_text())
            except (OSError, json.JSONDecodeError):
                return [], [index.name]
            names.update(data.get("weight_map", {}).values())
        shards = [model_path / name for name in sorted(names)]
    else:
        shards = sorted(model_path.glob("*.safetensors"))

    missing = [p.name for p in shards if not p.is_file() or p.stat().st_size == 0]
    if not shards:
        missing.append("no safetensors weights")
    return shards, missing


def inspect_model(model_path: Path) -> ModelInfo | None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    sections = _config_sections(config)
    num_experts = _first_value(sections, EXPERT_KEYS)
    enabled = _first_value(sections, ("enable_moe_block",))
    is_moe = isinstance(num_experts, int) and num_experts > 0 and enabled is not False
    quant = _first_value(sections, ("quantization", "quantization_config"))
    quant_bits = quant.get("bits") if isinstance(quant, dict) else None
    name_bits = re.search(r"(?:^|[-_])(\d+)bit(?:$|[-_])", model_path.name, re.I)
    if name_bits:
        quant_bits = int(name_bits.group(1))
    architecture = (config.get("architectures") or [""])[0]
    model_type = str(config.get("model_type") or _first_value(sections, ("model_type",)) or "")
    shards, missing = _referenced_shards(model_path)
    weight_bytes = sum(p.stat().st_size for p in shards if p.is_file())

    if is_moe and model_type in MLX_MOE_MODEL_TYPES:
        backend = "mlx-moe"
    elif is_moe:
        backend = "mlx-vlm"
    else:
        backend = "mlx-lm"

    return ModelInfo(
        path=str(model_path),
        name=f"{model_path.parent.name}/{model_path.name}",
        model_type=model_type,
        architecture=architecture,
        backend=backend,
        num_experts=int(num_experts) if is_moe else None,
        quant_bits=int(quant_bits) if isinstance(quant_bits, int) else None,
        weight_gb=weight_bytes / 1e9,
        complete=not missing,
        missing_shards=missing,
    )


def discover_models(models_root: Path) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    for config_path in sorted(models_root.glob("*/*/config.json")):
        info = inspect_model(config_path.parent)
        if info is not None:
            models.append(info)
    return models


def resolve_model(value: str, models_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    candidate = models_root / value
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Model not found: {value}")


def _chat_prompt(tokenizer: Any, text: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return text
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    try:
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], **kwargs
        )
    except (TypeError, ValueError):
        kwargs.pop("enable_thinking")
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], **kwargs
            )
        except (TypeError, ValueError):
            return text


def _fallback_delta(model: Any, before: dict[str, Any]) -> tuple[int, int, float]:
    from mlx_moe.lazy_experts.core import get_fallback_stats

    after = get_fallback_stats(model)
    requests = after["total_requests"] - before["total_requests"]
    fallbacks = after["total_fallbacks"] - before["total_fallbacks"]
    return requests, fallbacks, fallbacks / requests if requests else 0.0


def run_worker(
    info: ModelInfo,
    result_path: Path,
    tokens: int,
    capacity: int | None,
    warmup: str,
) -> int:
    import mlx.core as mx
    import mlx_lm

    if info.backend == "mlx-vlm":
        raise ValueError(
            f"{info.model_type} requires the mlx-vlm-lazy benchmark path"
        )

    mx.reset_peak_memory()
    load_start = time.perf_counter()
    if info.backend == "mlx-moe":
        from mlx_moe.lazy_experts.generate import _startup

        model, tokenizer, _ = _startup(
            info.path,
            PROMPTS[0][1],
            capacity=capacity,
            warmup=warmup,
            prepacked=False,
        )
    else:
        from mlx_moe.lazy_experts.generate import _load_model_and_tokenizer

        model, tokenizer, _ = _load_model_and_tokenizer(info.path, lazy=True)
        mx.eval(model.parameters())
    load_time = time.perf_counter() - load_start
    startup_memory_gb = mx.get_active_memory() / 1e9

    actual_capacity = None
    if info.backend == "mlx-moe":
        from mlx_moe.lazy_experts.loading import _find_switch_mlp

        for layer in model.layers:
            switch, _ = _find_switch_mlp(layer)
            if switch is None:
                continue
            cache = getattr(getattr(switch, "gate_proj", None), "_cache", None)
            if cache is not None:
                actual_capacity = cache.capacity
                break

    rows: list[dict[str, Any]] = []
    result = {
        "status": "running",
        "model": asdict(info),
        "capacity": actual_capacity,
        "load_time": load_time,
        "startup_memory_gb": startup_memory_gb,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "prompts": rows,
    }
    result_path.write_text(json.dumps(result, indent=2))

    for prompt_name, prompt_text in PROMPTS:
        prompt = _chat_prompt(tokenizer, prompt_text)
        before = None
        if info.backend == "mlx-moe":
            from mlx_moe.lazy_experts.core import get_fallback_stats

            before = get_fallback_stats(model)

        started = time.perf_counter()
        text = ""
        last_response = None
        growth_total = 0
        growth_events = 0

        responses = mlx_lm.stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=tokens
        )
        if info.backend == "mlx-moe":
            from mlx_moe.lazy_experts.generate import _stream_with_dynamic_updates

            def track_update(_stats: Any, growth: dict[str, Any] | None = None) -> None:
                nonlocal growth_total, growth_events
                grown = growth.get("grown", 0) if growth else 0
                if grown:
                    growth_total += grown
                    growth_events += 1

            responses = _stream_with_dynamic_updates(
                model,
                responses,
                on_update=track_update,
                adaptive_growth=True,
            )

        for response in responses:
            text += response.text
            last_response = response

        elapsed = time.perf_counter() - started
        generated = int(getattr(last_response, "generation_tokens", 0))
        row = {
            "prompt": prompt_name,
            "prompt_tokens": getattr(last_response, "prompt_tokens", None),
            "tokens": generated,
            "seconds": elapsed,
            "tokens_per_second": generated / elapsed if elapsed else 0.0,
            "active_memory_gb": mx.get_active_memory() / 1e9,
            "grown_experts": growth_total,
            "growth_events": growth_events,
            "output": text,
        }
        if before is not None:
            requests, fallbacks, rate = _fallback_delta(model, before)
            row.update(
                fallback_requests=requests,
                fallbacks=fallbacks,
                fallback_rate=rate,
            )
        rows.append(row)
        result["peak_memory_gb"] = mx.get_peak_memory() / 1e9
        result_path.write_text(json.dumps(result, indent=2))
        print(
            f"{prompt_name}: {generated} tokens, {row['tokens_per_second']:.2f} tok/s, "
            f"{row['active_memory_gb']:.2f} GB",
            flush=True,
        )

    result["status"] = "ok"
    result["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    result_path.write_text(json.dumps(result, indent=2))
    return 0


def memory_free_percent() -> int | None:
    try:
        output = subprocess.check_output(
            ["memory_pressure", "-Q"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"free percentage:\s*(\d+)%", output)
    return int(match.group(1)) if match else None


def run_isolated(
    info: ModelInfo,
    run_dir: Path,
    tokens: int,
    capacity: int | None,
    warmup: str,
    min_free_percent: int,
    memory_guard_samples: int,
    timeout: int,
) -> dict[str, Any]:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", info.name).strip("-")
    result_path = run_dir / f"{slug}.json"
    log_path = run_dir / f"{slug}.log"
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        info.path,
        "--result",
        str(result_path),
        "--tokens",
        str(tokens),
        "--warmup",
        warmup,
    ]
    if capacity is not None:
        cmd.extend(["--capacity", str(capacity)])

    started = time.monotonic()
    low_samples = 0
    with log_path.open("w") as log:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        reason = None
        while process.poll() is None:
            if time.monotonic() - started > timeout:
                reason = f"timeout after {timeout}s"
                break
            free = memory_free_percent() if min_free_percent > 0 else None
            low_samples = (
                low_samples + 1
                if free is not None and free < min_free_percent
                else 0
            )
            if min_free_percent > 0 and low_samples >= memory_guard_samples:
                reason = (
                    "memory guard: free percentage stayed below "
                    f"{min_free_percent}% for {memory_guard_samples}s"
                )
                break
            time.sleep(1)

        if reason is not None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("status") != "ok":
            result["status"] = "partial" if result.get("prompts") else "failed"
            result["error"] = reason or f"worker exited {process.returncode}"
            result["log"] = str(log_path)
        return result
    tail = log_path.read_text(errors="replace").splitlines()[-20:]
    return {
        "status": "failed",
        "model": asdict(info),
        "error": reason or f"worker exited {process.returncode}",
        "log": str(log_path),
        "log_tail": tail,
    }


def write_summary(run_dir: Path, results: list[dict[str, Any]]) -> None:
    payload = {"created_at": datetime.now().isoformat(), "results": results}
    (run_dir / "benchmark.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Local model matrix",
        "",
        "| Model | Backend | Cap | Load s | Peak GB | tok/s | Fallback | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        model = result["model"]
        rows = result.get("prompts", [])
        tps = sum(r["tokens_per_second"] for r in rows) / len(rows) if rows else None
        fb_rows = [r["fallback_rate"] for r in rows if "fallback_rate" in r]
        fallback = sum(fb_rows) / len(fb_rows) if fb_rows else None
        lines.append(
            f"| {model['name']} | {model['backend']} | {result.get('capacity') or '-'} | "
            f"{result.get('load_time', 0):.1f} | {result.get('peak_memory_gb', 0):.2f} | "
            f"{tps:.2f} | {fallback:.1%} | {result['status']} |"
            if tps is not None and fallback is not None
            else f"| {model['name']} | {model['backend']} | {result.get('capacity') or '-'} | "
            f"{result.get('load_time', 0):.1f} | {result.get('peak_memory_gb', 0):.2f} | "
            f"{tps:.2f} | - | {result['status']} |"
            if tps is not None
            else f"| {model['name']} | {model['backend']} | - | - | - | - | - | {result['status']} |"
        )
    (run_dir / "benchmark.md").write_text("\n".join(lines) + "\n")


def print_inventory(models: list[ModelInfo]) -> None:
    print(f"{'Model':58} {'Kind':8} {'Experts':>7} {'Bits':>4} {'Weights':>8} State")
    print("-" * 106)
    for info in models:
        state = "ready" if info.complete else "incomplete"
        print(
            f"{info.name[:58]:58} {info.backend:8} "
            f"{str(info.num_experts or '-'):>7} {str(info.quant_bits or '-'):>4} "
            f"{info.weight_gb:7.1f}G {state}"
        )


def summarize_existing(results_root: Path) -> Path:
    latest_by_model: dict[str, dict[str, Any]] = {}
    for path in sorted(results_root.glob("*/benchmark.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for result in payload.get("results", []):
            model_path = result.get("model", {}).get("path")
            if not model_path:
                continue
            previous = latest_by_model.get(model_path)
            if previous is None or result.get("status") == "ok":
                latest_by_model[model_path] = result

    if not latest_by_model:
        raise ValueError(f"No benchmark results found under {results_root}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_root / f"summary-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_summary(run_dir, list(latest_by_model.values()))
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--dense", action="append", default=[], metavar="MODEL")
    parser.add_argument("--model", action="append", default=[], metavar="MODEL")
    parser.add_argument("--list", action="store_true", help="Inventory models without running")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Roll latest per-model runs into one report without generating",
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--warmup", choices=("hybrid", "full", "none"), default="hybrid")
    parser.add_argument(
        "--min-free-percent",
        type=int,
        default=0,
        help="Opt-in memory_pressure cutoff; 0 disables it (default)",
    )
    parser.add_argument("--memory-guard-samples", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--worker", metavar="MODEL", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        info = inspect_model(Path(args.worker))
        if info is None or args.result is None:
            raise ValueError("Worker requires a valid model and --result")
        return run_worker(info, args.result, args.tokens, args.capacity, args.warmup)

    inventory = discover_models(args.models_root.expanduser())
    print_inventory(inventory)
    if args.list:
        return 0
    if args.summarize_existing:
        run_dir = summarize_existing(args.results_root)
        print(f"\nResults: {run_dir / 'benchmark.md'}")
        return 0

    if args.model:
        selected_paths = [resolve_model(value, args.models_root) for value in args.model]
        selected = [inspect_model(path) for path in selected_paths]
    else:
        selected = [info for info in inventory if info.backend == "mlx-moe"]
        selected.extend(
            inspect_model(resolve_model(value, args.models_root)) for value in args.dense
        )
    selected = [info for info in selected if info is not None]
    selected = list({info.path: info for info in selected}.values())

    incomplete = [info for info in selected if not info.complete]
    for info in incomplete:
        print(f"Skipping incomplete {info.name}: {', '.join(info.missing_shards[:3])}")
    selected = [info for info in selected if info.complete]
    if not selected:
        print("No complete models selected.")
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.results_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    for index, info in enumerate(selected, 1):
        print(f"\n[{index}/{len(selected)}] {info.name} ({info.backend})", flush=True)
        result = run_isolated(
            info,
            run_dir,
            args.tokens,
            args.capacity,
            args.warmup,
            args.min_free_percent,
            args.memory_guard_samples,
            args.timeout,
        )
        results.append(result)
        write_summary(run_dir, results)
        print(f"  {result['status']}", flush=True)

    print(f"\nResults: {run_dir / 'benchmark.md'}")
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
