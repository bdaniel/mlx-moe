#!/usr/bin/env bash
# Run the model comparison serially so each process releases its GPU memory.
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin="${1:-.venv/bin/python}"
models="$HOME/.lmstudio/models"
"$python_bin" benchmarks/check_metal.py

mkdir -p logs/single-mac-comparison
out="$(mktemp -d "logs/single-mac-comparison/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
printf 'Results directory: %s/%s\n' "$PWD" "$out"
common=(--thinking-strategy off --context-mode hybrid --max-turns 12 --max-tokens 700)
qwen="$models/mlx-community/Qwen3.5-35B-A3B-4bit"
failed=0

run_agent_benchmark() {
    local name="$1"
    shift
    if ! "$python_bin" -u benchmarks/benchmark_dev_agent_loop.py "$@" \
        "${common[@]}" --results-root "$out/$name" 2>&1 | tee "$out/$name.log"; then
        printf '%s failed. See %s/%s.log\n' "$name" "$out" "$name" >&2
        failed=1
    fi
}

run_agent_benchmark bonsai "$models/prism-ml/Ternary-Bonsai-27B-mlx-2bit" --backend mlx-lm
run_agent_benchmark gpt-oss "$models/mlx-community/gpt-oss-20b-OptiQ-4bit" --backend mlx-lm
run_agent_benchmark qwen "$qwen" --backend mlx-moe --expert-policy faithful --capacity 32

if ! "$python_bin" -u benchmarks/benchmark_pipeline_baseline.py faithful \
    "$qwen" --capacity 32 --tokens 128 2>&1 | tee "$out/qwen-expert-loads.log"; then
    printf 'Qwen profiling failed. See %s/qwen-expert-loads.log\n' "$out" >&2
    failed=1
fi

exit "$failed"
