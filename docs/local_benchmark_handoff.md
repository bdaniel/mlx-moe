# Local MLX Runtime Benchmark Handoff

## Scope

This repo develops a local MLX runtime for sparse MoE models on 16 GB Apple Silicon machines. The current focus is memory-bounded expert loading and local benchmark reliability.

## Current Branch

Branch: `codex/integrate-qwen35-upstream`

Recent commits:

- `351f19c feat: support GPT-OSS lazy expert execution`
- `0a3b0b8 fix: bound adaptive expert growth by memory`

Before editing, inspect the working tree:

```bash
git status --short
git diff --stat
```

## Verified Test State

After the adaptive memory guard changes:

```bash
uv run pytest
```

Observed result: `133 passed`.

## Recent Implementation Work

The runtime now supports an additional sparse MoE layout where experts are stored under layer MLP modules and routed through the layer MLP router.

Important implementation areas:

- `mlx_moe/lazy_experts/loading.py`
- `mlx_moe/lazy_experts/discovery.py`
- `mlx_moe/lazy_experts/modules.py`
- `mlx_moe/lazy_experts/core.py`
- `mlx_moe/lazy_experts/persistence.py`
- `benchmarks/benchmark_dev_agent_loop.py`
- `benchmarks/benchmark_agent_quality.py`
- `benchmarks/test_model.py`
- `tests/test_unit_core.py`

Recent fixes included:

- preserving expert linear bias separately from quantization bias tensors
- disabling KV quantization for a model family whose MLX cache implementation does not support it
- improving benchmark completion parsing for native model response formatting
- adding memory-bounded adaptive capacity growth

## Local Model Paths

Primary model under test:

```text
/Users/bdaniel/.lmstudio/models/mlx-community/gpt-oss-20b-OptiQ-4bit
```

Secondary model previously tested:

```text
/Users/bdaniel/.lmstudio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-AWQ
```

## Benchmark Commands

Primary matrix command:

```bash
uv run --python 3.13 python benchmarks/benchmark_dev_agent_loop.py \
  /Users/bdaniel/.lmstudio/models/mlx-community/gpt-oss-20b-OptiQ-4bit \
  --backend mlx-moe \
  --capacity 20 \
  --max-turns 12 \
  --max-tokens 1024 \
  --thinking-strategy always \
  --context-mode hybrid \
  --adaptive-growth \
  --repeat 3
```

Previously useful single-scenario command:

```bash
uv run --python 3.13 python benchmarks/benchmark_dev_agent_loop.py \
  /Users/bdaniel/.lmstudio/models/mlx-community/gpt-oss-20b-OptiQ-4bit \
  --backend mlx-moe \
  --capacity 24 \
  --scenario timeout_zero \
  --max-turns 8 \
  --max-tokens 1024 \
  --thinking-strategy always \
  --context-mode hybrid \
  --adaptive-growth
```

## Observed Benchmark Results

The primary model at capacity 24 passed the `timeout_zero` scenario in repeated runs after parser and memory fixes:

- score: 100%
- peak memory: about 11 GB
- mean fallback: about 6%
- mean speed: about 12.5 tok/s

Capacity 20 initially looked faster in one single-scenario run:

- score: 100%
- peak memory: about 10.5 GB
- fallback: about 4%
- speed: about 17.6 tok/s

A later full matrix at capacity 20 showed weaker benchmark completion behavior:

- first scenario reached passing code state but did not emit a parseable completion within 12 turns
- observed speed around 10.4 tok/s for that failed run
- continue by inspecting latest artifacts under `logs/dev-agent-loop/`

DeepSeek-Coder-V2-Lite at capacity 48 loaded and ran, but did not follow the benchmark response format:

- result: 0 score
- output was mostly prose rather than valid structured actions
- result file observed: `logs/dev-agent-loop/20260807-014311/benchmark.md`

Capacity 64 for that DeepSeek model hit memory failure during startup.

## Current Open Question

Inspect the newest benchmark artifact under:

```text
logs/dev-agent-loop/
```

Determine whether the primary model's latest failures are caused by:

- model output not reaching completion within the turn budget
- parser missing a valid completion/action format

If parser-related, patch the benchmark parser narrowly and add or update tests.

## Constraints

- Python 3.13 via uv
- use `uv run pytest` for verification
- do not revert unrelated local changes
- avoid destructive git commands
- keep memory bounded for 16 GB Apple Silicon
