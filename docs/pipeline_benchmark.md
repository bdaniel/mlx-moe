# Two-Mac MLX pipeline benchmark

Compare Qwen3.5-35B-A3B-4bit on the M4 mini with faithful SSD expert loading against the same text checkpoint split across the M4 mini and M1 MacBook Pro over 1 GbE. Run both from a normal macOS terminal.

## Prepare both Macs

Use the same absolute checkout path, Python executable path, and local checkpoint path on both Macs. The checkpoint already exists under `~/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit` on the inspected Mac. Compare `shasum -a 256` for the checkpoint config, tokenizer, template, index and all safetensors shards on both Macs. Keep the checkpoint on local storage on each host.

The project's pinned MLX-LM 0.30.7 lacks Qwen 3.5 pipeline support. Use a separate benchmark environment on both Macs so the project lock stays intact:

```zsh
uv venv --python 3.13 .venv-pipeline
uv pip install --python .venv-pipeline/bin/python -e . 'mlx==0.32.4' 'mlx-lm==0.32.2'

.venv-pipeline/bin/python benchmarks/check_metal.py
.venv-pipeline/bin/python -c 'from mlx_lm.models.qwen3_5 import Qwen3_5TextModel; assert callable(getattr(Qwen3_5TextModel, "pipeline", None))'
```

The preflight confirms that Metal exposes a GPU to the process and runs a small MLX operation before either benchmark loads weights. If it reports no GPU in a managed session, repeat it from Terminal.app. The second check verifies Qwen pipeline support. Use the separate environment for both timed modes.

## Check the Ethernet ring from the M4 mini

Launch both ranks from the mini. With `127.0.0.1` in the `ssh` field, the launcher starts the mini rank locally. The `ips` field still uses the mini's Ethernet address for MLX traffic. Only the mini needs to initiate an SSH login, and both Macs need to exchange MLX ring traffic.

```zsh
set -o pipefail
out="$PWD/logs/pipeline-baseline/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"
```

Save this as `$out/hosts.json` for the direct cable setup. Adjust the MacBook SSH username if needed.

```json
[
  {"ssh": "127.0.0.1", "ips": ["10.77.77.1"]},
  {"ssh": "bdaniel@10.77.77.2", "ips": ["10.77.77.2"]}
]
```

```zsh
ssh -o BatchMode=yes bdaniel@10.77.77.2 hostname
.venv-pipeline/bin/mlx.launch --backend ring --hostfile "$out/hosts.json" \
  --python "$PWD/.venv-pipeline/bin/python" -- \
  "$PWD/benchmarks/check_mlx_ring.py" 2>&1 | tee "$out/ring.log"
```

Both ranks should print `ring OK (sum=3)`. If they hang after printing `initializing MLX ring`, stop the launcher with Ctrl-C and inspect the wired connectivity before loading the checkpoint.

## Run the 35B benchmark from the M4 mini

Set `model` and `runner` to paths valid on both Macs. On each Mac, record `sysctl vm.swapusage` before and after each mode. The JSON lines contain per-rank MLX allocator peaks and full greedy token sequences.

```zsh
model="$HOME/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit"
runner="$PWD/benchmarks/benchmark_pipeline_baseline.py"

.venv-pipeline/bin/python "$runner" faithful "$model" --repeat 3 \
  2>&1 | tee "$out/faithful.log"

.venv-pipeline/bin/mlx.launch --backend ring --hostfile "$out/hosts.json" \
  --python "$PWD/.venv-pipeline/bin/python" -- \
  "$runner" pipeline "$model" --repeat 3 2>&1 | tee "$out/pipeline.log"
```

Check that both ranks report the same prompt length and token IDs. Then compare rank 0's time to first token and decode rate against the single-Mac run. Report the first trial separately because later faithful trials reuse the expert cache. Inspect both ranks' peak memory and swap changes. The defaults exercise a short prompt and up to 64 output tokens. Repeat with the same longer `--prompt-file` on both modes when the smoke comparison completes.

## Single-Mac coding-agent comparison

Run all three models on the same Mac and benchmark environment after the Metal preflight passes. Use the existing scenarios and the same turn, token, and context settings. Bonsai and GPT-OSS run resident. Qwen uses faithful expert loading at capacity 32.

```zsh
bash benchmarks/run_single_mac_comparison.sh
```

The runner defaults to `.venv/bin/python` and checks Metal before starting. Pass another Python executable as the first argument to use a different environment. The dependencies for the two-Mac pipeline are covered under **Prepare both Macs**.

Compare task completion, invalid actions, elapsed time, and peak memory in the agent runs. The Qwen profiling run separately reports expert load calls, bytes per generated token, and the fraction of elapsed time spent in the synchronous expert loader. The load counters cover prefill and decode. Bytes count quantized expert payloads materialized by the loader, which can come from the macOS page cache. Loader time includes the MLX evaluation barrier and cache insertion. The first trial starts with a fresh expert cache, while the OS page cache may already be warm.

## Checkpoint and follow-on

The local checkpoint has 40 text layers. Safetensors headers show 9.463 GB of tensor payload in layers 0–19 and 9.463 GB in 20–39, plus 0.572 GB of embeddings, output head and final norm shared by the stages. A 20-layer stage therefore has about 10.04 GB of stored text weights if it retains all shared tensors. This is a weight estimate before cache, runtime buffers, Metal, and macOS memory.

Use these measurements to decide whether to try Qwen3-Coder-Next with local SSD expert loading on both ranks. That experiment requires each rank to partition its layers before weights materialize, keep original layer indices for expert lookup, and maintain a stage-local memory budget. It is a separate runtime change.
