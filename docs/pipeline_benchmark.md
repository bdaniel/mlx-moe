# Two-Mac MLX pipeline benchmark

Compare Qwen3.5-35B-A3B-4bit on the M4 mini with faithful SSD expert loading against the same text checkpoint split across the M4 mini and M1 MacBook Pro over 1 GbE. Run both from a normal macOS terminal.

## Prepare both Macs

Use the same benchmark revision on both Macs. The commands below allow separate checkout paths. The checkpoint already exists under `~/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit` on the inspected Mac. Compare `shasum -a 256` for the checkpoint config, tokenizer, template, index and all safetensors shards on both Macs. Keep the checkpoint on local storage on each host.

The project's pinned MLX-LM 0.30.7 lacks Qwen 3.5 pipeline support. Use a separate benchmark environment on both Macs so the project lock stays intact. These benchmarks can run directly from the checkout with MLX, MLX-LM, and NumPy installed.

```zsh
uv venv --python 3.13 .venv-pipeline &&
uv pip install --python .venv-pipeline/bin/python 'mlx==0.32.4' 'mlx-lm==0.32.2' 'numpy>=2' &&
.venv-pipeline/bin/python benchmarks/check_metal.py &&
.venv-pipeline/bin/python -c 'from mlx_lm.models.qwen3_5 import Qwen3_5TextModel; assert callable(getattr(Qwen3_5TextModel, "pipeline", None))'
```

If the install times out connecting to PyPI, check `curl -I --connect-timeout 10 https://pypi.org/simple/numpy/` from the same terminal. Once it responds, retry from the `uv pip install` line. The existing virtual environment can be reused. For the direct Ethernet setup, Wi-Fi can provide internet access while the Ethernet interface uses its manual address with the Router field blank.

If curl can fetch the package index but uv still times out, try pip in the existing environment:

```zsh
.venv-pipeline/bin/python -m ensurepip --upgrade &&
.venv-pipeline/bin/python -m pip install --timeout 20 --retries 1 \
  'mlx==0.32.4' 'mlx-lm==0.32.2' 'numpy>=2'
```

The preflight confirms that Metal exposes a GPU to the process and runs a small MLX operation before either benchmark loads weights. If it reports no GPU in a managed session, repeat it from Terminal.app. The second check verifies Qwen pipeline support. Use the separate environment for both timed modes.

## Check the Ethernet ring from the M4 mini

To run without SSH, open a local terminal in each Mac's checkout. Run the block below on both Macs, setting `MLX_RANK=0` on the mini and `MLX_RANK=1` on the MacBook. Start the second rank while the first is waiting. The manual hostfile lists TCP endpoints in rank order.

```zsh
export MLX_RANK=0
export MLX_HOSTFILE="$(mktemp)"
printf '%s\n' '[["10.77.77.1:50000"],["10.77.77.2:50001"]]' > "$MLX_HOSTFILE"
.venv-pipeline/bin/python -u benchmarks/check_mlx_ring.py
```

Once both ranks report `ring OK (sum=3)`, run the following on both Macs in the same terminals. Each rank loads its own local checkpoint.

```zsh
model="$HOME/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit"
set -o pipefail
mkdir -p logs/pipeline-baseline
.venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
  pipeline "$model" --tokens 64 --repeat 1 2>&1 \
  | tee "logs/pipeline-baseline/manual-rank-${MLX_RANK}-$(date -u +%Y%m%dT%H%M%SZ).log"
```

If both ranks report `loaded` and Metal times out during prefill, stop any remaining rank. Check Metal locally on both Macs and retry with smaller prefill chunks. Keep fast Metal synchronization unset for the TCP ring.

```zsh
unset MLX_METAL_FAST_SYNCH
.venv-pipeline/bin/python benchmarks/check_metal.py &&
MLX_RING_VERBOSE=1 .venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
  pipeline "$model" --prefill-step-size 4 --tokens 8 --repeat 1
```

This is a diagnostic run. If it completes on both ranks, keep the 4-token prefill chunks and rerun with `--tokens 64` for the speed comparison.

If prefill still times out, retry with `--pipeline-synchronous` on both ranks. This evaluates model output and cache state together and prints progress before each prefill chunk. It records its timing in the same JSON format with `pipeline_synchronous: true`.

If the first synchronous prefill step times out, stop both ranks and isolate the GPU activation handoff without loading a model. Run this on each Mac with the same rank and hostfile settings.

```zsh
unset MLX_METAL_FAST_SYNCH
.venv-pipeline/bin/python benchmarks/check_metal.py &&
MLX_RING_VERBOSE=1 .venv-pipeline/bin/python -u benchmarks/check_mlx_ring.py --gpu
```

Both ranks should report `GPU activation handoff OK`. If only the scalar ring test succeeds, the failure also occurs without Qwen weights or caches. If the activation test succeeds, the next diagnostic is a smaller sharded checkpoint or an isolated Qwen stage.

Use the launcher below if you prefer to start both ranks from one terminal over SSH.

Launch both ranks from the mini. With `127.0.0.1` in the `ssh` field, the launcher starts the mini rank locally. The `ips` field still uses the mini's Ethernet address for MLX traffic. Only the mini needs to initiate an SSH login, and both Macs need to exchange MLX ring traffic.

```zsh
set -o pipefail
mini_repo="$PWD"
macbook_repo="$HOME/Documents/Dev/mlx_moe"
out="$PWD/logs/pipeline-baseline/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"
```

Set `macbook_repo` to the path printed by `pwd` in the MacBook checkout where you installed `.venv-pipeline`.

Save this as `$out/hosts.json` for the direct cable setup. Adjust the MacBook SSH username if needed.

```json
[
  {"ssh": "127.0.0.1", "ips": ["10.77.77.1"]},
  {"ssh": "bdaniel@10.77.77.2", "ips": ["10.77.77.2"]}
]
```

```zsh
ssh -o BatchMode=yes bdaniel@10.77.77.2 hostname
worker='
  mini_repo=$1
  macbook_repo=$2
  shift 2
  case "${MLX_RANK:?}" in
    0) cd "$mini_repo" || exit ;;
    1) cd "$macbook_repo" || exit ;;
    *) exit 1 ;;
  esac
  exec .venv-pipeline/bin/python -u "$@"
'
.venv-pipeline/bin/mlx.launch --backend ring --hostfile "$out/hosts.json" \
  -- bash -c "$worker" mlx-worker "$mini_repo" "$macbook_repo" \
  benchmarks/check_mlx_ring.py 2>&1 | tee "$out/ring.log"
```

Both ranks should print `ring OK (sum=3)`. If they hang after printing `initializing MLX ring`, stop the launcher with Ctrl-C and inspect the wired connectivity before loading the checkpoint.

## Run the 35B benchmark from the M4 mini

Keep using the same terminal session on the mini. On each Mac, record `sysctl vm.swapusage` before and after each mode. The JSON lines contain per-rank MLX allocator peaks and full greedy token sequences.

```zsh
model="$HOME/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit"
runner="benchmarks/benchmark_pipeline_baseline.py"

.venv-pipeline/bin/mlx.launch --backend ring --hostfile "$out/hosts.json" \
  -- bash -c "$worker" mlx-worker "$mini_repo" "$macbook_repo" \
  "$runner" pipeline "$model" --tokens 64 --repeat 1 2>&1 | tee "$out/pipeline.log"

PYTHONPATH="$PWD" .venv-pipeline/bin/python "$runner" faithful "$model" --tokens 64 --repeat 1 \
  2>&1 | tee "$out/faithful.log"
```

Check that both ranks report the same prompt length and token IDs. Then compare rank 0's time to first token and decode rate against the single-Mac run. Report the first trial separately because later faithful trials reuse the expert cache. Inspect both ranks' peak memory and swap changes. The defaults exercise a short prompt and up to 64 output tokens. Repeat with the same longer `--prompt-file` on both modes when the smoke comparison completes.

## Single-Mac coding-agent comparison

Run all three models on the same Mac and benchmark environment after the Metal preflight passes. Use the existing scenarios and the same turn, token, and context settings. Bonsai and GPT-OSS run resident. Qwen uses faithful expert loading at capacity 32.

```zsh
uv sync
bash benchmarks/run_single_mac_comparison.sh
```

`uv sync` sets up the project environment on this Mac. The runner defaults to `.venv/bin/python` and checks Metal before starting. You can also pass the Python executable from an environment with the project and its development dependencies installed.

Compare task completion, invalid actions, elapsed time, and peak memory in the agent runs. The Qwen profiling run separately reports expert load calls, bytes per generated token, and the fraction of elapsed time spent in the synchronous expert loader. The load counters cover prefill and decode. Bytes count quantized expert payloads materialized by the loader, which can come from the macOS page cache. Loader time includes the MLX evaluation barrier and cache insertion. The first trial starts with a fresh expert cache, while the OS page cache may already be warm.

## Checkpoint and follow-on

The local checkpoint has 40 text layers. Safetensors headers show 9.463 GB of tensor payload in layers 0–19 and 9.463 GB in 20–39, plus 0.572 GB of embeddings, output head and final norm shared by the stages. A 20-layer stage therefore has about 10.04 GB of stored text weights if it retains all shared tensors. This is a weight estimate before cache, runtime buffers, Metal, and macOS memory.

Use these measurements to decide whether to try Qwen3-Coder-Next with local SSD expert loading on both ranks. That experiment requires each rank to partition its layers before weights materialize, keep original layer indices for expert lookup, and maintain a stage-local memory budget. It is a separate runtime change.
