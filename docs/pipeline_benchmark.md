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

To test the LM Studio Qwen 3.5 0.8B MLX checkpoint, run this on both Macs with the same rank and hostfile settings:

```zsh
.venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
  pipeline "$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit" \
  --pipeline-synchronous --prefill-step-size 4 --tokens 8 --repeat 1
```

If the smaller model completes but the 35B model times out during prefill, run this on both Macs to time each loaded 35B shard locally:

```zsh
.venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
  pipeline "$HOME/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit" \
  --pipeline-stage-only --prefill-step-size 4
```

Both ranks participate in sharded loading. Each then runs its own layers with local token embeddings as input. Compare `forward_s`, the layer ranges, and peak allocation in the `stage_probe` records.

If both local forwards finish but the distributed 35B forward times out, try the host-synchronized mode on both Macs. It completes the upstream GPU computation before sending activations and completes the TCP receive before submitting work to the downstream GPU.

```zsh
unset MLX_METAL_FAST_SYNCH
.venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
  pipeline "$HOME/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit" \
  --pipeline-host-sync --prefill-step-size 4 --tokens 8 --repeat 1
```

## Benchmark distinct coding prompts

After the short host-synchronized trial completes, run this on both Macs with the same rank and hostfile settings. Use the same benchmark script and prompt suite on both ranks.

```zsh
model="$HOME/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit"
set -o pipefail
mkdir -p logs/pipeline-baseline
.venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
  pipeline "$model" --pipeline-host-sync --prefill-step-size 4 \
  --prompt-suite benchmarks/pipeline_coding_prompts.json --tokens 128 --repeat 1 2>&1 \
  | tee "logs/pipeline-baseline/coding-rank-${MLX_RANK}-$(date -u +%Y%m%dT%H%M%SZ).log"
```

Each rank loads once and runs the three coding prompts in file order. Each prompt starts with a fresh generation cache. Compare the rank-zero time to first token and decode rate by `prompt_name`, and check that both ranks produced identical token IDs for each prompt. The first prompt also includes any one-time GPU work after loading.

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

## Run the coding prompt suite on resident single-Mac models

Run Bonsai 27B and GPT-OSS 20B on the mini to compare against the two-Mac coding prompt runs. Keep the same prompts, prefill chunk size, and output-token budget. Each model runs in a separate process so its allocations are released before the next model starts.

```zsh
set -o pipefail
mkdir -p logs/pipeline-baseline
(
  set -e
  .venv-pipeline/bin/python benchmarks/check_metal.py
  for checkpoint in \
    prism-ml/Ternary-Bonsai-27B-mlx-2bit \
    mlx-community/gpt-oss-20b-OptiQ-4bit; do
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
      .venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
      resident "$HOME/.lmstudio/models/$checkpoint" --prefill-step-size 16 \
      --prompt-suite benchmarks/pipeline_coding_prompts.json --tokens 128 --repeat 2 2>&1 \
      | tee "logs/pipeline-baseline/${checkpoint##*/}-resident-$(date -u +%Y%m%dT%H%M%SZ).log"
  done
)
```

Compare model load, time to first token, decode rate, and peak MLX memory. Each model uses its own tokenizer and chat template, so record prompt and output lengths alongside timing. GPT-OSS can spend its token budget on reasoning. Use the coding-agent benchmark below for test passes and task completion.

### Test Bonsai and GPT-OSS sharding

Use the same Ethernet hostfile and ranks as the Qwen run. Both Macs need local copies of the same checkpoint. Bonsai uses layer pipelining with the host-synchronized handoff. GPT-OSS uses MLX-LM's tensor parallelism, which splits work within layers and exchanges intermediate results during each forward pass.

Sync `benchmarks/benchmark_pipeline_baseline.py` from the MacBook checkout to `~/Dev/mlx-moe/benchmarks/` on the mini before running. Compare any existing edits on the mini before replacing the file. Run `shasum -a 256 benchmarks/benchmark_pipeline_baseline.py benchmarks/pipeline_coding_prompts.json` from each checkout and check that the hashes agree.

Run one pair at a time. Start with eight output tokens and a prefill chunk of four on both ranks.

```zsh
unset MLX_METAL_FAST_SYNCH
set -o pipefail
mkdir -p logs/pipeline-baseline
.venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
  pipeline "$HOME/.lmstudio/models/prism-ml/Ternary-Bonsai-27B-mlx-2bit" \
  --pipeline-host-sync --prefill-step-size 4 --tokens 8 --repeat 1 2>&1 \
  | tee "logs/pipeline-baseline/bonsai-sharded-rank-${MLX_RANK}-$(date -u +%Y%m%dT%H%M%SZ).log"
```

```zsh
.venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
  tensor "$HOME/.lmstudio/models/mlx-community/gpt-oss-20b-OptiQ-4bit" \
  --prefill-step-size 4 --tokens 8 --repeat 1 2>&1 \
  | tee "logs/pipeline-baseline/gpt-oss-sharded-rank-${MLX_RANK}-$(date -u +%Y%m%dT%H%M%SZ).log"
```

Check both ranks for matching tokens, successful completion, and peak allocation. If a rank times out, save both logs and stop any rank still waiting before starting the next pair. For GPT-OSS, rank agreement also checks that both workers stayed in step through token generation. After a smoke test passes, compare with the resident run using `--prefill-step-size 16 --prompt-suite benchmarks/pipeline_coding_prompts.json --tokens 128 --repeat 2` on both ranks. For GPT-OSS final-answer comparisons, use the same effort and 1,024-token cap on the resident and tensor runs. Report rank agreement separately from resident-to-sharded token parity.

Both eight-token smokes passed on MLX 0.32.4 and MLX-LM 0.32.2. The ranks matched all eight tokens in each run and completed without timeouts.

| Model and mode | TTFT, Mini / M1 Pro | Elapsed, Mini / M1 Pro | Generation peak, Mini / M1 Pro | Load peak, Mini / M1 Pro |
| --- | ---: | ---: | ---: | ---: |
| Bonsai 27B layer pipeline | 1.15 / 1.14 s | 1.67 / 1.67 s | 4.28 / 4.29 GB | 4.14 / 4.14 GB |
| GPT-OSS 20B tensor | 4.67 / 4.66 s | 5.33 / 5.32 s | 6.30 / 6.30 GB | 7.18 / 7.18 GB |

The resident reference used the three coding prompts with prefill 16, up to 128 output tokens per prompt, and two passes. Suite times below sum the second pass and exclude loading. Use these timings when the two-rank suite with matching settings completes.

| Model | Mini warm suite | M1 Pro warm suite | Resident generation peak, Mini / M1 Pro |
| --- | ---: | ---: | ---: |
| Bonsai 27B | 65.57 s | 55.96 s | 7.90 / 7.93 GB |
| GPT-OSS 20B, medium | 21.36 s | 33.79 s | 11.68 / 11.70 GB |

Sources are the [Mini resident logs](https://chatgpt.com/library?file_id=libfile_55801f1ffee8819182df3c6d4a6f1fd1&account_id=personal) and the MacBook logs under `logs/pipeline-baseline/`. The GPT-OSS M1 Pro reference uses the idle rerun.

At the matched coding settings, the Bonsai pipeline took 47.30 s on pass one and 47.15 s on pass two on rank 0, with 4.37 GB and 4.39 GB peak generation allocation on the Mini and M1 Pro, respectively. Decode held at 13.17–13.37 tok/s. The warm pass was 1.39× as fast as the Mini resident reference and 1.19× as fast as the M1 Pro resident reference. All six 128-token sequences matched across both ranks and both resident baselines. Sources are the [Mini coding log](https://chatgpt.com/library?file_id=libfile_9faa9041ff78819198fb385d23cc7839&account_id=personal) and `logs/pipeline-baseline/bonsai-coding-rank-1-20261119T044916Z.log`.

At the same settings, the GPT-OSS tensor run took 68.04 s and 69.73 s over the two passes on rank 0. Peak generation allocation was 6.34 GB on the Mini and 6.35 GB on the M1 Pro. Both peaked at 7.18 GB during load. The warm pass took 3.26× the Mini resident time and 2.06× the M1 Pro resident time. Both ranks matched all six 128-token sequences. Each trial hit the cap without final-channel text. Compared with both resident baselines, the first differing generated token was at positions 8, 10, and 63 for the three prompts, respectively, on both passes. Sources are the [Mini coding log](https://chatgpt.com/library?file_id=libfile_887863dfc50c81918fff44f9893eb7fd&account_id=personal) and `logs/pipeline-baseline/gpt-oss-coding-rank-1-20261119T045133Z.log`.

### Measure GPT-OSS time to a final answer

Run this independently on each Mac. It uses the same prompts, prefill chunk size, and 1,024-token budget as the medium-reasoning run. Omit `--reasoning-effort low` to use the checkpoint's default medium effort.

```zsh
set -o pipefail
mkdir -p logs/pipeline-baseline
(
  sysctl vm.swapusage
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
    .venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
    resident "$HOME/.lmstudio/models/mlx-community/gpt-oss-20b-OptiQ-4bit" \
    --prefill-step-size 16 --prompt-suite benchmarks/pipeline_coding_prompts.json \
    --tokens 1024 --repeat 2 --reasoning-effort low
  status_code=$?
  sysctl vm.swapusage
  exit "$status_code"
) 2>&1 | tee "logs/pipeline-baseline/gpt-oss-low-$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ).log"
```

`ttft_final_s` measures time from generation start to the first content token in the final answer. It is null when the run generates no final-answer content. Compare it with `ttft_s`, `final_answer_tokens`, and `finish_reason` to see how long reasoning took and whether the output hit its cap.

The medium baseline used prefill 16 and a 1,024-token cap over two passes through the three prompts. Counts below are over six trials per Mac. Latencies are shown in pass order and cover only trials that reached final content.

| Mac | Final answers started | Natural stops | Prompt reaching final text | Time to first final token | Final tokens per answer | Warm suite time |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| M4 mini | 2/6 | 0/6 | `async_singleflight` | 15.75 / 15.61 s | 520 | 86.9 s |
| M1 Pro | 2/6 | 2/6 | `timeout_zero` | 34.42 / 23.46 s | 381 | 132.0 s |

The mini hit the cap on all six trials. The M1 Pro hit it on the other four trials. Compare latency at low effort by host and prompt because the medium runs reached different final answers. Sources are [M4 medium log](https://chatgpt.com/library?file_id=libfile_c55dd51bd9f88191b288be6a7c849055&account_id=personal) and `logs/pipeline-baseline/gpt-oss-final-Router-20261119T041456Z.log` on the MacBook.

The low-effort runs matched the baseline's checkpoint, prompt suite, prefill, and token cap.

| Mac | Final answers started | Natural stops | Natural stops after final text | Suite time, pass 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| M4 mini | 6/6 | 2/6 | 2/6 | 78.8 / 72.0 s |
| M1 Pro | 4/6 | 4/6 | 2/6 | 160.4 / 110.3 s |

| Mac | Prompt | First final token, pass 1 / 2 | Final tokens per pass | Finish |
| --- | --- | ---: | ---: | --- |
| M4 mini | `timeout_zero` | 9.52 / 4.62 s | 368 | Stop |
| M4 mini | `async_singleflight` | 18.83 / 17.57 s | 424 | Length |
| M4 mini | `benchmark_metrics_review` | 6.83 / 6.81 s | 998 | Length |
| M1 Pro | `timeout_zero` | 24.36 / 6.81 s | 395 | Stop |
| M1 Pro | `async_singleflight` | — | 0 | Stop in commentary |
| M1 Pro | `benchmark_metrics_review` | 10.08 / 7.88 s | 998 | Length |

On the M1 Pro, `async_singleflight` emitted a patch in the Harmony commentary channel and stopped with `<|return|>` in both passes. It contributes two natural stops and zero final-channel starts. On both Macs, only `timeout_zero` reached final text and stopped naturally at low effort. Sources are [M4 low log](https://chatgpt.com/library?file_id=libfile_3cb14ca2f1f48191af12bf9197fc61fb&account_id=personal) and `logs/pipeline-baseline/gpt-oss-low-Router-20261119T043348Z.log` on the MacBook.

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

### Qwen3.5-35B coding suite results

Each row is one pass over the same three prompts with up to 128 output tokens per prompt, using MLX 0.32.4 and MLX-LM 0.32.2. Times exclude model load. Pipeline times come from the mini, the output rank.

| Setup | Suite time | Peak MLX allocation per host | Decode |
| --- | ---: | ---: | ---: |
| M1 Pro SSD, capacity 64 | 723.4 s | 6.49 GB | 1.6–2.2 tok/s |
| M4 mini SSD, capacity 64 | 305.5 s | 6.46 GB | 3.8–4.7 tok/s |
| M4 mini SSD, capacity 112 | 258.9 s | 9.66 GB | 4.2–6.0 tok/s |
| M4 mini SSD, capacity 118 | 264.3 s | 10.06 GB | 4.2–5.3 tok/s |
| M4 mini + M1 Pro, resident pipeline | 50.6 s | ~10.1 GB each | 39.0–39.5 tok/s |

The pipeline was 5.2× faster than the capacity-118 mini run at nearly equal peak allocation on the mini. The pipeline also used about 10.1 GB on the MacBook. Both pipeline ranks agreed on all tokens. The mini SSD runs agreed across capacities, and one of the three prompts matched across all setups.

### Qwen3.5-122B SSD smoke test

Check each local checkpoint before starting Metal. The command reads only the safetensors headers to check that all files referenced by the index exist and that their payload sizes match. Set `model` to the local checkpoint directory on each Mac.

```zsh
model="$HOME/.lmstudio/models/mlx-community/Qwen3.5-122B-A10B-4bit"
.venv-pipeline/bin/python - "$model" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
if config.get("model_type") != "qwen3_5_moe":
    raise SystemExit(f"Unexpected model type in {root}")
index = json.loads((root / "model.safetensors.index.json").read_text())
keys_by_shard = {}
for key, shard_name in index["weight_map"].items():
    keys_by_shard.setdefault(shard_name, set()).add(key)
if not keys_by_shard:
    raise SystemExit(f"No shards in {root}")
total_bytes = 0
for name, keys in sorted(keys_by_shard.items()):
    shard = root / name
    with shard.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        if not 1 <= header_size <= 100_000_000:
            raise SystemExit(f"Invalid safetensors header length in {shard}")
        header = json.loads(handle.read(header_size))
    missing = keys - header.keys()
    if missing:
        raise SystemExit(f"{shard}: missing indexed tensor {min(missing)}")
    payload_size = max(
        tensor["data_offsets"][1]
        for key, tensor in header.items()
        if key != "__metadata__"
    )
    expected_size = 8 + header_size + payload_size
    actual_size = shard.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(f"{shard}: expected {expected_size} bytes, found {actual_size}")
    total_bytes += actual_size
print(f"{len(keys_by_shard)} indexed shards, {total_bytes:,} bytes ({total_bytes / 2**30:.2f} GiB)")
PY
```

The MacBook copy at the path above passed the index and header check with 14 shards totaling 69,593,636,939 bytes (64.81 GiB). Its config reports 48 layers and 256 experts. Verify the Mini copy locally and record the storage type for each checkpoint alongside the timings.

After the check passes, run this on the MacBook with the short default prompt and an eight-token budget. Record startup, time to first token, peak MLX allocation, and expert loader counters.

```zsh
set -o pipefail
mkdir -p logs/pipeline-baseline
model="$HOME/.lmstudio/models/mlx-community/Qwen3.5-122B-A10B-4bit"
.venv-pipeline/bin/python benchmarks/check_metal.py &&
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
    faithful "$model" --capacity 8 --prefill-step-size 4 --tokens 8 --repeat 1 2>&1 \
  | tee "logs/pipeline-baseline/122b-ssd-$(date -u +%Y%m%dT%H%M%SZ).log"
```

The M1 Pro faithful smoke passed at capacity 8 and prefill 4. End-to-end startup took 3.05 s. Generation took 101.96 s for eight tokens, with 76.11 s to the first token and 0.31 decode tok/s. Peak MLX allocation was 10.16 GB. Expert loading materialized 39.07 GB of payload across 812 calls and took 90.53 s (88.8% of elapsed generation time). The payload counter measures bytes materialized by the loader and may include data served from the OS page cache. The token IDs were `[8160, 369, 264, 12654, 709, 310, 10562, 48142]`. The source is `logs/pipeline-baseline/122b-ssd-20261119T052311Z.log`.

With the single-Mac smoke complete, run on both Macs with the Ethernet hostfile and each Mac's verified local checkpoint path:

```zsh
unset MLX_METAL_FAST_SYNCH
set -o pipefail
mkdir -p logs/pipeline-baseline
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
    pipeline-ssd "$model" --capacity 8 --prefill-step-size 4 --tokens 8 --repeat 1 2>&1 \
  | tee "logs/pipeline-baseline/122b-pipeline-ssd-rank-${MLX_RANK}-$(date -u +%Y%m%dT%H%M%SZ).log"
```

Expect rank 0 to report layers `[24, 48)` and rank 1 to report `[0, 24)`. Record both ranks' token IDs, peak MLX allocation, expert bytes and load time, and any timeouts. Check rank agreement before comparing tokens or timings with the single-Mac faithful run.

The two-Mac smoke passed at capacity 8 and prefill 4. The ranks processed their expected layer ranges and agreed on all eight tokens. Times exclude startup.

| Mode and host | TTFT | Elapsed | Decode | Peak MLX allocation | Expert bytes materialized | Expert load time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Faithful, M1 Pro | 76.11 s | 101.96 s | 0.31 tok/s | 10.16 GB | 39.07 GB | 90.53 s |
| Pipeline SSD, Mini rank 0 | 46.48 s | 61.11 s | 0.48 tok/s | 5.36 GB | 15.54 GB | 22.93 s |
| Pipeline SSD, M1 Pro rank 1 | 46.47 s | 61.10 s | 0.48 tok/s | 5.85 GB | 20.86 GB | 39.13 s |

The sharded smoke took 40% less wall time and reduced peak MLX allocation on the M1 Pro by 42%. These are end-to-end observations from the short runs. The sharded tokens were `[71093, 12305, 198, 1445, 18904, 1120, 1697, 11]`, diverging from the single-Mac run at token one. The two runners use different prefill boundaries even with the same chunk size. For this 31-token prompt, MLX-LM streaming reserves one prompt token for sampling and runs a two-token prefill chunk before it. The synchronous pipeline samples from a final three-token chunk. Prefill 3 would align the prompt chunk boundaries for a follow-up parity check. The rank 1 source is `logs/pipeline-baseline/122b-pipeline-ssd-rank-1-20261119T052619Z.log`. Rank 0 metrics came from the Mini output shared in chat.

The prefill-3 follow-up split the 31-token prompt into ten three-token chunks and one final single-token chunk. Both ranks and the matched single-Mac faithful run produced `[8160, 369, 264, 12654, 709, 310, 10562, 48142]`. The pipeline reduced elapsed generation time by 40.7%, time to first token by 42.5%, and peak MLX allocation on the M1 Pro by 43.4%. These are eight-token smoke measurements with capacity 8.

| Mode and host, prefill 3 | TTFT | Elapsed | Decode | Peak MLX allocation | Expert bytes materialized | Expert load time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Faithful, M1 Pro | 75.34 s | 100.75 s | 0.32 tok/s | 8.94 GB | 40.29 GB | 90.04 s |
| Pipeline SSD, Mini rank 0 | 43.35 s | 59.73 s | 0.43 tok/s | 4.87 GB | 17.07 GB | 16.99 s |
| Pipeline SSD, M1 Pro rank 1 | 43.34 s | 59.72 s | 0.43 tok/s | 5.06 GB | 21.96 GB | 37.51 s |

The faithful source is `logs/pipeline-baseline/122b-ssd-20261119T053341Z.log`. The rank 1 source is `logs/pipeline-baseline/122b-pipeline-ssd-rank-1-20261119T052945Z.log`. Rank 0 metrics came from the Mini output shared in chat.

The 32-token profile used capacity 8, prefill 3, and two trials in one process. The `pipeline_timings` record splits time into prefill through the first sampled token and decode for the remaining tokens:

- `local_forward_s` includes local model work, cache evaluation, and expert loading. The existing `experts.load_seconds` counter measures time inside the expert loader over the full trial, and overlaps with local forward time.
- `activation_handoff_s` measures the blocking send on rank 1 and the blocking receive on rank 0. The receive duration can include waiting for rank 1 to finish its stage.
- `token_sync_s` measures the token collective, which can include waiting for the other rank. Compare the timings per rank because their waits and forwards can overlap.

Both ranks matched all 32 token IDs in both trials. Rank 0 elapsed time was 127.14 s and 124.75 s, with decode at 0.375 and 0.384 tok/s. Rank 1 expert misses fell from 7,489 to 7,437 on the second trial, and payload materialized fell from 39.75 GB to 39.48 GB. The second trial saved 1.9% wall time.

| Second trial | Prefill forward | Decode forward | Decode activation send or receive | Decode token collective | Expert loading, whole trial | Peak MLX allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mini rank 0 | 26.43 s | 16.83 s | 63.80 s receive | 0.04 s | 29.85 s | 4.87 GB |
| M1 Pro rank 1 | 42.84 s | 63.78 s | 0.006 s send | 16.88 s | 64.62 s | 5.06 GB |

The Mini's decode receive time tracks the M1 Pro's forward time, and the M1 Pro's token collective time tracks the Mini's forward time. The stages ran in sequence during decode. The M1 Pro's expert loader accounted for 64.62 s of its 106.62 s of local forward time across both phases. For a cache-size comparison, keep the 32-token workload and prefill fixed and try capacity 32 on both ranks. Compare M1 Pro misses, loader time, rank agreement, and peak allocation with the capacity-8 profile. The rank 1 source is `logs/pipeline-baseline/122b-profile-rank-1-20261119T053749Z.log`. Rank 0 metrics came from the Mini output shared in chat.

Capacity 32 preserved all 32 token IDs across both ranks and both trials, including agreement with capacity 8. The first trial took 125.18 s on rank 0 compared with 127.14 s at capacity 8. The second trial took 117.47 s, down from 124.75 s (5.8%). Its TTFT improved from 44.08 s to 39.23 s (11.0%), and decode rose from 0.384 to 0.396 tok/s (3.1%).

| Second trial | Mini capacity 8 → 32 | M1 Pro capacity 8 → 32 |
| --- | ---: | ---: |
| Expert misses | 5,801 → 2,975 | 7,437 → 5,264 |
| Expert payload materialized | 30.79 → 15.79 GB | 39.48 → 27.94 GB |
| Expert load time | 29.85 → 24.88 s | 64.62 → 53.13 s |
| Peak MLX allocation | 4.87 → 6.50 GB | 5.06 → 6.53 GB |
| Local forward during decode | 16.83 → 18.40 s | 63.78 → 59.78 s |

The M1 Pro still spent about 3.25× as long in its decode forward as the Mini. The next comparison can isolate that stage by keeping rank 0 at capacity 32 and raising only rank 1 to capacity 64, with the same prefill and token budget. Capacity applies to the local expert cache. Compare both passes and watch rank 1 peak allocation alongside misses, load time, and end-to-end latency. The capacity-32 rank 1 source is `logs/pipeline-baseline/122b-profile-rank-1-20261119T054912Z.log`. Rank 0 metrics came from the Mini output shared in chat.

With capacity 32 on the Mini and 64 on the M1 Pro, both ranks matched all 32 token IDs from the earlier profiles in both trials. On rank 0, the first trial saved 1.30 s overall. TTFT increased from 44.19 s to 51.10 s, while decode improved from 0.383 to 0.426 tok/s. On the second trial, the pipeline saved 11.67 s overall and almost all of the gain came after the first token.

| Second trial | Mini 32 / M1 Pro 32 | Mini 32 / M1 Pro 64 |
| --- | ---: | ---: |
| TTFT, rank 0 | 39.23 s | 39.06 s |
| Elapsed, rank 0 | 117.47 s | 105.81 s |
| Decode | 0.396 tok/s | 0.464 tok/s |
| M1 Pro decode forward | 59.78 s | 47.66 s |
| M1 Pro expert misses | 5,264 | 2,946 |
| M1 Pro expert load time | 53.13 s | 37.54 s |
| Peak MLX allocation, Mini / M1 Pro | 6.50 / 6.53 GB | 6.50 / 10.59 GB |

The larger M1 Pro cache improved second-trial decode by 17.2% and used 4.05 GB more peak MLX allocation on the M1 Pro. The next prefill comparison can hold capacities at 32/64 and change only `--prefill-step-size 6` on both ranks, with the same 32-token budget and two trials. The 31-token prompt then runs in five six-token chunks and one final single-token chunk. Compare TTFT, peak allocation, and token IDs. The capacity-64 rank 1 source is `logs/pipeline-baseline/122b-profile-rank-1-20261119T055701Z.log`. Rank 0 metrics came from the Mini output shared in chat.

At capacities 32/64, prefill 6 preserved all 32 token IDs across both ranks and both trials, including agreement with prefill 3. It reduced TTFT by 7.59 s on the first trial and 3.50 s on the second. Decode was about 8% slower in both trials, leaving total time within 2.3 s of the prefill-3 profile. The measurements use the same 31-token prompt and report rank 0 timings.

| Metric | Prefill 3, trial 1 / 2 | Prefill 6, trial 1 / 2 |
| --- | ---: | ---: |
| TTFT | 51.10 / 39.06 s | 43.51 / 35.56 s |
| Elapsed | 123.88 / 105.81 s | 122.66 / 108.08 s |
| Decode | 0.426 / 0.464 tok/s | 0.392 / 0.427 tok/s |
| Mini peak MLX allocation | 6.51 / 6.50 GB | 6.66 / 6.70 GB |
| M1 Pro peak MLX allocation | 10.55 / 10.59 GB | 10.65 / 10.75 GB |

Prefill 3 remains the faster second-trial configuration for generating 32 tokens. Prefill 6 had the lower TTFT in both trials. The prefill-6 rank 1 source is `logs/pipeline-baseline/122b-profile-rank-1-20261119T060442Z.log`. Rank 0 metrics came from the Mini output shared in chat.

The stage-placement test used `--pipeline-split 32 16` on both ranks, prefill 3, 32 generated tokens, and two trials. Capacity remained 32 on the Mini and 64 on the M1 Pro. Rank 0 ran `[16, 48)` and rank 1 ran `[0, 16)`. All 32 token IDs matched across ranks, trials, and the previous 24/24 split.

| Metric | 24/24 layers | 32/16 layers |
| --- | ---: | ---: |
| First-trial elapsed, rank 0 | 123.88 s | 112.67 s |
| Second-trial TTFT, rank 0 | 39.06 s | 36.67 s |
| Second-trial elapsed, rank 0 | 105.81 s | 96.93 s |
| Second-trial decode | 0.464 tok/s | 0.514 tok/s |
| Second-trial decode forward, Mini / M1 Pro | 19.02 / 47.66 s | 36.97 / 23.24 s |
| Second-trial peak MLX allocation, Mini / M1 Pro | 6.50 / 10.59 GB | 8.34 / 7.42 GB |

The new split reduced second-trial elapsed time by 8.4% and the larger of the two per-host MLX peaks by 21.2%. Rank 0 reported 31.85 s for startup versus 20.23 s in the 24/24 run, so startup plus first generation was about 144 s in both runs. Startup includes waiting for the peer. The Mini accounted for the larger share of second-trial decode forward and spent 46.02 s in the expert loader across both phases. A follow-up cache test can raise only Mini capacity to 48 with the same split and workload. The rank 1 source is `logs/pipeline-baseline/122b-split32-16-rank-1-20261119T071529Z.log`. Rank 0 metrics came from the Mini output shared in chat.

Raising only the Mini's capacity to 48 preserved all 32 token IDs across both ranks and both trials. First-trial elapsed time rose from 112.67 s to 177.08 s on rank 0. On the second trial, elapsed time rose from 96.93 s to 132.37 s (36.6%). Decode fell from 0.514 to 0.462 tok/s and the Mini's peak MLX allocation rose from 8.34 GB to 11.05 GB. Mini expert misses fell from 4,241 to 2,752 and loader time fell from 46.02 s to 38.93 s. The Mini's forward time outside the measured loader rose from 25.38 s to 69.12 s across prefill and decode. The profile identifies where the regression occurred, and system memory pressure was not recorded. The rank 1 source is `logs/pipeline-baseline/122b-split32-16-rank-1-20261119T072211Z.log`. Rank 0 metrics came from the Mini output shared in chat.

#### Profile packed hot experts

Sync the benchmark runner and the updated `core.py`, `modules.py`, and `pipeline.py` under `mlx_moe/lazy_experts/` to both Macs. Keep the 32/16 layer split, prefill 3, 32 generated tokens, and two trials. Use capacity 32 with `--pipeline-hot-experts 8` on the Mini and capacity 64 with `--pipeline-hot-experts 32` on the M1 Pro.

The first pass populates the exact cache. The benchmark then packs the highest-priority cached experts into reusable quantized buffers and pins those entries within the existing per-layer capacity. The flag reserves room for the maximum number of distinct routes in one prefill chunk. Dispatches covered entirely by the packed buffer can use it directly. Mixed routes use the exact cache and load any missing expert weights from the checkpoint.

Compare the second trial with the 32/16 profile that used the same capacities, without packing. Check token IDs, decode rate, and memory on both ranks. The `packed` event reports setup time and peak allocation during packing. `experts.packed_dispatches` counts projection calls that used the packed buffers directly.

### Validate SSD streaming on both ranks with the 35B checkpoint

Use the same Ethernet hostfile and ranks as the resident run. Each Mac needs a full local copy of the checkpoint and the updated checkout. The loader opens the checkpoint lazily, removes the other rank's layers, and replaces its own expert weights before evaluating the remaining model parameters. At inference time it reads expert slices from the local SSD and sends activations over Ethernet.

```zsh
unset MLX_METAL_FAST_SYNCH
set -o pipefail
mkdir -p logs/pipeline-baseline
model="$HOME/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv-pipeline/bin/python -u benchmarks/benchmark_pipeline_baseline.py \
    pipeline-ssd "$model" --capacity 8 --prefill-step-size 4 --tokens 8 --repeat 1 2>&1 \
  | tee "logs/pipeline-baseline/ssd-rank-${MLX_RANK}-$(date -u +%Y%m%dT%H%M%SZ).log"
```

Expect rank 0 to report layers `[20, 40)` and rank 1 to report `[0, 20)`. Both trial records should show positive expert bytes loaded and the same token IDs. The resident 35B run on the default prompt produced `[8160, 725, 264, 12654, 709, 421, 78161, 48142]` for these first eight tokens.

Once the smoke test passes, use `--capacity 64 --tokens 128 --prompt-suite benchmarks/pipeline_coding_prompts.json` on both ranks. Compare elapsed time on rank 0 and peak MLX allocation and expert data loaded on each rank. Compare tokens with the resident run and record the first difference for any prompt that diverges.
