"""Faithful SSD expert loading for a local Qwen 3.5 pipeline stage."""

import json
from pathlib import Path

import mlx.core as mx

from .core import _enable_cached
from .generate import _WARMUP_CACHE, _load_model_and_tokenizer
from .loading import SafetensorsMap, _build_shard_map, _find_switch_mlp


def load_faithful_pipeline_stage(model_path: Path, group, capacity: int):
    config = json.loads((model_path / "config.json").read_text())
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError("SSD pipeline requires a quantized Qwen 3.5 MoE checkpoint")

    shard_map = _build_shard_map(model_path)
    for path in sorted(set(shard_map.values())):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing checkpoint shard: {path}")

    model, tokenizer, _ = _load_model_and_tokenizer(str(model_path), lazy=True)
    stage = model.model
    # Discard the other rank's layers and replace this rank's experts before eval.
    stage.pipeline(group)
    replaced = _enable_cached(model, shard_map, capacity, layer_offset=stage.start_idx)
    if replaced != len(model.layers) * 3:
        raise ValueError("SSD pipeline requires three quantized expert projections per layer")

    projections = []
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        projections.extend(getattr(switch, name) for name in ("gate_proj", "up_proj", "down_proj"))
    keys = {
        f"{projection._key_prefix}.{suffix}"
        for projection in projections
        for suffix in ("weight", "scales", "biases")
    }
    expert_map = SafetensorsMap(sorted({shard_map[key] for key in keys if key in shard_map}))
    required = {key for key in keys if not key.endswith(".biases") or key in shard_map}
    missing = sorted(key for key in required if key not in expert_map)
    if missing:
        expert_map.close()
        raise ValueError(f"Missing stacked expert tensor: {missing[0]}")
    for projection in projections:
        projection._st_map = expert_map

    mx.set_cache_limit(_WARMUP_CACHE)
    mx.clear_cache()
    mx.eval(model.parameters())
    mx.eval(mx.distributed.all_sum(mx.array(1), group=group, stream=mx.cpu))
    return model, tokenizer
