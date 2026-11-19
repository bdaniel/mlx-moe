# Copyright © 2023-2025 Apple Inc.

from pathlib import Path

import numpy as np
import mlx.core as mx

from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from .loading import (
    _find_switch_mlp,
    _find_moe_block,
    _detect_num_experts,
    _build_shard_map,
    _load_proj_experts,
    _load_experts,
    _with_cache_limit_zero,
    compute_adaptive_allocations,
    select_capacity,
    SafetensorsMap,
)
from .modules import (
    ExpertCache,
    LazyQuantizedSwitchLinear,
    CachedQuantizedSwitchLinear,
    PredictiveExpertCache,
    PredictiveCachedSwitchLinear,
    SyncPredictiveCachedSwitchLinear,
)


def enable_lazy_experts(
    model,
    model_path: Path,
    cache_capacity_per_layer: int = 0,
    predictive: bool = False,
) -> int:
    """Replace QuantizedSwitchLinear modules in MoE layers with lazy/cached versions.

    Args:
        model: The loaded MLX model (with lazy=True).
        model_path: Path to the model directory containing safetensors shards.
        cache_capacity_per_layer: Number of experts to cache per layer. 0 = no cache
            (Phase 1 lazy loading). > 0 = LCP-cached or predictive loading.
        predictive: If True and cache_capacity > 0, use zero-eval predictive cache
            (Phase 3). Pre-loads experts at startup, eliminates per-layer mx.eval.

    Returns:
        Number of modules replaced (expected: 48 layers x 3 = 144).
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)

    if predictive and cache_capacity_per_layer > 0:
        return _enable_predictive(model, shard_map, cache_capacity_per_layer)
    elif cache_capacity_per_layer > 0:
        return _enable_cached(model, shard_map, cache_capacity_per_layer)
    else:
        return _enable_lazy(model, shard_map)


_SWITCH_LINEAR_TYPES = (
    QuantizedSwitchLinear, LazyQuantizedSwitchLinear, CachedQuantizedSwitchLinear,
    PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear,
)


def _enable_lazy(model, shard_map: dict) -> int:
    replaced = 0
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i, shard_map=shard_map)
        if switch is None:
            continue
        num_experts = _detect_num_experts(switch)
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(switch, name)
            if not isinstance(orig, _SWITCH_LINEAR_TYPES):
                continue
            key_prefix = f"{key_base}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = LazyQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=orig.group_size,
                bits=orig.bits,
                mode=orig.mode,
                shard_map=shard_map,
                linear_bias=orig.get("bias"),
            )
            replacement.num_experts = num_experts
            setattr(switch, name, replacement)
            replaced += 1
    return replaced


def _enable_cached(model, shard_map: dict, capacity: int, *, layer_offset: int = 0) -> int:
    replaced = 0
    for i, layer in enumerate(model.layers, start=layer_offset):
        switch, key_base = _find_switch_mlp(layer, i, shard_map=shard_map)
        if switch is None:
            continue
        num_experts = _detect_num_experts(switch)
        layer_cache = ExpertCache(capacity)
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(switch, name)
            if not isinstance(orig, _SWITCH_LINEAR_TYPES):
                continue
            key_prefix = f"{key_base}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = CachedQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=orig.group_size,
                bits=orig.bits,
                mode=orig.mode,
                proj_name=name,
                cache=layer_cache,
                shard_map=shard_map,
                linear_bias=orig.get("bias"),
            )
            replacement.num_experts = num_experts
            setattr(switch, name, replacement)
            replaced += 1
    return replaced


def _enable_predictive(model, shard_map: dict, capacity: int) -> int:
    """Install Phase 2 modules for warmup. Call upgrade_to_predictive() after."""
    return _enable_cached(model, shard_map, capacity)


def reset_to_cached(model, model_path: Path, capacity: int) -> int:
    """Downgrade predictive modules back to Phase 2 cached for re-warmup.

    Replaces PredictiveCachedSwitchLinear with fresh CachedQuantizedSwitchLinear,
    freeing the pre-stacked expert tensors. Use this to re-warm on a new prompt
    without reloading the entire model.

    Args:
        model: The loaded MLX model with predictive modules installed.
        model_path: Path to the model directory containing safetensors shards.
        capacity: Number of experts to cache per layer.

    Returns:
        Number of modules reset.
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)

    reset = 0
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i, shard_map=shard_map)
        if switch is None:
            continue
        first = getattr(switch, "gate_proj")
        if not isinstance(first, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        num_experts = first._cache.num_experts
        layer_cache = ExpertCache(capacity)
        for name in ("gate_proj", "up_proj", "down_proj"):
            pred_mod = getattr(switch, name)
            key_prefix = f"{key_base}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = CachedQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=pred_mod.group_size,
                bits=pred_mod.bits,
                mode=pred_mod.mode,
                proj_name=name,
                cache=layer_cache,
                shard_map=shard_map,
                linear_bias=pred_mod.get("bias"),
            )
            replacement.num_experts = num_experts
            setattr(switch, name, replacement)
            reset += 1

    mx.clear_cache()
    return reset


def upgrade_to_predictive(
    model,
    model_path: Path,
    capacity,
    sync: bool = False,
    st_map: SafetensorsMap | None = None,
) -> int:
    """Harvest Phase 2 LCP caches into zero-eval predictive tensors.

    Batched shard loading: groups expert loads by safetensors shard file and
    loads each shard only once (9 loads instead of 144). Evals per-layer within
    each shard batch to control memory.

    Args:
        model: The loaded MLX model with Phase 2 cached modules.
        model_path: Path to the model directory containing safetensors shards.
        capacity: int for uniform capacity, or list[int] for per-MoE-layer capacities.
        sync: If True, use SyncPredictiveCachedSwitchLinear (adds mx.eval per layer).
        st_map: Optional SafetensorsMap for mmap-based loading of stacked-format experts.

    Returns:
        Number of modules upgraded.
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)
    per_layer_caps = isinstance(capacity, (list, tuple))
    if st_map is None:
        st_map = getattr(model, "_st_map", None)

    # Pass 1: harvest LCP caches, determine what needs disk loading
    layer_meta = {}
    moe_idx = 0

    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i, shard_map=shard_map)
        if switch is None:
            continue
        first_proj = getattr(switch, "gate_proj")
        if not isinstance(first_proj, CachedQuantizedSwitchLinear):
            continue

        lcp_cache = first_proj._cache
        num_experts = _detect_num_experts(switch)
        C = min(capacity[moe_idx] if per_layer_caps else capacity, num_experts)
        moe_idx += 1

        discovered = sorted(
            lcp_cache.entries.keys(),
            key=lambda eid: lcp_cache._priority(eid),
            reverse=True,
        )[:C]
        discovered_set = set(discovered)

        filler = []
        for eid in range(num_experts):
            if len(discovered) + len(filler) >= C:
                break
            if eid not in discovered_set:
                filler.append(eid)
        cached_ids = list(discovered) + filler

        pred_cache = PredictiveExpertCache(C, num_experts)
        harvested = {}
        to_load = {}
        has_bias = None
        phase2_mods = {}

        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mods[name] = getattr(switch, name)
            h_list = []
            load_list = []
            for slot, eid in enumerate(cached_ids):
                cached = lcp_cache.lookup(eid, name)
                if cached is not None:
                    w, s, b = cached
                    if has_bias is None:
                        has_bias = b is not None
                    h_list.append((slot, w, s, b))
                else:
                    load_list.append((slot, eid))
            harvested[name] = h_list
            to_load[name] = load_list

        layer_meta[i] = {
            "cached_ids": cached_ids,
            "pred_cache": pred_cache,
            "harvested": harvested,
            "to_load": to_load,
            "has_bias": has_bias if has_bias is not None else any(
                f"{key_base}.{name}.biases" in shard_map
                for name in ("gate_proj", "up_proj", "down_proj")
            ),
            "phase2_mods": phase2_mods,
            "disc_count": len(discovered),
            "filler_count": len(filler),
            "C": C,
            "lcp_cache": lcp_cache,
            "key_base": key_base,
        }

    # Pass 2: group disk loads by shard, load each shard once
    shard_groups: dict[str, list[tuple]] = {}
    for i, meta in layer_meta.items():
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not meta["to_load"][name]:
                continue
            key_prefix = f"{meta['key_base']}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            shard_groups.setdefault(shard_path, []).append(
                (i, name, key_prefix, meta["to_load"][name]))

    loaded: dict[int, dict[str, dict[int, tuple]]] = {}

    for shard_path, group in shard_groups.items():
        shard = None if st_map is not None else mx.load(shard_path)

        layers_in_batch = sorted(set(layer_i for layer_i, _, _, _ in group))
        for layer_i in layers_in_batch:
            layer_entries = [(n, kp, slots) for li, n, kp, slots in group if li == layer_i]
            to_eval = []
            for name, key_prefix, slot_eids in layer_entries:
                load_ids = mx.array([eid for _, eid in slot_eids])
                w_batch, s_batch, b_batch = _load_experts(key_prefix, load_ids,
                                                          shard=shard, shard_map=shard_map,
                                                          st_map=st_map)
                to_eval.extend([w_batch, s_batch])
                if b_batch is not None:
                    to_eval.append(b_batch)

                slot_map = {}
                for j, (slot, _) in enumerate(slot_eids):
                    slot_map[slot] = (w_batch[j], s_batch[j],
                                      b_batch[j] if b_batch is not None else None)
                loaded.setdefault(layer_i, {})[name] = slot_map

            mx.eval(*to_eval)

        del shard

    # Pass 3: assemble stacked tensors, build lookups, install modules
    upgraded = 0
    cls = SyncPredictiveCachedSwitchLinear if sync else PredictiveCachedSwitchLinear

    for i, meta in layer_meta.items():
        pred_cache = meta["pred_cache"]
        cached_ids = meta["cached_ids"]
        has_bias = meta["has_bias"]
        C = meta["C"]

        for name in ("gate_proj", "up_proj", "down_proj"):
            ws, ss, bs = [], [], []
            harvested_map = {slot: (w, s, b) for slot, w, s, b in meta["harvested"][name]}
            loaded_map = loaded.get(i, {}).get(name, {})

            for slot in range(C):
                if slot in harvested_map:
                    w, s, b = harvested_map[slot]
                else:
                    w, s, b = loaded_map[slot]
                ws.append(w)
                ss.append(s)
                if has_bias:
                    bs.append(b)

            pred_cache.weights[name] = mx.stack(ws)
            pred_cache.scales[name] = mx.stack(ss)
            pred_cache.biases[name] = mx.stack(bs) if has_bias else None

        key_base = meta["key_base"]
        for name in ("gate_proj", "up_proj", "down_proj"):
            key_prefix = f"{key_base}.{name}"
            pred_cache._shard_paths[name] = shard_map[f"{key_prefix}.weight"]
            pred_cache._key_prefixes[name] = key_prefix
        pred_cache._shard_map = shard_map
        pred_cache._st_map = st_map

        pred_cache.build_lookup(cached_ids)
        mx.eval(pred_cache.lookup)

        switch, _ = _find_switch_mlp(model.layers[i], i)
        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mod = meta["phase2_mods"][name]
            replacement = cls(
                group_size=phase2_mod.group_size,
                bits=phase2_mod.bits,
                mode=phase2_mod.mode,
                proj_name=name,
                cache=pred_cache,
                linear_bias=phase2_mod.get("bias"),
            )
            replacement.num_experts = pred_cache.num_experts
            setattr(switch, name, replacement)
            upgraded += 1

        meta["lcp_cache"].entries.clear()
        meta["lcp_cache"].frequency.clear()
        meta["lcp_cache"].last_active.clear()

        print(f"  Layer {i}: {meta['disc_count']} discovered + {meta['filler_count']} filler "
              f"= {C} experts ({mx.get_active_memory() / 1e9:.1f} GB)")

    return upgraded


def upgrade_to_predictive_with_pinning(
    model,
    model_path: Path,
    capacity: int,
    universal_profile: dict,
    pin_threshold: float = 0.5,
    sync: bool = False,
    st_map: SafetensorsMap | None = None,
) -> int:
    """Like upgrade_to_predictive but pins universal experts.

    Universal experts occupy the first N slots and are marked as non-evictable.
    Remaining slots are filled with LCP-ranked discovered experts (evictable).

    Args:
        model: The loaded MLX model with Phase 2 cached modules.
        model_path: Path to the model directory containing safetensors shards.
        capacity: Number of experts to cache per layer.
        universal_profile: Dict from load_universal_profile() or profile_experts.py.
        pin_threshold: Minimum activation fraction to consider an expert universal.
        sync: If True, use SyncPredictiveCachedSwitchLinear.
        st_map: Optional SafetensorsMap for mmap-based loading of stacked-format experts.

    Returns:
        Number of modules upgraded.
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)
    if st_map is None:
        st_map = getattr(model, "_st_map", None)
    num_prompts = universal_profile["num_prompts"]
    min_count = int(pin_threshold * num_prompts)

    universal_per_layer: dict[int, list[int]] = {}
    for layer_str, layer_data in universal_profile["layers"].items():
        layer_idx = int(layer_str)
        counts = layer_data.get("activation_counts", {})
        universal = sorted(
            int(eid) for eid, cnt in counts.items()
            if int(cnt) >= min_count
        )
        universal_per_layer[layer_idx] = universal

    # Pass 1: harvest LCP caches with pinned experts in front
    layer_meta = {}
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i, shard_map=shard_map)
        if switch is None:
            continue
        first_proj = getattr(switch, "gate_proj")
        if not isinstance(first_proj, CachedQuantizedSwitchLinear):
            continue

        lcp_cache = first_proj._cache
        num_experts = _detect_num_experts(switch)
        C = min(capacity, num_experts)

        pinned = universal_per_layer.get(i, [])[:C]
        pinned_set_local = set(pinned)
        n_pinned = len(pinned)

        discovered = sorted(
            (eid for eid in lcp_cache.entries.keys() if eid not in pinned_set_local),
            key=lambda eid: lcp_cache._priority(eid),
            reverse=True,
        )[:C - n_pinned]
        discovered_set = set(discovered) | pinned_set_local

        filler = []
        for eid in range(num_experts):
            if len(pinned) + len(discovered) + len(filler) >= C:
                break
            if eid not in discovered_set:
                filler.append(eid)

        cached_ids = list(pinned) + list(discovered) + filler

        pred_cache = PredictiveExpertCache(C, num_experts)
        harvested = {}
        to_load = {}
        has_bias = None
        phase2_mods = {}

        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mods[name] = getattr(switch, name)
            h_list = []
            load_list = []
            for slot, eid in enumerate(cached_ids):
                cached = lcp_cache.lookup(eid, name)
                if cached is not None:
                    w, s, b = cached
                    if has_bias is None:
                        has_bias = b is not None
                    h_list.append((slot, w, s, b))
                else:
                    load_list.append((slot, eid))
            harvested[name] = h_list
            to_load[name] = load_list

        layer_meta[i] = {
            "cached_ids": cached_ids,
            "pred_cache": pred_cache,
            "harvested": harvested,
            "to_load": to_load,
            "has_bias": has_bias if has_bias is not None else any(
                f"{key_base}.{name}.biases" in shard_map
                for name in ("gate_proj", "up_proj", "down_proj")
            ),
            "phase2_mods": phase2_mods,
            "lcp_cache": lcp_cache,
            "n_pinned": n_pinned,
            "pinned_set": pinned_set_local,
            "C": C,
            "key_base": key_base,
        }

    # Pass 2: group disk loads by shard, load each shard once
    shard_groups: dict[str, list[tuple]] = {}
    for i, meta in layer_meta.items():
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not meta["to_load"][name]:
                continue
            key_prefix = f"{meta['key_base']}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            shard_groups.setdefault(shard_path, []).append(
                (i, name, key_prefix, meta["to_load"][name]))

    loaded: dict[int, dict[str, dict[int, tuple]]] = {}

    for shard_path, group in shard_groups.items():
        shard = None if st_map is not None else mx.load(shard_path)
        layers_in_batch = sorted(set(layer_i for layer_i, _, _, _ in group))
        for layer_i in layers_in_batch:
            layer_entries = [(n, kp, slots) for li, n, kp, slots in group if li == layer_i]
            to_eval = []
            for name, key_prefix, slot_eids in layer_entries:
                load_ids = mx.array([eid for _, eid in slot_eids])
                w_batch, s_batch, b_batch = _load_experts(key_prefix, load_ids,
                                                          shard=shard, shard_map=shard_map,
                                                          st_map=st_map)
                to_eval.extend([w_batch, s_batch])
                if b_batch is not None:
                    to_eval.append(b_batch)

                slot_map = {}
                for j, (slot, _) in enumerate(slot_eids):
                    slot_map[slot] = (w_batch[j], s_batch[j],
                                      b_batch[j] if b_batch is not None else None)
                loaded.setdefault(layer_i, {})[name] = slot_map

            mx.eval(*to_eval)
        del shard

    # Pass 3: assemble stacked tensors, build lookups, install modules
    upgraded = 0
    cls = SyncPredictiveCachedSwitchLinear if sync else PredictiveCachedSwitchLinear

    for i, meta in layer_meta.items():
        pred_cache = meta["pred_cache"]
        cached_ids = meta["cached_ids"]
        has_bias = meta["has_bias"]
        C = meta["C"]

        for name in ("gate_proj", "up_proj", "down_proj"):
            ws, ss, bs = [], [], []
            harvested_map = {slot: (w, s, b) for slot, w, s, b in meta["harvested"][name]}
            loaded_map = loaded.get(i, {}).get(name, {})

            for slot in range(C):
                if slot in harvested_map:
                    w, s, b = harvested_map[slot]
                else:
                    w, s, b = loaded_map[slot]
                ws.append(w)
                ss.append(s)
                if has_bias:
                    bs.append(b)

            pred_cache.weights[name] = mx.stack(ws)
            pred_cache.scales[name] = mx.stack(ss)
            pred_cache.biases[name] = mx.stack(bs) if has_bias else None

        key_base = meta["key_base"]
        for name in ("gate_proj", "up_proj", "down_proj"):
            key_prefix = f"{key_base}.{name}"
            pred_cache._shard_paths[name] = shard_map[f"{key_prefix}.weight"]
            pred_cache._key_prefixes[name] = key_prefix
        pred_cache._shard_map = shard_map
        pred_cache._st_map = st_map

        pred_cache.build_lookup(cached_ids)
        pred_cache.pinned_set = meta["pinned_set"]
        mx.eval(pred_cache.lookup)

        switch, _ = _find_switch_mlp(model.layers[i], i)
        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mod = meta["phase2_mods"][name]
            replacement = cls(
                group_size=phase2_mod.group_size,
                bits=phase2_mod.bits,
                mode=phase2_mod.mode,
                proj_name=name,
                cache=pred_cache,
                linear_bias=phase2_mod.get("bias"),
            )
            replacement.num_experts = pred_cache.num_experts
            setattr(switch, name, replacement)
            upgraded += 1

        meta["lcp_cache"].entries.clear()
        meta["lcp_cache"].frequency.clear()
        meta["lcp_cache"].last_active.clear()

        print(f"  Layer {i}: {meta['n_pinned']} pinned + "
              f"{C - meta['n_pinned']} dynamic = {C} experts "
              f"({mx.get_active_memory() / 1e9:.1f} GB)")

    return upgraded


def dynamic_cache_update(model, max_layer_updates: int = 12) -> list[dict]:
    """Process buffered router indices and swap cold experts for missed ones.

    Call between tokens during generation. Handles the async_eval
    double-buffering by skipping in-flight indices.

    Args:
        model: The loaded MLX model with predictive modules.
        max_layer_updates: Max layers to perform swaps on per call. Limits
            transient memory from shard loading. Other layers still track
            stats but defer swaps to later calls.

    Returns:
        Per-layer stats: [{"layer": i, "swaps": n, "fallbacks": n, "requests": n}, ...]
    """
    stats = []
    swap_budget = max_layer_updates
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache

        if swap_budget > 0:
            layer_stats = cache.update()
            if layer_stats["swaps"] > 0:
                swap_budget -= 1
        else:
            if len(cache._indices_buffer) < 2:
                layer_stats = {"swaps": 0, "fallbacks": 0, "requests": 0}
            else:
                to_process = cache._indices_buffer[:-1]
                cache._indices_buffer = cache._indices_buffer[-1:]
                all_requested: set[int] = set()
                for indices in to_process:
                    flat = np.asarray(indices.reshape(-1))
                    all_requested |= set(int(x) for x in np.unique(flat))
                cache.step += 1
                for eid in all_requested:
                    cache.frequency[eid] = cache.frequency.get(eid, 0) + 1
                    cache.last_active[eid] = cache.step
                misses = all_requested - cache.cached_set
                cache.total_requests += len(all_requested)
                cache.total_fallbacks += len(misses)
                layer_stats = {"swaps": 0, "fallbacks": len(misses), "requests": len(all_requested)}

        layer_stats["layer"] = i
        stats.append(layer_stats)
    return stats


def get_faithful_stats(model) -> dict:
    """Aggregate exact-loading cache telemetry across MoE layers."""
    caches = {}
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue
        caches[id(proj._cache)] = proj._cache

    hits = sum(cache.hits for cache in caches.values())
    misses = sum(cache.misses for cache in caches.values())
    requests = hits + misses
    return {
        "layers": len(caches),
        "requests": requests,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / requests if requests else 0.0,
        "load_calls": sum(cache.load_calls for cache in caches.values()),
        "packed_dispatches": sum(cache.packed_dispatches for cache in caches.values()),
        "experts_loaded": sum(cache.experts_loaded for cache in caches.values()),
        "bytes_loaded": sum(cache.bytes_loaded for cache in caches.values()),
        "load_seconds": sum(cache.load_seconds for cache in caches.values()),
    }


def adaptive_faithful_capacity(
    model,
    *,
    grow_by: int = 8,
    target_fraction: float = 0.60,
) -> dict:
    """Grow exact caches while projected full capacity stays below memory target."""
    caches = []
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue
        cache = proj._cache
        complete = next(
            (entry for entry in cache.entries.values() if len(entry) == 3),
            None,
        )
        if complete is None:
            continue
        slot_bytes = sum(
            tensor.nbytes
            for weights in complete.values()
            for tensor in weights
            if tensor is not None
        )
        caches.append((cache, int(proj.num_experts), slot_bytes))

    if not caches:
        return {"grown": 0, "layers": 0, "headroom_gb": None}

    target_bytes = int(
        mx.device_info()["max_recommended_working_set_size"] * target_fraction
    )
    active_bytes = mx.get_active_memory()
    allocator_cache_bytes = mx.get_cache_memory()
    reserved_bytes = sum(
        max(0, cache.capacity - len(cache.entries)) * slot_bytes
        for cache, _, slot_bytes in caches
    )
    projected_bytes = active_bytes + allocator_cache_bytes + reserved_bytes
    level_bytes = sum(slot_bytes for _, _, slot_bytes in caches)
    available_levels = max(0, (target_bytes - projected_bytes) // level_bytes)
    max_capacity_growth = min(
        num_experts - cache.capacity for cache, num_experts, _ in caches
    )
    increment = min(grow_by, available_levels, max_capacity_growth)
    headroom_gb = (target_bytes - projected_bytes) / 1e9
    if increment <= 0:
        return {"grown": 0, "layers": 0, "headroom_gb": headroom_gb}

    for cache, _, _ in caches:
        cache.capacity += increment
    return {
        "grown": increment * len(caches),
        "layers": len(caches),
        "headroom_gb": headroom_gb,
        "capacity": caches[0][0].capacity,
    }


def adaptive_cache_growth(
    model,
    fallback_rate: float,
    *,
    min_fallback_rate: float = 0.25,
    min_headroom_gb: float = 1.0,
    grow_by: int = 8,
    max_layer_growths: int = 16,
    target_fraction: float = 0.75,
) -> dict:
    """Append cache slots for recently missed experts when memory allows."""
    if fallback_rate < min_fallback_rate:
        return {"grown": 0, "layers": 0, "headroom_gb": None}

    info = mx.device_info()
    target_bytes = int(
        info["max_recommended_working_set_size"] * target_fraction
    )
    projected_bytes = mx.get_active_memory() + mx.get_cache_memory()
    headroom_gb = (target_bytes - projected_bytes) / 1e9
    if headroom_gb < min_headroom_gb:
        return {"grown": 0, "layers": 0, "headroom_gb": headroom_gb}

    candidates = []
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(
            proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)
        ):
            continue
        cache = proj._cache
        misses = [
            eid
            for eid in cache.last_misses
            if eid not in cache.cached_set and len(cache.cached_ids) < cache.num_experts
        ]
        if not misses:
            continue
        misses.sort(
            key=lambda eid: (cache.frequency.get(eid, 0), cache.last_active.get(eid, 0)),
            reverse=True,
        )
        candidates.append((len(misses), i, cache, misses[:grow_by]))

    grown = 0
    layers = 0
    for _, _, cache, expert_ids in sorted(candidates, reverse=True)[:max_layer_growths]:
        cached = max(1, len(cache.cached_ids))
        permanent_bytes = 0
        transient_bytes = 0
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            tensors = (
                cache.weights.get(proj_name),
                cache.scales.get(proj_name),
                cache.biases.get(proj_name),
            )
            current_bytes = sum(
                tensor.nbytes for tensor in tensors if tensor is not None
            )
            added_bytes = current_bytes * len(expert_ids) // cached
            permanent_bytes += added_bytes
            transient_bytes = max(
                transient_bytes,
                current_bytes + added_bytes,
            )
        projected_peak = (
            mx.get_active_memory()
            + mx.get_cache_memory()
            + permanent_bytes
            + transient_bytes
        )
        if projected_peak > target_bytes:
            continue
        added = cache.grow(expert_ids)
        if added:
            grown += added
            layers += 1

    return {"grown": grown, "layers": layers, "headroom_gb": headroom_gb}


def dynamic_cache_update_ml(
    model,
    eviction_models: dict,
    max_layer_updates: int = 12,
) -> list[dict]:
    """Like dynamic_cache_update but uses ML eviction scoring.

    Instead of LCP priority, uses a tiny FFN per layer to predict eviction
    scores (approximating Belady distance).

    Args:
        model: The loaded MLX model with predictive modules.
        eviction_models: Dict[int, nn.Module] mapping layer_idx to a trained
            FFN: input=[1/recency, freq/max_freq], output=eviction score.
            Higher score = evict first (longer predicted distance to next use).
        max_layer_updates: Max layers to perform swaps on per call.

    Returns:
        Per-layer stats list (same format as dynamic_cache_update).
    """
    stats = []
    swap_budget = max_layer_updates

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache

        if len(cache._indices_buffer) < 2:
            stats.append({"layer": i, "swaps": 0, "fallbacks": 0, "requests": 0})
            continue

        to_process = cache._indices_buffer[:-1]
        cache._indices_buffer = cache._indices_buffer[-1:]

        all_requested: set[int] = set()
        for indices in to_process:
            flat = np.asarray(indices.reshape(-1))
            all_requested |= set(int(x) for x in np.unique(flat))

        cache.step += 1
        for eid in all_requested:
            cache.frequency[eid] = cache.frequency.get(eid, 0) + 1
            cache.last_active[eid] = cache.step

        misses = all_requested - cache.cached_set
        n_requests = len(all_requested)
        n_fallbacks = len(misses)
        cache.total_requests += n_requests
        cache.total_fallbacks += n_fallbacks

        if not misses or not cache._shard_paths or swap_budget <= 0:
            stats.append({"layer": i, "swaps": 0, "fallbacks": n_fallbacks,
                         "requests": n_requests})
            continue

        ml_model = eviction_models.get(i)
        max_freq = max(cache.frequency.values()) if cache.frequency else 1

        evict_candidates = []
        for slot, eid in enumerate(cache.cached_ids):
            if eid in all_requested or eid in cache.pinned_set:
                continue
            if ml_model is not None:
                recency = cache.step - cache.last_active.get(eid, 0)
                freq = cache.frequency.get(eid, 0)
                features = mx.array([[1.0 / max(recency, 1), freq / max(max_freq, 1)]])
                score = float(ml_model(features).item())
            else:
                score = -cache._lcp_priority(eid)
            evict_candidates.append((score, slot, eid))

        evict_candidates.sort(reverse=True)

        swaps: list[tuple[int, int, int]] = []
        for new_eid in sorted(misses):
            if not evict_candidates:
                break
            _, slot, old_eid = evict_candidates.pop(0)
            swaps.append((slot, old_eid, new_eid))

        MAX_SWAPS = 10
        swaps = swaps[:MAX_SWAPS]

        if not swaps:
            stats.append({"layer": i, "swaps": 0, "fallbacks": n_fallbacks,
                         "requests": n_requests})
            continue

        new_eids = mx.array([new_eid for _, _, new_eid in swaps])
        slot_indices = mx.array([slot for slot, _, _ in swaps])
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            shard_path = cache._shard_paths[proj_name]
            key_prefix = cache._key_prefixes[proj_name]
            shard = mx.load(shard_path) if cache._st_map is None else None
            new_w, new_s, new_b = _load_experts(key_prefix, new_eids,
                                                 shard=shard, shard_map=cache._shard_map,
                                                 st_map=cache._st_map)
            del shard

            if new_b is None:
                mx.eval(new_w, new_s)
            else:
                mx.eval(new_w, new_s, new_b)

            w = cache.weights.pop(proj_name)
            w[slot_indices] = new_w
            cache.weights[proj_name] = w

            s = cache.scales.pop(proj_name)
            s[slot_indices] = new_s
            cache.scales[proj_name] = s

            if cache.biases[proj_name] is not None and new_b is not None:
                b = cache.biases.pop(proj_name)
                b[slot_indices] = new_b
                cache.biases[proj_name] = b

        mx.clear_cache()

        for slot, old_eid, new_eid in swaps:
            cache.cached_set.discard(old_eid)
            cache.cached_set.add(new_eid)
            cache.cached_ids[slot] = new_eid
            cache.frequency.pop(old_eid, None)
            cache.last_active.pop(old_eid, None)

        cache.rebuild_lookup()
        mx.eval(cache.lookup, cache.hit_mask)

        swap_budget -= 1
        stats.append({"layer": i, "swaps": len(swaps), "fallbacks": n_fallbacks,
                     "requests": n_requests})

    return stats


def enable_skip_fallback(model) -> int:
    """Monkey-patch MoE blocks to zero out scores for missing experts.

    Without this, missing experts fall back to cache slot 0 (wrong expert),
    injecting wrong-expert outputs weighted by the real router score. This
    corrupts the hidden state and compounds across layers.

    With skip-fallback, missing experts get score=0 and the remaining hits
    are renormalized. The MoE layer degrades to using only the cached
    expert(s), which is a partial but directionally correct signal. The
    residual connection (output = input + MoE(input)) passes through the
    unchanged input for the missing expert's contribution.

    Returns:
        Number of MoE blocks patched.
    """
    patched = 0
    for i, layer in enumerate(model.layers):
        moe_block = _find_moe_block(layer)
        if moe_block is None:
            continue
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue

        cache = proj._cache
        switch_attr = "switch_mlp" if hasattr(moe_block, "switch_mlp") else "experts"
        gate_attr = "gate" if hasattr(moe_block, "gate") else "router"
        _patch_moe_block_skip_fallback(moe_block, cache, switch_attr, gate_attr)
        patched += 1

    return patched


def _patch_moe_block_skip_fallback(
    moe_block, cache: PredictiveExpertCache, switch_attr="switch_mlp", gate_attr="gate"
):
    """Replace a MoE block's __call__ with one that masks missing expert scores.

    Handles two gate styles:
      - Standard (Qwen, Mixtral): gate returns raw logits
      - GLM-style: gate returns (inds, scores) tuple directly
    """
    def patched_call(self, x):
        gate_out = getattr(self, gate_attr)(x)
        if isinstance(gate_out, tuple):
            inds, scores = gate_out
        else:
            gates = gate_out
            k = getattr(self, "num_experts_per_tok", getattr(self, "top_k", 2))
            gates = mx.softmax(gates, axis=-1, precise=True)
            inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if getattr(self, "norm_topk_prob", False):
                scores = scores / scores.sum(axis=-1, keepdims=True)

        mask = cache.hit_mask[inds]
        scores = scores * mask
        score_sum = scores.sum(axis=-1, keepdims=True)
        scores = mx.where(score_sum > 0, scores / score_sum, scores)

        y = getattr(self, switch_attr)(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
            se = self.shared_expert(x)
            y = y + mx.sigmoid(self.shared_expert_gate(x)) * se
        elif hasattr(self, "shared_experts"):
            y = y + self.shared_experts(x)

        return y

    import types
    moe_block.__call__ = types.MethodType(patched_call, moe_block)


def get_fallback_stats(model) -> dict:
    """Collect cumulative fallback stats from all PredictiveExpertCache instances.

    Args:
        model: The loaded MLX model with predictive modules.

    Returns:
        Dict with total_requests, total_fallbacks, fallback_rate, and per-layer stats.
    """
    total_requests = 0
    total_fallbacks = 0
    layer_stats = []

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache
        total_requests += cache.total_requests
        total_fallbacks += cache.total_fallbacks
        if cache.total_requests > 0:
            layer_stats.append({
                "layer": i,
                "requests": cache.total_requests,
                "fallbacks": cache.total_fallbacks,
                "fallback_rate": cache.total_fallbacks / cache.total_requests,
                "cached_experts": len(cache.cached_set),
            })

    return {
        "total_requests": total_requests,
        "total_fallbacks": total_fallbacks,
        "fallback_rate": total_fallbacks / total_requests if total_requests > 0 else 0.0,
        "layers": layer_stats,
    }


def get_cache_stats(model) -> dict:
    """Collect hit/miss stats from all ExpertCache instances in the model.

    Args:
        model: The loaded MLX model with Phase 2 cached modules.

    Returns:
        Dict with total_hits, total_misses, total_hit_rate, and per-layer stats.
    """
    total_hits = 0
    total_misses = 0
    layer_stats = []

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue
        cache = proj._cache
        hits = cache.hits
        misses = cache.misses
        total = hits + misses
        rate = hits / total if total > 0 else 0.0
        layer_stats.append({
            "layer": i,
            "hits": hits,
            "misses": misses,
            "hit_rate": rate,
            "cached_experts": len(cache.entries),
        })
        total_hits += hits
        total_misses += misses

    total = total_hits + total_misses
    return {
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_hit_rate": total_hits / total if total > 0 else 0.0,
        "layers": layer_stats,
    }


def measure_fallback(model) -> dict:
    """Compute fallback stats by draining buffered indices post-generation.

    Unlike get_fallback_stats() (which reads counters updated by cache.update()),
    this directly inspects which requested expert IDs are not in the cached set.
    Call after mlx_lm.generate() completes.

    Args:
        model: The loaded MLX model with predictive modules.

    Returns:
        Dict with total_requests, total_fallbacks, fallback_rate, and per-layer stats.
    """
    total_requests = 0
    total_fallbacks = 0
    layer_stats = []

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        cache = proj._cache
        all_requested = set()
        for indices in cache._indices_buffer:
            flat = np.asarray(indices.reshape(-1))
            all_requested |= set(int(x) for x in np.unique(flat))
        cache._indices_buffer.clear()

        missing = all_requested - cache.cached_set
        n_req = len(all_requested)
        n_fb = len(missing)
        total_requests += n_req
        total_fallbacks += n_fb

        if n_req > 0:
            layer_stats.append({
                "layer": i,
                "requested": n_req,
                "missing": n_fb,
                "fallback_rate": n_fb / n_req,
            })

    return {
        "total_requests": total_requests,
        "total_fallbacks": total_fallbacks,
        "fallback_rate": total_fallbacks / total_requests if total_requests > 0 else 0.0,
        "layers": layer_stats,
    }


def adaptive_capacity_upgrade(
    model,
    model_path: Path,
    total_budget_experts: int,
    min_per_layer: int = 32,
    sync: bool = False,
) -> dict:
    """Compute per-layer capacities from LCP warmup and upgrade to predictive.

    After LCP warmup, inspects each layer's discovered expert count and allocates
    capacity proportionally (with 30% headroom), subject to total budget and min floor.
    Layers that discover more experts get more capacity.

    Args:
        model: The loaded MLX model with Phase 2 cached modules post-warmup.
        model_path: Path to the model directory containing safetensors shards.
        total_budget_experts: Total expert slots to allocate across all layers.
        min_per_layer: Minimum slots per layer.
        sync: If True, use SyncPredictiveCachedSwitchLinear.

    Returns:
        Dict with upgraded count, allocations, discovered counts, capacities,
        total_experts, and estimated_memory_gb.
    """
    layer_counts = []
    moe_layers = []

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj")
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue
        layer_counts.append(len(proj._cache.all_seen))
        moe_layers.append(i)

    n = len(layer_counts)

    raw = [max(min_per_layer, int(count * 1.3)) for count in layer_counts]
    total_raw = sum(raw)

    scale = total_budget_experts / total_raw if total_raw > 0 else 1.0
    capacities = [max(min_per_layer, min(512, round(r * scale))) for r in raw]

    diff = total_budget_experts - sum(capacities)
    sorted_idx = sorted(range(n), key=lambda j: capacities[j], reverse=True)
    for j in sorted_idx:
        if diff == 0:
            break
        if diff > 0:
            if capacities[j] < 512:
                capacities[j] += 1
                diff -= 1
        elif capacities[j] > min_per_layer:
            capacities[j] -= 1
            diff += 1

    upgraded = upgrade_to_predictive(model, model_path, capacities, sync=sync)

    total_experts = sum(capacities)
    estimated_gb = sum(c * 1.769 for c in capacities) / 1000

    return {
        "upgraded": upgraded,
        "allocations": dict(zip(moe_layers, capacities)),
        "discovered": dict(zip(moe_layers, layer_counts)),
        "capacities": capacities,
        "total_experts": total_experts,
        "estimated_memory_gb": estimated_gb,
    }
