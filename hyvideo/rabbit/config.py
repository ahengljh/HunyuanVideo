from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class RabbitRuntimeConfig:
    """Configuration flags for Rabbit-style runtime optimisations."""

    enabled: bool = False
    offload_mode: str = "none"  # none | weights
    offload_plan: Dict[str, List[int]] = field(
        default_factory=lambda: {"double": [], "single": []}
    )
    offload_device: str = "cpu"
    prefetch_distance: int = 1
    memory_budget_mb: Optional[float] = None
    min_device_blocks: int = 0
    plan_metadata: Dict[str, float] = field(default_factory=dict)
    resident_plan: Dict[str, List[int]] = field(
        default_factory=lambda: {"double": [], "single": []}
    )
    aggressive_offload: bool = True
    hot_resident_limit: int = 0
    hot_resident_threshold: int = 24
    trace_residency_path: Optional[str] = None
    profile_steps: int = 0
    profile_low_ratio: float = 0.5

    skip_strategy: str = "none"  # none | ema | schedule
    skip_threshold: float = 1e-3
    skip_progress_power: float = 2.0
    skip_warmup_steps: int = 4
    skip_min_progress: float = 0.2
    skip_cooldown: int = 1
    skip_max_streak: int = 4
    skip_ema_decay: float = 0.9
    skip_stage: str = "both"

    log_stats: bool = False
    diagnostics_interval: int = 5

    # Frame caching for computation reuse (disabled by default for safety)
    enable_frame_cache: bool = False
    cache_similarity_threshold: float = 0.98  # Higher threshold for safety
    memory_safety_margin: float = 0.15  # 15% safety margin for small devices

    # Gradient checkpointing for activation memory reduction
    enable_checkpointing: bool = True  # Recompute activations to save memory
    checkpoint_every_n_blocks: int = 1  # Checkpoint every block for maximum savings

    def stage_enabled(self, stage: str) -> bool:
        if self.skip_stage == "both":
            return True
        return stage == self.skip_stage

    @property
    def weights_offload_enabled(self) -> bool:
        return self.enabled and self.offload_mode != "none"

    @property
    def skip_enabled(self) -> bool:
        return self.enabled and self.skip_strategy != "none"


def build_runtime_config(args, transformer) -> RabbitRuntimeConfig:
    """Create a :class:`RabbitRuntimeConfig` from CLI args and model metadata."""

    cfg = RabbitRuntimeConfig()

    offload_flag = getattr(args, "rabbit_offload", None)
    if offload_flag is None:
        cfg.offload_mode = getattr(args, "rabbit_offload_mode", "none")
    else:
        cfg.offload_mode = "weights" if offload_flag else "none"

    cfg.enabled = cfg.offload_mode != "none"
    if not cfg.enabled:
        return cfg

    cfg.offload_device = getattr(args, "rabbit_offload_device", cfg.offload_device)
    cfg.prefetch_distance = max(0, int(getattr(args, "rabbit_prefetch_distance", cfg.prefetch_distance)))
    memory_budget = getattr(args, "rabbit_memory_budget_mb", None)
    cfg.memory_budget_mb = (
        float(memory_budget) if memory_budget is not None else None
    )
    if cfg.memory_budget_mb is not None and cfg.memory_budget_mb <= 0:
        cfg.memory_budget_mb = None

    # Auto-detect GPU memory and set sensible budget for small devices
    if cfg.memory_budget_mb is None and torch.cuda.is_available():
        try:
            # Get available GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Reserve budget for activations, leaving ~40-50% for model weights
            # For small GPUs (<=16GB), be more conservative
            if gpu_memory_gb <= 12:
                # For 8-12GB GPUs, allocate ~4-6GB for transformer weights
                cfg.memory_budget_mb = gpu_memory_gb * 1024 * 0.4
            elif gpu_memory_gb <= 16:
                # For 12-16GB GPUs, allocate ~6-8GB for transformer weights
                cfg.memory_budget_mb = gpu_memory_gb * 1024 * 0.45
            elif gpu_memory_gb <= 24:
                # For 16-24GB GPUs, allocate ~10-12GB for transformer weights
                cfg.memory_budget_mb = gpu_memory_gb * 1024 * 0.5
            # For larger GPUs (>24GB), don't set a budget - let it use what it needs
        except Exception:
            # If we can't detect GPU memory, don't set a budget
            pass
    min_device_blocks = getattr(args, "rabbit_min_device_blocks", None)
    if min_device_blocks is not None:
        cfg.min_device_blocks = max(0, int(min_device_blocks))
    else:
        # Smart default: keep only 2-3 blocks resident for small devices
        # This ensures maximum memory savings while maintaining reasonable performance
        cfg.min_device_blocks = 2

    cfg.aggressive_offload = bool(getattr(args, "rabbit_aggressive_offload", cfg.aggressive_offload))
    if cfg.aggressive_offload and cfg.prefetch_distance > 0:
        cfg.prefetch_distance = 0
    cfg.profile_steps = max(0, int(getattr(args, "rabbit_profile_steps", cfg.profile_steps)))

    cfg.skip_strategy = "none"

    cfg.log_stats = bool(getattr(args, "rabbit_log_stats", False))
    cfg.hot_resident_limit = max(0, int(getattr(args, "rabbit_hot_resident_limit", cfg.hot_resident_limit)))
    cfg.hot_resident_threshold = max(1, int(getattr(args, "rabbit_hot_resident_threshold", cfg.hot_resident_threshold)))
    trace_path = getattr(args, "rabbit_residency_trace", None)
    cfg.trace_residency_path = str(trace_path) if trace_path else None

    # Configure frame caching and memory management
    # Enable caching by default when rabbit offload is active - saves memory by reusing similar computations
    cfg.enable_frame_cache = bool(getattr(args, "rabbit_enable_cache", True))  # Enabled by default for memory savings
    cfg.cache_similarity_threshold = float(getattr(args, "rabbit_cache_threshold", 0.98))
    cfg.memory_safety_margin = float(getattr(args, "rabbit_memory_safety", 0.15))

    resident_str = getattr(args, "rabbit_resident_plan", None)
    cfg.resident_plan = parse_resident_plan(resident_str, transformer)
    if cfg.resident_plan:
        max_resident = max(len(indices) for indices in cfg.resident_plan.values())
        cfg.min_device_blocks = max(cfg.min_device_blocks, max_resident)

    plan_str = "auto"
    default_ratio = 1.0 if cfg.aggressive_offload else 0.5
    cfg.offload_plan = parse_offload_plan(
        plan_str,
        transformer,
        cfg.offload_mode,
        default_ratio,
    )

    if cfg.weights_offload_enabled and cfg.memory_budget_mb is not None:
        # Ensure pinned residents never get scheduled for offload
        cfg.offload_plan, cfg.plan_metadata = plan_for_memory_budget(
            transformer=transformer,
            budget_mb=cfg.memory_budget_mb,
            min_blocks=cfg.min_device_blocks,
            base_plan=cfg.offload_plan,
            pinned_blocks=cfg.resident_plan,
        )

    # Respect resident hints by removing them from offload plan explicitly
    for stage, pinned in cfg.resident_plan.items():
        if not pinned:
            continue
        pinned_set = set(pinned)
        cfg.offload_plan[stage] = [idx for idx in cfg.offload_plan.get(stage, []) if idx not in pinned_set]

    ensure_min_resident_blocks(cfg, transformer)

    return cfg


def parse_resident_plan(
    plan_str: Optional[str],
    transformer,
) -> Dict[str, List[int]]:
    plan = {"double": [], "single": []}
    if not plan_str:
        return plan
    spec = plan_str.strip().lower()
    if not spec:
        return plan
    segments = [segment.strip() for segment in spec.split(";") if segment.strip()]
    stage_to_tokens = {
        "double": len(getattr(transformer, "double_blocks", [])),
        "single": len(getattr(transformer, "single_blocks", [])),
    }
    for segment in segments:
        if ":" not in segment:
            raise ValueError(
                f"Invalid resident segment '{segment}'. Expected '<stage>:<indices>'."
            )
        stage_name, spec_body = segment.split(":", 1)
        stage_name = stage_name.strip().lower()
        if stage_name not in stage_to_tokens:
            raise ValueError(
                f"Unknown resident stage '{stage_name}'. Expected 'double' or 'single'."
            )
        plan[stage_name] = parse_stage_spec(spec_body.strip(), stage_to_tokens[stage_name])
    return plan


def parse_offload_plan(
    plan_str: Optional[str],
    transformer,
    offload_mode: str,
    default_ratio: Optional[float],
) -> Dict[str, List[int]]:
    """Parse a human readable offload description into stage -> indices."""

    plan = {"double": [], "single": []}
    if offload_mode == "none":
        return plan

    num_double = len(getattr(transformer, "double_blocks", []))
    num_single = len(getattr(transformer, "single_blocks", []))

    spec = (plan_str or "").strip().lower()
    if spec in {"", "auto"}:
        ratio = coerce_ratio(default_ratio, 0.5)
        plan["double"] = pick_tail_indices(num_double, ratio)
        plan["single"] = pick_tail_indices(num_single, ratio)
        return plan

    segments = [segment.strip() for segment in spec.split(";") if segment.strip()]
    if not segments:
        ratio = coerce_ratio(default_ratio, 0.5)
        plan["double"] = pick_tail_indices(num_double, ratio)
        plan["single"] = pick_tail_indices(num_single, ratio)
        return plan

    stage_to_tokens = {"double": num_double, "single": num_single}
    filled = set()

    for segment in segments:
        if ":" not in segment:
            raise ValueError(
                f"Invalid offload segment '{segment}'. Expected '<stage>:<spec>'."
            )
        stage_name, spec_body = segment.split(":", 1)
        stage_name = stage_name.strip().lower()

        if stage_name in {"auto", "all"}:
            ratio_override = coerce_ratio(
                float_or_none(spec_body.strip()) if spec_body.strip() else default_ratio,
                0.5,
            )
            for stage_key, count in stage_to_tokens.items():
                plan[stage_key] = pick_tail_indices(count, ratio_override)
                filled.add(stage_key)
            continue

        if stage_name not in stage_to_tokens:
            raise ValueError(
                f"Unknown offload stage '{stage_name}'. Expected 'double' or 'single'."
            )
        count = stage_to_tokens[stage_name]
        plan[stage_name] = parse_stage_spec(spec_body.strip(), count)
        filled.add(stage_name)

    ratio = coerce_ratio(default_ratio, 0.5)
    for stage_name, count in stage_to_tokens.items():
        if plan[stage_name]:
            continue
        plan[stage_name] = pick_tail_indices(count, ratio)

    return plan


def parse_stage_spec(spec: str, max_len: int) -> List[int]:
    if spec in {"", "none"}:
        return []
    if spec.startswith("auto"):
        ratio = 0.5
        if ":" in spec:
            _, ratio_str = spec.split(":", 1)
            ratio = coerce_ratio(float_or_none(ratio_str), 0.5)
        return pick_tail_indices(max_len, ratio)

    indices: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else max_len - 1
            if end < start:
                start, end = end, start
            indices.extend(range(start, min(end, max_len - 1) + 1))
        else:
            idx = int(token)
            indices.append(idx)

    valid = sorted({idx for idx in indices if 0 <= idx < max_len})
    return valid


def pick_tail_indices(count: int, ratio: float) -> List[int]:
    if count == 0 or ratio <= 0:
        return []
    if ratio >= 1:
        return list(range(count))
    tail = max(1, int(math.ceil(count * ratio)))
    start = max(0, count - tail)
    return list(range(start, count))


def coerce_ratio(value: Optional[float], default: float) -> float:
    ratio = default if value is None else float(value)
    if ratio < 0:
        ratio = 0.0
    if ratio > 1:
        ratio = 1.0
    return ratio


def float_or_none(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


BYTES_IN_MB = 1024 * 1024


def plan_for_memory_budget(
    transformer,
    budget_mb: float,
    min_blocks: int,
    base_plan: Dict[str, List[int]],
    pinned_blocks: Optional[Dict[str, List[int]]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, float]]:
    """Derive a block offload plan to keep on-device weights under the requested budget."""

    stage_blocks = {
        "double": list(getattr(transformer, "double_blocks", [])),
        "single": list(getattr(transformer, "single_blocks", [])),
    }
    stage_sizes = {
        stage: [module_num_bytes(block) for block in blocks]
        for stage, blocks in stage_blocks.items()
    }
    persistent_attrs = [
        "img_in",
        "txt_in",
        "time_in",
        "vector_in",
        "guidance_in",
        "final_layer",
    ]
    persistent_bytes = sum(
        module_num_bytes(getattr(transformer, attr, None)) for attr in persistent_attrs
    )

    total_block_bytes = sum(sum(sizes) for sizes in stage_sizes.values())
    budget_bytes = budget_mb * BYTES_IN_MB
    pinned_sets = {
        stage: set((pinned_blocks or {}).get(stage, []))
        for stage in stage_blocks
    }
    base_sets = {
        stage: {idx for idx in base_plan.get(stage, []) if 0 <= idx < len(stage_blocks[stage])}
        for stage in stage_blocks
    }
    base_offloaded_bytes = sum(
        stage_sizes[stage][idx] for stage in stage_sizes for idx in base_sets[stage]
    )
    device_block_bytes = total_block_bytes - base_offloaded_bytes
    target_block_budget = max(0, budget_bytes - persistent_bytes)
    bytes_to_remove = max(device_block_bytes - target_block_budget, 0)
    added_offloads = {stage: set() for stage in stage_blocks}
    stage_min_blocks = {
        stage: min(min_blocks, len(stage_sizes[stage])) for stage in stage_blocks
    }

    if bytes_to_remove > 0:
        # Create a global priority list that evicts the largest blocks first while respecting per-stage minima.
        stage_priority = {"single": 1, "double": 0}
        candidates: List[Tuple[int, str, int]] = []
        for stage, sizes in stage_sizes.items():
            for idx, size in enumerate(sizes):
                if idx in base_sets[stage]:
                    continue
                candidates.append((size, stage, idx))

        candidates.sort(
            key=lambda item: (item[0], stage_priority.get(item[1], 0), -item[2]),
            reverse=True,
        )

        for size, stage, idx in candidates:
            if bytes_to_remove <= 0:
                break
            if idx in added_offloads[stage]:
                continue
            resident_after = len(stage_sizes[stage]) - (
                len(base_sets[stage]) + len(added_offloads[stage]) + 1
            )
            if idx in pinned_sets[stage]:
                continue
            if resident_after < max(stage_min_blocks[stage], len(pinned_sets[stage])):
                continue
            added_offloads[stage].add(idx)
            bytes_to_remove -= size
            if bytes_to_remove <= 0:
                break

    final_plan = {}
    for stage in stage_blocks:
        combined = base_sets[stage].union(added_offloads[stage])
        combined -= pinned_sets[stage]
        final_plan[stage] = sorted(combined)
    total_offloaded_bytes = sum(
        stage_sizes[stage][idx] for stage in stage_sizes for idx in final_plan[stage]
    )
    remaining_block_bytes = max(total_block_bytes - total_offloaded_bytes, 0)
    estimated_device_mb = (persistent_bytes + remaining_block_bytes) / BYTES_IN_MB
    shortfall_mb = max(0.0, estimated_device_mb - budget_mb)

    metadata = {
        "target_mb": float(budget_mb),
        "persistent_mb": persistent_bytes / BYTES_IN_MB,
        "blocks_mb": total_block_bytes / BYTES_IN_MB,
        "estimated_device_mb": estimated_device_mb,
        "offloaded_blocks_double": float(len(final_plan["double"])),
        "offloaded_blocks_single": float(len(final_plan["single"])),
        "shortfall_mb": shortfall_mb,
        "resident_blocks_double": float(len(stage_blocks["double"]) - len(final_plan["double"])),
        "resident_blocks_single": float(len(stage_blocks["single"]) - len(final_plan["single"])),
        "min_device_blocks": float(min_blocks),
        "pinned_blocks_double": float(len(pinned_sets["double"])),
        "pinned_blocks_single": float(len(pinned_sets["single"])),
    }
    return final_plan, metadata


def ensure_min_resident_blocks(cfg: RabbitRuntimeConfig, transformer) -> None:
    """Ensure each stage keeps at least ``cfg.min_device_blocks`` residents."""

    stage_blocks = {
        "double": list(getattr(transformer, "double_blocks", [])),
        "single": list(getattr(transformer, "single_blocks", [])),
    }
    for stage, blocks in stage_blocks.items():
        total_blocks = len(blocks)
        if total_blocks == 0:
            continue
        pinned = set(cfg.resident_plan.get(stage, []))
        required_residents = max(cfg.min_device_blocks, len(pinned))
        max_offload = max(0, total_blocks - required_residents)
        current_plan = set(cfg.offload_plan.get(stage, []))
        # Remove any pinned blocks that might have slipped in.
        current_plan -= pinned
        if len(current_plan) <= max_offload:
            cfg.offload_plan[stage] = sorted(current_plan)
            continue
        # Keep the smallest modules resident to minimize device footprint.
        sizes = [module_num_bytes(block) for block in blocks]
        # Sort candidate indices by (size asc, index asc)
        candidates = sorted(
            ((sizes[idx], idx) for idx in current_plan),
            key=lambda item: (item[0], item[1]),
        )
        while len(current_plan) > max_offload and candidates:
            _, idx = candidates.pop(0)
            if idx in current_plan:
                current_plan.remove(idx)
        cfg.offload_plan[stage] = sorted(current_plan)


def module_num_bytes(module: Optional[torch.nn.Module]) -> int:
    if module is None:
        return 0
    total = 0
    for param in module.parameters(recurse=True):
        total += param.numel() * param.element_size()
    for buf in module.buffers(recurse=True):
        total += buf.numel() * buf.element_size()
    return total
