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
    min_device_blocks: int = 1
    plan_metadata: Dict[str, float] = field(default_factory=dict)
    cache_outputs: bool = False
    cache_device: str = "cpu"
    cache_threshold: float = 0.02
    cache_max_age: int = 6
    cache_min_importance: float = 5e-4
    cache_stage: str = "both"
    cache_token_ratio: float = 1.0
    profile_steps: int = 0
    profile_low_ratio: float = 0.5
    profile_cache: bool = True
    profile_offload: bool = True

    latent_offload: bool = False
    latent_offload_device: str = "cpu"
    latent_pin_memory: bool = True

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

    @property
    def cache_enabled(self) -> bool:
        return self.enabled and self.cache_outputs


def build_runtime_config(args, transformer) -> RabbitRuntimeConfig:
    """Create a :class:`RabbitRuntimeConfig` from CLI args and model metadata."""

    cfg = RabbitRuntimeConfig()
    cfg.enabled = bool(getattr(args, "rabbit_enable", False))
    if not cfg.enabled:
        return cfg

    cfg.offload_mode = getattr(args, "rabbit_offload_mode", "none")
    cfg.offload_device = getattr(args, "rabbit_offload_device", "cpu")
    cfg.prefetch_distance = max(0, int(getattr(args, "rabbit_prefetch_distance", 1)))
    memory_budget = getattr(args, "rabbit_memory_budget_mb", None)
    cfg.memory_budget_mb = (
        float(memory_budget) if memory_budget is not None else None
    )
    if cfg.memory_budget_mb is not None and cfg.memory_budget_mb <= 0:
        cfg.memory_budget_mb = None
    cfg.min_device_blocks = max(0, int(getattr(args, "rabbit_min_device_blocks", 2)))
    cfg.cache_outputs = bool(getattr(args, "rabbit_cache_outputs", False))
    cfg.cache_device = getattr(args, "rabbit_cache_device", "cpu")
    cfg.cache_threshold = float(getattr(args, "rabbit_cache_threshold", 0.02))
    cfg.cache_max_age = max(1, int(getattr(args, "rabbit_cache_max_age", 6)))
    cfg.cache_min_importance = float(getattr(args, "rabbit_cache_min_importance", 5e-4))
    cfg.cache_stage = getattr(args, "rabbit_cache_stage", "both")
    cfg.cache_token_ratio = coerce_ratio(
        getattr(args, "rabbit_cache_token_ratio", 1.0), 1.0
    )
    cfg.profile_steps = max(0, int(getattr(args, "rabbit_profile_steps", 0)))
    cfg.profile_low_ratio = coerce_ratio(
        getattr(args, "rabbit_profile_low_ratio", 0.5), 0.5
    )
    cfg.profile_cache = bool(getattr(args, "rabbit_profile_cache", True))
    cfg.profile_offload = bool(getattr(args, "rabbit_profile_offload", True))

    cfg.latent_offload = bool(getattr(args, "rabbit_latent_offload", False))
    cfg.latent_offload_device = getattr(
        args, "rabbit_latent_offload_device", "cpu"
    )
    cfg.latent_pin_memory = bool(
        getattr(args, "rabbit_latent_pin_memory", True)
    )

    cfg.skip_strategy = getattr(args, "rabbit_skip_strategy", "none")
    cfg.skip_threshold = float(getattr(args, "rabbit_skip_threshold", 1e-3))
    cfg.skip_progress_power = float(
        getattr(args, "rabbit_skip_progress_power", 2.0)
    )
    cfg.skip_warmup_steps = max(0, int(getattr(args, "rabbit_skip_warmup", 4)))
    cfg.skip_min_progress = float(getattr(args, "rabbit_skip_min_progress", 0.2))
    cfg.skip_cooldown = max(0, int(getattr(args, "rabbit_skip_cooldown", 1)))
    cfg.skip_max_streak = max(1, int(getattr(args, "rabbit_skip_max_streak", 4)))
    cfg.skip_ema_decay = float(getattr(args, "rabbit_skip_ema_decay", 0.9))
    cfg.skip_stage = getattr(args, "rabbit_skip_stage", "both")

    cfg.log_stats = bool(getattr(args, "rabbit_log_stats", False))
    cfg.diagnostics_interval = max(
        1, int(getattr(args, "rabbit_diagnostics_interval", 5))
    )

    plan_str = getattr(args, "rabbit_offload_plan", "auto")
    default_ratio = getattr(args, "rabbit_offload_ratio", None)
    cfg.offload_plan = parse_offload_plan(
        plan_str,
        transformer,
        cfg.offload_mode,
        default_ratio,
    )

    if cfg.weights_offload_enabled and cfg.memory_budget_mb is not None:
        cfg.offload_plan, cfg.plan_metadata = plan_for_memory_budget(
            transformer=transformer,
            budget_mb=cfg.memory_budget_mb,
            min_blocks=cfg.min_device_blocks,
            base_plan=cfg.offload_plan,
        )

    return cfg


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

    if bytes_to_remove > 0:
        # Prefer evicting the tail of the single stream (executed last), then the double stream.
        for stage in ("single", "double"):
            sizes = stage_sizes[stage]
            if not sizes:
                continue
            stage_min_blocks = min(min_blocks, len(sizes))
            for idx in range(len(sizes) - 1, -1, -1):
                if bytes_to_remove <= 0:
                    break
                if idx in base_sets[stage] or idx in added_offloads[stage]:
                    continue
                remaining_if_removed = len(sizes) - (len(base_sets[stage]) + len(added_offloads[stage]) + 1)
                if remaining_if_removed < stage_min_blocks:
                    continue
                added_offloads[stage].add(idx)
                bytes_to_remove -= sizes[idx]
            if bytes_to_remove <= 0:
                break

    final_plan = {
        stage: sorted(base_sets[stage].union(added_offloads[stage]))
        for stage in stage_blocks
    }
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
    }
    return final_plan, metadata


def module_num_bytes(module: Optional[torch.nn.Module]) -> int:
    if module is None:
        return 0
    total = 0
    for param in module.parameters(recurse=True):
        total += param.numel() * param.element_size()
    for buf in module.buffers(recurse=True):
        total += buf.numel() * buf.element_size()
    return total
