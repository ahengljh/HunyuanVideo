from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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


def build_runtime_config(args, transformer) -> RabbitRuntimeConfig:
    """Create a :class:`RabbitRuntimeConfig` from CLI args and model metadata."""

    cfg = RabbitRuntimeConfig()
    cfg.enabled = bool(getattr(args, "rabbit_enable", False))
    if not cfg.enabled:
        return cfg

    cfg.offload_mode = getattr(args, "rabbit_offload_mode", "none")
    cfg.offload_device = getattr(args, "rabbit_offload_device", "cpu")
    cfg.prefetch_distance = max(0, int(getattr(args, "rabbit_prefetch_distance", 1)))

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

