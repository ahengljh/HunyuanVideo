from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Tuple, Union

import torch

from .config import RabbitRuntimeConfig

logger = logging.getLogger(__name__)

StageName = str


@dataclass
class BlockExecutionContext:
    step_index: int
    total_steps: int
    timestep: float
    max_timestep: float

    @property
    def step_progress(self) -> float:
        if self.total_steps <= 0:
            return 1.0
        return (self.step_index + 1) / self.total_steps

    @property
    def noise_progress(self) -> float:
        if self.max_timestep <= 0:
            return 1.0
        return 1.0 - (self.timestep / self.max_timestep)


@dataclass
class BlockState:
    ema_importance: float = 0.0
    executed: int = 0
    skipped: int = 0
    skip_cooldown: int = 0
    skip_streak: int = 0
    last_delta: float = 0.0

    def register_execution(self, delta: float, decay: float):
        self.executed += 1
        if self.executed == 1:
            self.ema_importance = delta
        else:
            self.ema_importance = self.ema_importance * decay + delta * (1.0 - decay)
        self.last_delta = delta
        if self.skip_cooldown > 0:
            self.skip_cooldown -= 1
        self.skip_streak = 0

    def register_skip(self, cooldown: int):
        self.skipped += 1
        self.skip_streak += 1
        if cooldown > 0:
            self.skip_cooldown = cooldown

    @property
    def total_observations(self) -> int:
        return self.executed + self.skipped


@dataclass
class BlockDecision:
    skip: bool
    outputs: Optional[Tuple[torch.Tensor, ...]] = None


class RabbitRuntimeManager:
    """Runtime controller implementing RabbitVideo-inspired optimisations."""

    def __init__(
        self,
        transformer,
        config: RabbitRuntimeConfig,
        device: Union[str, torch.device],
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.transformer = transformer
        self.config = config
        self.device = torch.device(device)
        self.logger = logger_instance or logger

        self.offload_device = torch.device(config.offload_device)

        self.double_blocks = list(getattr(transformer, "double_blocks", []))
        self.single_blocks = list(getattr(transformer, "single_blocks", []))

        self._blocks: Dict[StageName, List[torch.nn.Module]] = {
            "double": self.double_blocks,
            "single": self.single_blocks,
        }
        self._states: Dict[StageName, List[BlockState]] = {
            stage: [BlockState() for _ in modules]
            for stage, modules in self._blocks.items()
        }

        self._offload_sets: Dict[StageName, set] = {
            stage: set(config.offload_plan.get(stage, []))
            for stage in self._blocks.keys()
        }
        self._residency: Dict[StageName, Dict[int, str]] = {
            stage: {idx: "device" for idx in range(len(modules))}
            for stage, modules in self._blocks.items()
        }

        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        if (
            self.config.weights_offload_enabled
            and self.config.prefetch_distance > 0
            and self.device.type == "cuda"
        ):
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
        self._prefetched: Dict[Tuple[StageName, int], bool] = {}

        self._current_context: Optional[BlockExecutionContext] = None
        self._max_timestep: Optional[float] = None

        self.stats: Dict[str, Dict[StageName, List[float]]] = {
            "executed": {
                stage: [0 for _ in modules] for stage, modules in self._blocks.items()
            },
            "skipped": {
                stage: [0 for _ in modules] for stage, modules in self._blocks.items()
            },
            "importance": {
                stage: [0.0 for _ in modules] for stage, modules in self._blocks.items()
            },
        }
        self._hot_residents: Dict[StageName, set] = {
            stage: set() for stage in self._blocks.keys()
        }
        self._hot_order: Dict[StageName, List[int]] = {
            stage: [] for stage in self._blocks.keys()
        }
        self.trace_enabled = bool(self.config.trace_residency_path)
        self.trace_path: Optional[Path] = (
            Path(self.config.trace_residency_path)
            if self.config.trace_residency_path
            else None
        )
        self._trace_records: List[Dict[str, Union[str, int, float]]] = []
        if self.trace_enabled:
            self._trace_event(
                {
                    "event": "init",
                    "double_blocks": len(self.double_blocks),
                    "single_blocks": len(self.single_blocks),
                }
            )
        self.profiling_steps = max(0, self.config.profile_steps)
        self.profiling_active = self.profiling_steps > 0
        self._profile_finalized = False

        self._prepare_module_residency()
        self.transformer.enable_rabbit_runtime(self)
        self.active = config.enabled

        if config.plan_metadata:
            meta = config.plan_metadata
            self.logger.info(
                "[Rabbit] Memory-aware plan | target=%.1f MB | estimated=%.1f MB | "
                "persistent=%.1f MB | blocks=%.1f MB | shortfall=%.1f MB",
                meta.get("target_mb", 0.0),
                meta.get("estimated_device_mb", 0.0),
                meta.get("persistent_mb", 0.0),
                meta.get("blocks_mb", 0.0),
                meta.get("shortfall_mb", 0.0),
            )
            if "resident_blocks_double" in meta or "resident_blocks_single" in meta:
                self.logger.info(
                    "[Rabbit] Resident blocks | double=%d | single=%d | min-per-stage=%d",
                    int(meta.get("resident_blocks_double", 0)),
                    int(meta.get("resident_blocks_single", 0)),
                    self.config.min_device_blocks,
                )
                if "pinned_blocks_double" in meta or "pinned_blocks_single" in meta:
                    self.logger.info(
                        "[Rabbit] Pinned blocks | double=%d | single=%d",
                        int(meta.get("pinned_blocks_double", 0)),
                        int(meta.get("pinned_blocks_single", 0)),
                    )

        # Log initialization
        if config.enabled and config.log_stats:
            self.logger.info("=" * 60)
            self.logger.info("[Rabbit] Runtime Initialized")
            self.logger.info("=" * 60)
            if config.skip_enabled:
                self.logger.info(f"[Rabbit] Skip Strategy: {config.skip_strategy}")
                self.logger.info(f"[Rabbit] Skip Threshold: {config.skip_threshold}")
                self.logger.info(f"[Rabbit] Skip Stage: {config.skip_stage}")
            if config.weights_offload_enabled:
                self.logger.info(f"[Rabbit] Weight Offloading: {config.offload_mode}")
                self.logger.info(f"[Rabbit] Offload Device: {config.offload_device}")
                if config.memory_budget_mb is not None:
                    self.logger.info(
                        f"[Rabbit] Offload Memory Budget: {config.memory_budget_mb:.1f} MB "
                        f"(min-device-blocks={config.min_device_blocks})"
                    )
                if config.aggressive_offload:
                    self.logger.info("[Rabbit] Aggressive streaming: ENABLED (prefetch=0)")
                if config.hot_resident_limit > 0:
                    self.logger.info(
                        f"[Rabbit] Hot resident cache: keeping up to {config.hot_resident_limit} blocks "
                        f"after {config.hot_resident_threshold} executions"
                    )
                self.logger.info(
                    f"[Rabbit] Prefetch Distance: {config.prefetch_distance}"
                )
                for stage, indices in config.offload_plan.items():
                    if indices:
                        self.logger.info(f"[Rabbit] Offloading {stage} blocks: {len(indices)} blocks")
            self.logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Public API used by the pipeline
    # ------------------------------------------------------------------
    def register_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def latents_to_device(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def after_step(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def update_step(self, step_index: int, timestep: Union[float, torch.Tensor], total_steps: int):
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                timestep_value = 0.0
            else:
                timestep_value = float(timestep.detach().float().max().item())
        else:
            timestep_value = float(timestep)
        if self._max_timestep is None:
            self._max_timestep = max(timestep_value, 1.0)
        else:
            self._max_timestep = max(self._max_timestep, timestep_value)

        self._current_context = BlockExecutionContext(
            step_index=step_index,
            total_steps=total_steps,
            timestep=timestep_value,
            max_timestep=self._max_timestep,
        )

        if self.config.log_stats and step_index % 5 == 0:
            progress = max(self._current_context.step_progress, self._current_context.noise_progress)
            self.logger.info(f"[Rabbit] Step {step_index+1}/{total_steps} - "
                           f"Progress: {progress*100:.1f}% - Timestep: {timestep_value:.1f}")

        if (
            self.profiling_active
            and self.profiling_steps > 0
            and step_index + 1 >= self.profiling_steps
        ):
            self._finalize_profile()

        self.transformer.set_runtime_context(self._current_context)

    def before_block(
        self,
        stage: StageName,
        index: int,
        block: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
    ) -> BlockDecision:
        if self.profiling_active:
            return BlockDecision(skip=False)
        if self.config.skip_enabled and self.config.stage_enabled(stage):
            if self._should_skip(stage, index):
                outputs = tuple(t for t in inputs)
                state = self._states[stage][index]
                state.register_skip(self.config.skip_cooldown)
                self.stats["skipped"][stage][index] += 1
                if self.config.log_stats:
                    self.logger.info(f"[Rabbit] Block SKIPPED: {stage}[{index}] - "
                                   f"Total skips: {self.stats['skipped'][stage][index]}")
                return BlockDecision(skip=True, outputs=outputs)
        if self.config.weights_offload_enabled:
            self._ensure_on_device(stage, index, block)
            self._schedule_prefetch(stage, index)

        return BlockDecision(skip=False)

    def after_block(
        self,
        stage: StageName,
        index: int,
        block: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        outputs: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        state = self._states[stage][index]
        delta = self._compute_importance(stage, inputs, outputs)
        state.register_execution(delta, self.config.skip_ema_decay)
        self.stats["executed"][stage][index] += 1
        self.stats["importance"][stage][index] = state.ema_importance

        if self.config.log_stats and state.total_observations % 10 == 0:
            self.logger.info(f"[Rabbit] Block EXECUTED: {stage}[{index}] - "
                           f"delta={delta:.4e}, importance={state.ema_importance:.4e}")

        should_retain = self._maybe_promote_hot_block(stage, index, block)

        if self.config.weights_offload_enabled and not should_retain:
            self._offload_block(stage, index, block)

        return outputs

    def finalize(self) -> Optional[Dict[str, Dict[str, float]]]:
        self.transformer.set_runtime_context(None)
        if self.trace_enabled:
            self._trace_event({"event": "finalize"})
            if self.trace_path is not None:
                try:
                    self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.trace_path.open("w", encoding="utf-8") as f:
                        for record in self._trace_records:
                            f.write(json.dumps(record) + "\n")
                    if self.config.log_stats:
                        self.logger.info(
                            f"[Rabbit] Residency trace written to {self.trace_path}"
                        )
                except OSError as exc:
                    self.logger.warning(
                        f"[Rabbit] Failed to write residency trace to {self.trace_path}: {exc}"
                    )
        if not self.config.log_stats:
            return None

        summary = {
            "executed": {
                stage: sum(values) for stage, values in self.stats["executed"].items()
            },
            "skipped": {
                stage: sum(values) for stage, values in self.stats["skipped"].items()
            },
            "mean_importance": {
                stage: (mean(values) if values else 0.0)
                for stage, values in self.stats["importance"].items()
            },
        }

        # Enhanced summary logging
        self.logger.info("=" * 60)
        self.logger.info("[Rabbit] PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)

        total_blocks_exec = 0
        total_blocks_skip = 0

        for stage in ("double", "single"):
            if stage not in summary["executed"]:
                continue
            exec_count = summary["executed"][stage]
            skip_count = summary["skipped"][stage]
            total_blocks = exec_count + skip_count
            skip_rate = (skip_count / total_blocks * 100) if total_blocks > 0 else 0
            mean_importance = summary["mean_importance"].get(stage, 0.0)

            self.logger.info(f"[Rabbit] {stage.upper()} blocks:")
            self.logger.info(f"  - Executed: {exec_count}/{total_blocks} blocks")
            self.logger.info(f"  - Skipped:  {skip_count}/{total_blocks} blocks ({skip_rate:.1f}%)")
            self.logger.info(f"  - Mean importance: {mean_importance:.4e}")

            total_blocks_exec += exec_count
            total_blocks_skip += skip_count

        total_all = total_blocks_exec + total_blocks_skip
        if total_all > 0:
            overall_skip_rate = total_blocks_skip / total_all * 100
            self.logger.info("-" * 40)
            self.logger.info(f"[Rabbit] OVERALL: {overall_skip_rate:.1f}% blocks skipped")
            self.logger.info(f"[Rabbit] Computation saved: ~{overall_skip_rate:.1f}%")

        if self.config.weights_offload_enabled:
            self.logger.info(f"[Rabbit] Weight offloading: ENABLED (device: {self.config.offload_device})")

        self.logger.info("=" * 60)
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _trace_event(self, payload: Dict[str, Union[str, int, float]]) -> None:
        if not self.trace_enabled:
            return
        event = dict(payload)
        if "step" not in event:
            event["step"] = (
                self._current_context.step_index
                if self._current_context is not None
                else -1
            )
        event["order"] = len(self._trace_records)
        self._trace_records.append(event)

    def _trace_residency(
        self, stage: StageName, index: int, status: str, action: str
    ) -> None:
        self._trace_event(
            {
                "event": "residency",
                "stage": stage,
                "block": index,
                "status": status,
                "action": action,
            }
        )

    def _prepare_module_residency(self):
        if self.profiling_active or not self.config.weights_offload_enabled:
            self.transformer.to(self.device)
            for stage, modules in self._blocks.items():
                for idx, module in enumerate(modules):
                    module.to(self.device)
                    self._residency[stage][idx] = "device"
                    self._trace_residency(stage, idx, "device", "init")
            return

        self._move_persistent_modules_to_device()

        for stage, modules in self._blocks.items():
            target_set = self._offload_sets[stage]
            for idx, module in enumerate(modules):
                if idx in target_set:
                    module.to(self.offload_device)
                    self._residency[stage][idx] = "offload"
                    self._trace_residency(stage, idx, "offload", "init_plan")
                else:
                    module.to(self.device)
                    self._residency[stage][idx] = "device"
                    self._trace_residency(stage, idx, "device", "init_plan")

    def _move_persistent_modules_to_device(self):
        persistent_attrs = [
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
        ]
        for attr in persistent_attrs:
            module = getattr(self.transformer, attr, None)
            if module is None:
                continue
            module.to(self.device)

    def _ensure_on_device(
        self, stage: StageName, index: int, block: torch.nn.Module
    ):
        if self.profiling_active or not self.config.weights_offload_enabled:
            return
        residency = self._residency[stage][index]
        if residency == "device":
            return

        if residency == "prefetch":
            if self._prefetch_stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(self._prefetch_stream)
            self._residency[stage][index] = "device"
            self._prefetched.pop((stage, index), None)
            self._trace_residency(stage, index, "device", "prefetch_commit")
            return

        block.to(self.device, non_blocking=self.device.type == "cuda")
        self._residency[stage][index] = "device"
        self._trace_residency(stage, index, "device", "load")

    def _schedule_prefetch(self, stage: StageName, index: int):
        if (
            self._prefetch_stream is None
            or self.profiling_active
            or not self.config.weights_offload_enabled
        ):
            return
        distance = self.config.prefetch_distance
        if distance <= 0:
            return
        blocks = self._blocks[stage]
        targets = self._offload_sets[stage]
        for offset in range(1, distance + 1):
            next_index = index + offset
            if next_index >= len(blocks):
                break
            if next_index not in targets:
                continue
            key = (stage, next_index)
            if self._residency[stage][next_index] == "device" or key in self._prefetched:
                continue
            next_block = blocks[next_index]
            with torch.cuda.stream(self._prefetch_stream):
                next_block.to(self.device, non_blocking=True)
            self._prefetched[key] = True
            self._residency[stage][next_index] = "prefetch"
            self._trace_residency(stage, next_index, "prefetch", "prefetch_schedule")

    def _offload_block(
        self, stage: StageName, index: int, block: torch.nn.Module
    ):
        if self.profiling_active:
            return
        if index not in self._offload_sets[stage]:
            return
        residency = self._residency[stage][index]
        if residency == "offload":
            return
        block.to(self.offload_device, non_blocking=self.device.type == "cuda")
        self._residency[stage][index] = "offload"
        if index in self._hot_residents[stage]:
            self._hot_residents[stage].remove(index)
            if index in self._hot_order[stage]:
                self._hot_order[stage].remove(index)
        self._trace_residency(stage, index, "offload", "evict")

    def _maybe_promote_hot_block(
        self, stage: StageName, index: int, block: torch.nn.Module
    ) -> bool:
        if index in self._hot_residents[stage]:
            # Touch order to maintain LRU semantics.
            if index in self._hot_order[stage]:
                self._hot_order[stage].remove(index)
            self._hot_order[stage].append(index)
            return True

        if (
            not self.config.aggressive_offload
            or self.config.hot_resident_limit <= 0
            or not self.config.weights_offload_enabled
        ):
            return False

        state = self._states[stage][index]
        if state.executed < self.config.hot_resident_threshold:
            return False

        self._offload_sets[stage].discard(index)
        self._hot_residents[stage].add(index)
        self._hot_order[stage].append(index)
        if self.device.type == "cuda" and self._residency[stage][index] != "device":
            block.to(self.device, non_blocking=True)
            self._residency[stage][index] = "device"
            self._trace_residency(stage, index, "device", "hot_promote")

        while len(self._hot_residents[stage]) > self.config.hot_resident_limit:
            evict_index = self._hot_order[stage].pop(0)
            if evict_index not in self._hot_residents[stage]:
                continue
            self._demote_hot_block(stage, evict_index)
        return True

    def _demote_hot_block(self, stage: StageName, index: int):
        if index not in self._hot_residents[stage]:
            return
        self._hot_residents[stage].remove(index)
        if index in self._hot_order[stage]:
            self._hot_order[stage].remove(index)
        self._offload_sets[stage].add(index)
        if not self.config.weights_offload_enabled:
            return
        block = self._blocks[stage][index]
        block.to(self.offload_device, non_blocking=self.device.type == "cuda")
        self._residency[stage][index] = "offload"
        self._trace_residency(stage, index, "offload", "hot_demote")

    def _finalize_profile(self):
        if self._profile_finalized or self.profiling_steps <= 0:
            return
        self._profile_finalized = True
        self.profiling_active = False

        low_ratio = self.config.profile_low_ratio
        profile_sets: Dict[StageName, set] = {stage: set() for stage in self._blocks.keys()}

        if low_ratio > 0:
            for stage, states in self._states.items():
                if not states:
                    continue
                ranked = sorted(
                    enumerate(states),
                    key=lambda item: item[1].ema_importance,
                )
                count = max(1, int(len(states) * low_ratio))
                selected = {idx for idx, _ in ranked[:count]}
                profile_sets[stage] = selected

        if self.config.log_stats:
            for stage, indices in profile_sets.items():
                if indices:
                    self.logger.info(
                        "[Rabbit] Profiling selected low-importance %s blocks: %s",
                        stage,
                        sorted(indices),
                    )

    def _apply_offload_plan(self, plan: Dict[StageName, set]):
        for stage, modules in self._blocks.items():
            targets = set(plan.get(stage, set()))
            for idx, module in enumerate(modules):
                current = self._residency[stage][idx]
                desired = "offload" if idx in targets else "device"
                if desired == current:
                    continue
                if desired == "offload":
                    module.to(self.offload_device, non_blocking=self.device.type == "cuda")
                    self._residency[stage][idx] = "offload"
                else:
                    module.to(self.device, non_blocking=self.device.type == "cuda")
                    self._residency[stage][idx] = "device"
        self._offload_sets = {stage: set(plan.get(stage, set())) for stage in self._blocks.keys()}
        self._prefetched.clear()

    def _compute_importance(
        self,
        stage: StageName,
        inputs: Tuple[torch.Tensor, ...],
        outputs: Tuple[torch.Tensor, ...],
    ) -> float:
        with torch.no_grad():
            if stage == "double":
                img_in, txt_in = inputs
                img_out, txt_out = outputs
                delta_img = (img_out - img_in).abs().mean().detach().item()
                delta_txt = (txt_out - txt_in).abs().mean().detach().item()
                return max(delta_img, delta_txt)
            else:
                (x_in,) = inputs
                (x_out,) = outputs
                return (x_out - x_in).abs().mean().detach().item()

    def _should_skip(self, stage: StageName, index: int) -> bool:
        state = self._states[stage][index]

        # Log decision process when verbose
        if self.config.log_stats and (state.total_observations % 10 == 0 or state.executed == 0):
            self.logger.info(f"[Rabbit] Skip check {stage}[{index}]: executed={state.executed}, "
                           f"importance={state.ema_importance:.4e}, cooldown={state.skip_cooldown}, "
                           f"warmup={self.config.skip_warmup_steps}")

        # Need at least warmup executions before considering skipping
        # Ensure we always execute at least once to build initial importance
        min_executions = max(1, self.config.skip_warmup_steps)
        if state.executed < min_executions:
            return False
        if state.skip_cooldown > 0:
            return False
        if self._current_context is None:
            return False
        progress = max(
            self._current_context.step_progress,
            self._current_context.noise_progress,
        )
        if progress < self.config.skip_min_progress:
            return False
        threshold = self.config.skip_threshold * (
            progress ** self.config.skip_progress_power
        )
        if state.ema_importance < threshold and state.skip_streak < self.config.skip_max_streak:
            if self.config.log_stats:
                self.logger.info(f"[Rabbit] SKIPPING {stage}[{index}]: importance={state.ema_importance:.4e} < "
                               f"threshold={threshold:.4e} (progress={progress:.2f})")
            return True
        return False
