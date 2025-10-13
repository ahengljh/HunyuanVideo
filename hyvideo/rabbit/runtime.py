from __future__ import annotations

import logging
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
        self.latent_offload_device = torch.device(config.latent_offload_device)

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

        self.latent_cache_enabled = self.config.latent_offload

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

        self._prepare_module_residency()
        self.transformer.enable_rabbit_runtime(self)
        self.active = config.enabled

    # ------------------------------------------------------------------
    # Public API used by the pipeline
    # ------------------------------------------------------------------
    def register_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.latent_cache_enabled:
            return latents
        if latents.device == self.latent_offload_device:
            return latents
        latents_cpu = latents.to(self.latent_offload_device, non_blocking=self.device.type == "cuda")
        if self.latent_offload_device.type == "cpu" and self.config.latent_pin_memory:
            latents_cpu = latents_cpu.pin_memory()
        return latents_cpu

    def latents_to_device(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.latent_cache_enabled:
            return latents
        if latents.device == self.device:
            return latents
        return latents.to(self.device, non_blocking=self.device.type == "cuda")

    def after_step(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.latent_cache_enabled:
            return latents
        latents_cpu = latents.to(self.latent_offload_device, non_blocking=self.device.type == "cuda")
        if self.latent_offload_device.type == "cpu" and self.config.latent_pin_memory:
            latents_cpu = latents_cpu.pin_memory()
        return latents_cpu

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
        self.transformer.set_runtime_context(self._current_context)

    def before_block(
        self,
        stage: StageName,
        index: int,
        block: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
    ) -> BlockDecision:
        if self.config.skip_enabled and self.config.stage_enabled(stage):
            if self._should_skip(stage, index):
                outputs = tuple(t for t in inputs)
                state = self._states[stage][index]
                state.register_skip(self.config.skip_cooldown)
                self.stats["skipped"][stage][index] += 1
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

        if self.config.weights_offload_enabled:
            self._offload_block(stage, index, block)

        return outputs

    def finalize(self) -> Optional[Dict[str, Dict[str, float]]]:
        self.transformer.set_runtime_context(None)
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

        parts = []
        for stage in ("double", "single"):
            if stage not in summary["executed"]:
                continue
            exec_count = summary["executed"][stage]
            skip_count = summary["skipped"][stage]
            mean_importance = summary["mean_importance"].get(stage, 0.0)
            parts.append(
                f"{stage}: exec={exec_count} skip={skip_count} mean_importance={mean_importance:.4e}"
            )
        if parts:
            self.logger.info("Rabbit runtime summary | %s", " | ".join(parts))
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_module_residency(self):
        if not self.config.weights_offload_enabled:
            self.transformer.to(self.device)
            return

        self._move_persistent_modules_to_device()

        for stage, modules in self._blocks.items():
            target_set = self._offload_sets[stage]
            for idx, module in enumerate(modules):
                if idx in target_set:
                    module.to(self.offload_device)
                    self._residency[stage][idx] = "offload"
                else:
                    module.to(self.device)
                    self._residency[stage][idx] = "device"

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
        residency = self._residency[stage][index]
        if residency == "device":
            return

        if residency == "prefetch":
            if self._prefetch_stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(self._prefetch_stream)
            self._residency[stage][index] = "device"
            self._prefetched.pop((stage, index), None)
            return

        block.to(self.device, non_blocking=self.device.type == "cuda")
        self._residency[stage][index] = "device"

    def _schedule_prefetch(self, stage: StageName, index: int):
        if self._prefetch_stream is None:
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

    def _offload_block(
        self, stage: StageName, index: int, block: torch.nn.Module
    ):
        if index not in self._offload_sets[stage]:
            return
        residency = self._residency[stage][index]
        if residency == "offload":
            return
        block.to(self.offload_device, non_blocking=self.device.type == "cuda")
        self._residency[stage][index] = "offload"

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
        if state.executed < self.config.skip_warmup_steps:
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
            return True
        return False

