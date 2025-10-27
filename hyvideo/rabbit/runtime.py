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
    cache_entry: Optional["BlockCacheEntry"] = None
    cache_age: int = 0

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


@dataclass
class BlockCacheEntry:
    outputs: Tuple[torch.Tensor, ...]
    signature: torch.Tensor
    step_index: int
    token_masks: Optional[Tuple[Optional[torch.Tensor], ...]] = None


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
        self.cache_device = torch.device(config.cache_device)

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
            "cache_hits": {
                stage: [0 for _ in modules] for stage, modules in self._blocks.items()
            },
        }
        self._cache_whitelist: Dict[StageName, Optional[set]] = {
            stage: None for stage in self._blocks.keys()
        }
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
                for stage, indices in config.offload_plan.items():
                    if indices:
                        self.logger.info(f"[Rabbit] Offloading {stage} blocks: {len(indices)} blocks")
            if config.cache_enabled:
                self.logger.info(
                    "[Rabbit] Output caching enabled on %s blocks (device: %s, threshold=%.3g)",
                    config.cache_stage,
                    config.cache_device,
                    config.cache_threshold,
                )
            if config.latent_offload:
                self.logger.info(f"[Rabbit] Latent Offloading: ENABLED")
            self.logger.info("=" * 60)

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
        cached = self._try_cache_reuse(stage, index, inputs)
        if cached is not None:
            return BlockDecision(skip=True, outputs=cached)

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

        if self.config.weights_offload_enabled:
            self._offload_block(stage, index, block)
        if self._stage_cache_enabled(stage):
            self._update_cache_entry(stage, index, block_inputs=inputs, block_outputs=outputs)

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
            "cache_hits": {
                stage: sum(values) for stage, values in self.stats["cache_hits"].items()
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
            cache_hits = summary["cache_hits"].get(stage, 0)
            if cache_hits > 0:
                self.logger.info(f"  - Cache hits: {cache_hits}")

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
        if self.config.cache_enabled:
            self.logger.info(f"[Rabbit] Output caching: ENABLED (device: {self.config.cache_device})")
        if self.config.latent_offload:
            self.logger.info(f"[Rabbit] Latent offloading: ENABLED")

        self.logger.info("=" * 60)
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_module_residency(self):
        if self.profiling_active or not self.config.weights_offload_enabled:
            self.transformer.to(self.device)
            for stage, modules in self._blocks.items():
                for idx, module in enumerate(modules):
                    module.to(self.device)
                    self._residency[stage][idx] = "device"
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
            return

        block.to(self.device, non_blocking=self.device.type == "cuda")
        self._residency[stage][index] = "device"

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

    def _stage_cache_enabled(self, stage: StageName) -> bool:
        if not self.config.cache_enabled or self.profiling_active:
            return False
        allowlist = self._cache_whitelist.get(stage)
        if allowlist is not None and len(allowlist) == 0:
            return False
        if self.config.cache_stage == "both":
            return True
        return stage == self.config.cache_stage

    def _build_signature(self, tensors: Tuple[torch.Tensor, ...]) -> Optional[torch.Tensor]:
        stats: List[torch.Tensor] = []
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            flat = tensor.detach().float()
            if flat.numel() == 0:
                continue
            mean_val = flat.mean()
            std_val = flat.std(unbiased=False)
            stats.append(torch.stack((mean_val, std_val)))
        if not stats:
            return None
        stacked = torch.stack(stats).mean(dim=0)
        return stacked.to(self.cache_device)

    def _signature_drift(self, current: torch.Tensor, cached: torch.Tensor) -> float:
        diff = torch.mean(torch.abs(current - cached))
        norm = torch.mean(torch.abs(cached)).clamp(min=1e-6)
        return (diff / norm).item()

    def _try_cache_reuse(
        self, stage: StageName, index: int, inputs: Tuple[torch.Tensor, ...]
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        if not self._stage_cache_enabled(stage):
            return None
        state = self._states[stage][index]
        entry = state.cache_entry
        if entry is None or self._current_context is None:
            return None
        allowlist = self._cache_whitelist.get(stage)
        if allowlist is not None and index not in allowlist:
            return None
        age = self._current_context.step_index - entry.step_index
        if age <= 0 or age > self.config.cache_max_age:
            return None
        # Inspired by ToCa (Zou et al., 2025), prefer caching for low-impact tokens/blocks only.
        if state.ema_importance > self.config.cache_min_importance:
            return None
        signature = self._build_signature(inputs)
        if signature is None:
            return None
        drift = self._signature_drift(signature, entry.signature)
        if drift > self.config.cache_threshold:
            return None

        outputs_list: List[torch.Tensor] = []
        for i, tensor in enumerate(entry.outputs):
            out_tensor = (
                tensor.to(self.device, non_blocking=self.device.type == "cuda")
                if tensor.device != self.device
                else tensor.clone()
            )
            mask = None
            if entry.token_masks is not None and i < len(entry.token_masks):
                mask = entry.token_masks[i]
            if mask is not None:
                mask_dev = mask.to(self.device, non_blocking=self.device.type == "cuda")
                expanded = self._expand_token_mask(mask_dev, out_tensor)
                input_tensor = inputs[i].to(self.device)
                out_tensor = torch.where(expanded, out_tensor, input_tensor)
            outputs_list.append(out_tensor)
        outputs = tuple(outputs_list)
        self.stats["cache_hits"][stage][index] += 1
        state.cache_age += 1
        if self.config.log_stats:
            self.logger.info(
                "[Rabbit] Cache REUSE: %s[%d] age=%d drift=%.3g",
                stage,
                index,
                age,
                drift,
            )
        return outputs

    def _update_cache_entry(
        self,
        stage: StageName,
        index: int,
        block_inputs: Tuple[torch.Tensor, ...],
        block_outputs: Tuple[torch.Tensor, ...],
    ):
        # Inspired by ProfilingDiT (Ma et al., 2025) and TeaCache (Liu et al., 2025),
        # we only cache low-variance blocks and track lightweight signatures.
        signature = self._build_signature(block_inputs)
        if signature is None or self._current_context is None:
            self._states[stage][index].cache_entry = None
            return
        allowlist = self._cache_whitelist.get(stage)
        if allowlist is not None and index not in allowlist:
            self._states[stage][index].cache_entry = None
            return
        outputs = tuple(
            tensor.detach().to(self.cache_device, non_blocking=self.device.type == "cuda")
            for tensor in block_outputs
        )
        token_masks: List[Optional[torch.Tensor]] = []
        ratio = self.config.cache_token_ratio
        if ratio < 1.0:
            for inp, out in zip(block_inputs, block_outputs):
                mask = self._compute_token_mask(inp, out, ratio)
                token_masks.append(
                    mask.to(self.cache_device) if mask is not None else None
                )
        else:
            token_masks = [None for _ in block_outputs]
        self._states[stage][index].cache_entry = BlockCacheEntry(
            outputs=outputs,
            signature=signature,
            step_index=self._current_context.step_index,
            token_masks=tuple(token_masks),
        )
        self._states[stage][index].cache_age = 0

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

        if self.config.profile_cache:
            for stage in self._blocks.keys():
                self._cache_whitelist[stage] = set(profile_sets.get(stage, set()))

        if self.config.profile_offload and self.config.weights_offload_enabled:
            new_plan: Dict[StageName, set] = {}
            for stage in self._blocks.keys():
                base = set(self._offload_sets.get(stage, set()))
                new_plan[stage] = base.union(profile_sets.get(stage, set()))
            self._apply_offload_plan(new_plan)

        if self.config.log_stats:
            for stage, indices in profile_sets.items():
                if indices:
                    self.logger.info(
                        "[Rabbit] Profiling selected low-importance %s blocks: %s",
                        stage,
                        sorted(indices),
                    )
            if self.config.profile_offload and self.config.weights_offload_enabled:
                self.logger.info("[Rabbit] Profiling-adjusted offload plan applied")
            if self.config.profile_cache:
                self.logger.info("[Rabbit] Profiling-adjusted caching whitelist applied")

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

    def _compute_token_mask(
        self, inputs: torch.Tensor, outputs: torch.Tensor, ratio: float
    ) -> Optional[torch.Tensor]:
        if ratio <= 0 or outputs.ndim < 3:
            return None
        if inputs.shape[:2] != outputs.shape[:2]:
            return None
        # Expect shape (B, L, ...). Compute per-token delta averaged over batch and channels.
        deltas = (outputs - inputs).abs()
        dims = list(range(2, deltas.ndim))
        if dims:
            deltas = deltas.mean(dim=dims)
        deltas = deltas.mean(dim=0)  # average across batch
        tokens = deltas.shape[0]
        keep = max(0, int(round(tokens * ratio)))
        if keep >= tokens:
            return None
        mask = torch.zeros(tokens, dtype=torch.bool, device=deltas.device)
        if keep > 0:
            _, indices = torch.topk(-deltas, k=keep, largest=False)
            mask.scatter_(0, indices, True)
        return mask

    def _expand_token_mask(self, mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            return mask
        shape = [1] * tensor.ndim
        shape[1] = mask.shape[0]
        expanded = mask.view(*shape)
        return expanded

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
