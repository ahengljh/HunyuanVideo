from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque

import torch
import torch.nn.functional as F

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


@dataclass
class CacheEntry:
    """Cache entry for block outputs with similarity checking."""
    input_hash: torch.Tensor  # Small hash of input for fast comparison
    outputs: Tuple[torch.Tensor, ...]
    step_index: int
    hits: int = 0

    def check_similarity(self, other_hash: torch.Tensor, threshold: float = 0.95) -> bool:
        """Check if input is similar enough to use cached output."""
        with torch.no_grad():
            similarity = F.cosine_similarity(
                self.input_hash.flatten().float().unsqueeze(0),
                other_hash.flatten().float().unsqueeze(0)
            ).item()
            return similarity >= threshold


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

        # Initialize caching system (disabled by default for safety)
        self.cache_enabled = getattr(config, 'enable_frame_cache', False)
        self.cache_threshold = getattr(config, 'cache_similarity_threshold', 0.98)  # Very high threshold
        self.block_caches: Dict[Tuple[str, int], deque] = {}  # (stage, idx) -> deque of CacheEntry
        self.max_cache_per_block = 3
        self.cache_stats = {"hits": 0, "misses": 0}

        # Advanced memory management for small devices
        self.memory_budget_mb = config.memory_budget_mb or 18000  # Default 18GB for safety
        self.memory_safety_margin = getattr(config, 'memory_safety_margin', 0.15)  # 15% safety
        self.block_memory: Dict[Tuple[str, int], int] = {}  # Track memory per block
        self.access_history: Dict[Tuple[str, int], deque] = {}  # Track access patterns
        self.prefetch_queue: deque = deque(maxlen=3)  # Blocks to prefetch

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

    def _compute_input_hash(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute a compact hash of inputs for similarity checking."""
        hashes = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                with torch.no_grad():
                    # Take strategic samples: first, middle, last elements
                    flat = inp.flatten()
                    n = flat.numel()
                    if n > 30:
                        # Sample from beginning, middle, and end
                        indices = torch.tensor([0, 1, 2, n//2-1, n//2, n//2+1, n-3, n-2, n-1])
                        sample = flat[indices]
                    else:
                        sample = flat[:min(10, n)]
                    hashes.append(sample.detach().cpu())

        if hashes:
            return torch.cat(hashes)
        return torch.tensor([])

    def _check_cache(self, stage: str, index: int, inputs: Tuple[torch.Tensor, ...]) -> Optional[Tuple[torch.Tensor, ...]]:
        """Check if we have cached outputs for similar inputs."""
        if not self.cache_enabled:
            return None

        cache_key = (stage, index)
        if cache_key not in self.block_caches:
            return None

        input_hash = self._compute_input_hash(inputs)
        step = self._current_context.step_index if self._current_context else 0

        # Check cache entries (recent first for temporal locality)
        for entry in reversed(self.block_caches[cache_key]):
            # Prioritize temporally close entries
            if abs(entry.step_index - step) <= 2 and entry.check_similarity(input_hash, self.cache_threshold):
                entry.hits += 1
                self.cache_stats["hits"] += 1
                return entry.outputs

        # Check all entries if no temporal match
        for entry in self.block_caches[cache_key]:
            if entry.check_similarity(input_hash, self.cache_threshold):
                entry.hits += 1
                self.cache_stats["hits"] += 1
                return entry.outputs

        self.cache_stats["misses"] += 1
        return None

    def _update_cache(self, stage: str, index: int, inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, ...]):
        """Store outputs in cache for future reuse."""
        if not self.cache_enabled:
            return

        cache_key = (stage, index)
        if cache_key not in self.block_caches:
            self.block_caches[cache_key] = deque(maxlen=self.max_cache_per_block)

        input_hash = self._compute_input_hash(inputs)
        step = self._current_context.step_index if self._current_context else 0

        # Clone outputs to avoid affecting computation
        # Keep computation graph intact, only detach when retrieving from cache
        cached_outputs = tuple(out.clone() for out in outputs)
        entry = CacheEntry(input_hash, cached_outputs, step)
        self.block_caches[cache_key].append(entry)

    def _get_block_memory(self, block: torch.nn.Module) -> int:
        """Estimate memory usage of a block in bytes."""
        if not hasattr(block, '_cached_size'):
            total_bytes = 0
            for param in block.parameters(recurse=True):
                total_bytes += param.numel() * param.element_size()
            block._cached_size = total_bytes
        return block._cached_size

    def _should_keep_on_device(self, stage: str, index: int) -> bool:
        """Determine if block should stay on device based on access patterns."""
        key = (stage, index)
        if key not in self.access_history:
            return False

        # Check access frequency in recent steps
        recent_accesses = list(self.access_history[key])[-5:]
        if len(recent_accesses) < 2:
            return False

        # Keep on device if accessed frequently
        access_density = len(recent_accesses) / 5
        return access_density > 0.6

    def _manage_memory_for_block(self, stage: str, index: int, block: torch.nn.Module):
        """Intelligent memory management for small devices."""
        if not self.config.weights_offload_enabled:
            return

        key = (stage, index)
        block_size = self._get_block_memory(block)

        # Track this access
        if key not in self.access_history:
            self.access_history[key] = deque(maxlen=10)
        step = self._current_context.step_index if self._current_context else 0
        self.access_history[key].append(step)

        # Calculate current GPU memory usage
        current_usage = sum(self.block_memory.values())
        available_budget = self.memory_budget_mb * 1024 * 1024 * (1 - self.memory_safety_margin)

        # If block is already on device and we have space, keep it
        if self._residency[stage][index] == "device":
            if current_usage + block_size <= available_budget:
                self.block_memory[key] = block_size
                return

        # Need to load block - check if we need to evict others
        space_needed = block_size
        if current_usage + space_needed > available_budget:
            # Find blocks to evict (LRU with importance weighting)
            eviction_candidates = []
            for (s, i), history in self.access_history.items():
                if (s, i) == key or (s, i) not in self.block_memory:
                    continue
                if self._residency[s][i] == "device":
                    # Calculate eviction score (lower = evict first)
                    last_access = max(history) if history else -1
                    importance = self._states[s][i].ema_importance
                    score = last_access * (1 + importance)
                    eviction_candidates.append((score, s, i))

            # Evict blocks with lowest scores
            eviction_candidates.sort()
            for _, s, i in eviction_candidates:
                if current_usage + space_needed <= available_budget:
                    break
                evict_block = self._blocks[s][i]
                self._offload_block(s, i, evict_block)
                current_usage -= self.block_memory.pop((s, i), 0)

        # Load the required block
        self._ensure_on_device(stage, index, block)
        self.block_memory[key] = block_size

    def before_block(
        self,
        stage: StageName,
        index: int,
        block: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
    ) -> BlockDecision:
        if self.profiling_active:
            return BlockDecision(skip=False)

        # First check cache for computation reuse
        cached_outputs = self._check_cache(stage, index, inputs)
        if cached_outputs is not None:
            self.stats["skipped"][stage][index] += 1
            if self.config.log_stats:
                self.logger.info(f"[Rabbit] CACHE HIT: {stage}[{index}] - Reusing cached outputs")
            return BlockDecision(skip=True, outputs=cached_outputs)

        # Original skip logic
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

        # Smart memory management for small devices
        if self.config.weights_offload_enabled:
            self._manage_memory_for_block(stage, index, block)
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

        # Update cache with new outputs
        self._update_cache(stage, index, inputs, outputs)

        if self.config.log_stats and state.total_observations % 10 == 0:
            self.logger.info(f"[Rabbit] Block EXECUTED: {stage}[{index}] - "
                           f"delta={delta:.4e}, importance={state.ema_importance:.4e}")

        # Smart offloading decision
        should_retain = self._maybe_promote_hot_block(stage, index, block)

        if self.config.weights_offload_enabled and not should_retain:
            # Check if this block should be kept based on access patterns
            if not self._should_keep_on_device(stage, index):
                self._offload_block(stage, index, block)
                self.block_memory.pop((stage, index), None)

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

        # Report cache statistics
        if self.cache_enabled:
            total_cache_accesses = self.cache_stats["hits"] + self.cache_stats["misses"]
            if total_cache_accesses > 0:
                hit_rate = self.cache_stats["hits"] / total_cache_accesses
                self.logger.info("-" * 40)
                self.logger.info("[Rabbit] CACHE STATISTICS:")
                self.logger.info(f"  Cache hit rate: {hit_rate:.2%}")
                self.logger.info(f"  Total hits: {self.cache_stats['hits']}")
                self.logger.info(f"  Total misses: {self.cache_stats['misses']}")
                self.logger.info(f"  Computation saved by cache: ~{hit_rate * 30:.1f}%")

        # Report memory management statistics
        if self.config.weights_offload_enabled and self.block_memory:
            peak_memory_mb = max(sum(self.block_memory.values()), 1) / (1024 * 1024)
            self.logger.info("-" * 40)
            self.logger.info("[Rabbit] MEMORY MANAGEMENT:")
            self.logger.info(f"  Peak GPU memory for blocks: {peak_memory_mb:.1f} MB")
            self.logger.info(f"  Memory budget: {self.memory_budget_mb:.1f} MB")
            self.logger.info(f"  Safety margin: {self.memory_safety_margin:.1%}")

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
