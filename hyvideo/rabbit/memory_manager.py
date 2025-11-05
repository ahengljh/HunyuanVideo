"""
RabbitVideo Core Memory Manager
Orchestrates all memory optimization strategies.
"""

import torch
import gc
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from ..utils.memory_profiler import MemoryProfiler, get_memory_profiler, set_memory_profiler

@dataclass
class MemoryConfig:
    """Configuration for RabbitVideo memory management."""
    target_memory_gb: float = 24.0  # Target GPU memory usage
    offload_threshold_gb: float = 20.0  # Start offloading when above this
    prefetch_blocks: int = 2  # Number of blocks to prefetch
    cache_similarity_threshold: float = 0.95  # Threshold for frame similarity
    enable_profiling: bool = True
    enable_offloading: bool = True
    enable_caching: bool = True
    enable_gradient_checkpointing: bool = True
    aggressive_mode: bool = False  # More aggressive offloading if True
    debug_mode: bool = False

class RabbitMemoryManager:
    """Main memory management orchestrator for RabbitVideo."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.profiler = None
        self.offload_manager = None
        self.block_manager = None
        self.cache_manager = None

        # Memory tracking
        self.current_memory_gb = 0.0
        self.peak_memory_gb = 0.0
        self.memory_history = []

        # Model information
        self.model_config = {}
        self.total_blocks = 0
        self.block_memory_map = {}

        # Runtime state
        self.is_initialized = False
        self.in_inference = False
        self.current_timestep = 0
        self.total_timesteps = 0

        # Initialize profiler
        if self.config.enable_profiling:
            self.profiler = MemoryProfiler(enable_profiling=True)
            set_memory_profiler(self.profiler)
            self.profiler.start_monitoring()

    def initialize(self, model: torch.nn.Module, model_config: Dict[str, Any]):
        """Initialize memory manager with model information."""
        from .offload_manager import OffloadManager
        from .block_manager import BlockManager
        from .cache_manager import TemporalCacheManager

        self.model_config = model_config

        # Count transformer blocks
        self.total_blocks = self._count_blocks(model)
        print(f"[RabbitVideo] Detected {self.total_blocks} transformer blocks")

        # Analyze model memory footprint
        self._analyze_model_memory(model)

        # Initialize sub-managers
        if self.config.enable_offloading:
            self.offload_manager = OffloadManager(
                target_memory_gb=self.config.target_memory_gb,
                prefetch_blocks=self.config.prefetch_blocks,
                aggressive_mode=self.config.aggressive_mode
            )
            self.offload_manager.initialize(model)

        self.block_manager = BlockManager(total_blocks=self.total_blocks)
        self.block_manager.initialize(model)

        if self.config.enable_caching:
            self.cache_manager = TemporalCacheManager(
                similarity_threshold=self.config.cache_similarity_threshold
            )

        self.is_initialized = True

        # Print initialization summary
        self._print_init_summary()

    def _count_blocks(self, model: torch.nn.Module) -> int:
        """Count the number of transformer blocks in the model."""
        count = 0
        for name, module in model.named_modules():
            if 'MMDoubleStreamBlock' in module.__class__.__name__ or \
               'MMSingleStreamBlock' in module.__class__.__name__:
                count += 1
        return count

    def _analyze_model_memory(self, model: torch.nn.Module):
        """Analyze model memory footprint."""
        total_params = 0
        total_memory_mb = 0

        for name, param in model.named_parameters():
            params = param.numel()
            memory_mb = param.element_size() * params / (1024 * 1024)
            total_params += params
            total_memory_mb += memory_mb

            # Map blocks to their memory usage
            if 'double_blocks' in name or 'single_blocks' in name:
                block_idx = self._extract_block_index(name)
                if block_idx not in self.block_memory_map:
                    self.block_memory_map[block_idx] = 0
                self.block_memory_map[block_idx] += memory_mb

        print(f"[RabbitVideo] Model Analysis:")
        print(f"  Total parameters: {total_params / 1e9:.2f}B")
        print(f"  Total memory: {total_memory_mb / 1024:.2f}GB")

    def _extract_block_index(self, param_name: str) -> int:
        """Extract block index from parameter name."""
        import re
        match = re.search(r'blocks\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        return -1

    def _print_init_summary(self):
        """Print initialization summary."""
        print("\n" + "="*60)
        print("RabbitVideo Memory Manager Initialized")
        print("="*60)
        print(f"Target Memory: {self.config.target_memory_gb:.1f} GB")
        print(f"Offloading: {'Enabled' if self.config.enable_offloading else 'Disabled'}")
        print(f"Caching: {'Enabled' if self.config.enable_caching else 'Disabled'}")
        print(f"Gradient Checkpointing: {'Enabled' if self.config.enable_gradient_checkpointing else 'Disabled'}")
        print(f"Aggressive Mode: {'Yes' if self.config.aggressive_mode else 'No'}")
        print(f"Total Blocks: {self.total_blocks}")
        print("="*60 + "\n")

    def start_inference(self, num_timesteps: int, batch_size: int = 1):
        """Start inference phase and prepare memory management."""
        self.in_inference = True
        self.total_timesteps = num_timesteps
        self.current_timestep = 0

        if self.profiler:
            self.profiler.set_phase("inference_start")

        # Estimate memory requirements
        self._estimate_memory_requirements(batch_size)

        # Prepare offload manager
        if self.offload_manager:
            self.offload_manager.prepare_inference(num_timesteps)

        print(f"[RabbitVideo] Starting inference with {num_timesteps} timesteps")

    def _estimate_memory_requirements(self, batch_size: int):
        """Estimate memory requirements for inference."""
        # Rough estimation based on model size and batch
        base_memory_gb = sum(self.block_memory_map.values()) / 1024
        activation_memory_gb = base_memory_gb * batch_size * 2  # Rough estimate

        total_estimated_gb = base_memory_gb + activation_memory_gb

        if total_estimated_gb > self.config.target_memory_gb:
            blocks_to_offload = int((total_estimated_gb - self.config.target_memory_gb) /
                                   (base_memory_gb / self.total_blocks))
            print(f"[RabbitVideo] Estimated memory: {total_estimated_gb:.1f}GB")
            print(f"[RabbitVideo] Need to offload ~{blocks_to_offload} blocks")

    def before_block(self, block_idx: int, block_type: str = 'double') -> Dict[str, Any]:
        """Called before executing a transformer block."""
        context = {'block_idx': block_idx, 'block_type': block_type}

        # Update block manager
        if self.block_manager:
            self.block_manager.on_block_start(block_idx)

        # Handle offloading
        if self.offload_manager:
            # Ensure current block is on GPU first
            self.offload_manager.ensure_on_device(block_idx, 'cuda')

            # Check if we need to offload blocks (only if memory is really high)
            current_memory = self.get_current_memory_gb()
            if current_memory > self.config.offload_threshold_gb * 1.1:  # Add 10% buffer
                if self.config.debug_mode:
                    print(f"[RabbitVideo] Memory {current_memory:.1f}GB > threshold, considering offloading")
                self.offload_manager.offload_cold_blocks(block_idx)

            # Prefetch next blocks (but not too many)
            for i in range(1, min(2, self.config.prefetch_blocks + 1)):
                next_idx = block_idx + i
                if next_idx < self.total_blocks:
                    self.offload_manager.prefetch_block(next_idx)

        return context

    def after_block(self, block_idx: int, context: Dict[str, Any]):
        """Called after executing a transformer block."""
        if self.block_manager:
            self.block_manager.on_block_end(block_idx)

        # Update memory tracking
        self.update_memory_stats()

    def before_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        frame_idx: int) -> Tuple[Optional[torch.Tensor], bool]:
        """Called before attention computation to check cache."""
        if not self.cache_manager or not self.config.enable_caching:
            return None, False

        # Check if we have cached attention for similar frames
        cached_attention = self.cache_manager.get_cached_attention(
            q, k, v, frame_idx
        )

        if cached_attention is not None:
            if self.config.debug_mode:
                print(f"[RabbitVideo] Cache hit for frame {frame_idx}")
            return cached_attention, True

        return None, False

    def after_attention(self, attention_output: torch.Tensor, q: torch.Tensor,
                       k: torch.Tensor, v: torch.Tensor, frame_idx: int):
        """Called after attention computation to update cache."""
        if self.cache_manager and self.config.enable_caching:
            self.cache_manager.cache_attention(
                attention_output, q, k, v, frame_idx
            )

    def update_timestep(self, timestep: int):
        """Update current timestep in denoising process."""
        self.current_timestep = timestep

        if self.profiler:
            self.profiler.set_phase(f"timestep_{timestep}")

        # Clear caches periodically
        if timestep % 10 == 0 and self.cache_manager:
            self.cache_manager.clear_old_entries()

        # Adjust offloading strategy based on progress
        if self.offload_manager:
            progress = timestep / self.total_timesteps
            if progress > 0.8:  # Last 20% of timesteps
                # Be more aggressive with offloading
                self.offload_manager.set_aggressive_mode(True)

    def get_current_memory_gb(self) -> float:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def update_memory_stats(self):
        """Update memory statistics."""
        current_memory = self.get_current_memory_gb()
        self.current_memory_gb = current_memory
        self.peak_memory_gb = max(self.peak_memory_gb, current_memory)

        self.memory_history.append({
            'timestep': self.current_timestep,
            'memory_gb': current_memory
        })

    def should_checkpoint_gradient(self, block_idx: int) -> bool:
        """Determine if gradient checkpointing should be used for this block."""
        if not self.config.enable_gradient_checkpointing:
            return False

        # Checkpoint every other block in aggressive mode
        if self.config.aggressive_mode:
            return block_idx % 2 == 0

        # Checkpoint blocks in the middle (they're accessed less frequently)
        middle_start = self.total_blocks // 3
        middle_end = 2 * self.total_blocks // 3
        return middle_start <= block_idx <= middle_end

    def optimize_vae_processing(self, enable_tiling: bool = True,
                              tile_size: Tuple[int, int] = (256, 256)):
        """Optimize VAE processing with tiling."""
        context = {
            'enable_tiling': enable_tiling,
            'tile_size': tile_size
        }
        print(f"[RabbitVideo] VAE optimization: tiling={'enabled' if enable_tiling else 'disabled'}")
        return context

    def cleanup(self):
        """Cleanup and save profiling data."""
        if self.profiler:
            self.profiler.stop_monitoring()
            self.profiler.save_profile()

        # Print final statistics
        print("\n" + "="*60)
        print("RabbitVideo Memory Statistics")
        print("="*60)
        print(f"Peak Memory: {self.peak_memory_gb:.2f} GB")
        print(f"Target Memory: {self.config.target_memory_gb:.1f} GB")
        print(f"Success: {'✓' if self.peak_memory_gb <= self.config.target_memory_gb else '✗'}")

        if self.cache_manager:
            cache_stats = self.cache_manager.get_statistics()
            print(f"Cache Hits: {cache_stats.get('hits', 0)}")
            print(f"Cache Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")

        print("="*60)

    def emergency_cleanup(self):
        """Emergency memory cleanup when approaching limits."""
        print("[RabbitVideo] ⚠️ Emergency cleanup triggered!")

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        # Offload all possible blocks
        if self.offload_manager:
            self.offload_manager.emergency_offload()

        # Clear all caches
        if self.cache_manager:
            self.cache_manager.clear_all()

        # Force synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        current_memory = self.get_current_memory_gb()
        print(f"[RabbitVideo] Memory after cleanup: {current_memory:.2f} GB")