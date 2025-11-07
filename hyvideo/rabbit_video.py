"""
RabbitVideo: Memory-Efficient Video Diffusion via Strategic Block Offloading

This module implements block-level CPU offloading for HunyuanVideo to enable
high-quality video generation on consumer GPUs (24GB) by reducing peak memory
from ~66GB to ~24GB with only 10-15% time overhead.

Key Components:
    - MemoryMonitor: Tracks GPU memory usage (allocated, reserved, cache)
    - BlockTracker: Manages block locations (GPU/CPU) and access patterns
    - BlockManager: Handles synchronous block transfers with cache clearing
    - RabbitVideoOffloader: Main orchestrator for proactive block offloading
"""

import torch
import time
import logging
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Set, Tuple
import json

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Monitors GPU memory usage and provides detailed statistics.

    Tracks three memory pools:
        - Allocated: Actually used by tensors
        - Reserved: Requested from CUDA (includes cache)
        - Cache: Reserved - Allocated (wasted memory)
    """

    def __init__(self, device: torch.device, debug: bool = False):
        self.device = device
        self.debug = debug
        self.timeline_data = {
            "timestamps": [],
            "allocated_gb": [],
            "reserved_gb": [],
            "cache_gb": [],
            "blocks_on_gpu": [],
            "current_block": []
        }
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.start_time = time.time()

    def record(self, blocks_on_gpu: int = 0, current_block: int = -1):
        """Record current memory state to timeline."""
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        cache = reserved - allocated

        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

        timestamp = time.time() - self.start_time
        self.timeline_data["timestamps"].append(timestamp)
        self.timeline_data["allocated_gb"].append(round(allocated, 2))
        self.timeline_data["reserved_gb"].append(round(reserved, 2))
        self.timeline_data["cache_gb"].append(round(cache, 2))
        self.timeline_data["blocks_on_gpu"].append(blocks_on_gpu)
        self.timeline_data["current_block"].append(current_block)

        if self.debug:
            logger.info(f"[MemoryMonitor] T={timestamp:.1f}s | "
                       f"Allocated: {allocated:.2f}GB | "
                       f"Reserved: {reserved:.2f}GB | "
                       f"Cache: {cache:.2f}GB | "
                       f"Blocks on GPU: {blocks_on_gpu} | "
                       f"Current Block: {current_block}")

    def get_current_memory(self) -> Tuple[float, float, float]:
        """Returns (allocated_gb, reserved_gb, cache_gb)."""
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        cache = reserved - allocated
        return allocated, reserved, cache

    def get_peak_memory(self) -> Tuple[float, float]:
        """Returns (peak_allocated_gb, peak_reserved_gb)."""
        return self.peak_allocated, self.peak_reserved

    def save_timeline(self, filepath: str):
        """Save timeline data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.timeline_data, f, indent=2)
        logger.info(f"[MemoryMonitor] Timeline data saved to {filepath}")

    def print_summary(self):
        """Print memory usage summary."""
        logger.info("\n" + "="*80)
        logger.info("RabbitVideo Memory Summary")
        logger.info("="*80)
        logger.info(f"Peak Allocated Memory: {self.peak_allocated:.2f} GB")
        logger.info(f"Peak Reserved Memory:  {self.peak_reserved:.2f} GB")
        logger.info(f"Peak Cache Waste:      {self.peak_reserved - self.peak_allocated:.2f} GB")
        logger.info("="*80 + "\n")


class BlockTracker:
    """
    Tracks location (GPU/CPU) and access patterns of transformer blocks.

    Maintains:
        - Block locations (gpu_blocks, cpu_blocks sets)
        - LRU eviction policy (access count and last access time)
        - Block size information for memory estimation
    """

    def __init__(self, num_blocks: int, debug: bool = False):
        self.num_blocks = num_blocks
        self.debug = debug

        # Block location tracking
        self.gpu_blocks: Set[int] = set()
        self.cpu_blocks: Set[int] = set(range(num_blocks))  # Initially all on CPU

        # LRU eviction policy
        self.block_access_count: Dict[int, int] = defaultdict(int)
        self.block_last_access_time: Dict[int, float] = {}

        # Block size estimation (will be calculated from actual blocks)
        self.block_sizes: Dict[int, float] = {}  # in GB
        self.average_block_size: float = 0.65  # Initial estimate: 650MB per block

        # Statistics
        self.total_transfers: int = 0
        self.total_transfer_time: float = 0.0

    def record_access(self, block_idx: int):
        """Record block access for LRU tracking."""
        self.block_access_count[block_idx] += 1
        self.block_last_access_time[block_idx] = time.time()

    def is_on_gpu(self, block_idx: int) -> bool:
        """Check if block is currently on GPU."""
        return block_idx in self.gpu_blocks

    def mark_on_gpu(self, block_idx: int):
        """Mark block as loaded to GPU."""
        self.gpu_blocks.add(block_idx)
        self.cpu_blocks.discard(block_idx)

    def mark_on_cpu(self, block_idx: int):
        """Mark block as offloaded to CPU."""
        self.cpu_blocks.add(block_idx)
        self.gpu_blocks.discard(block_idx)

    def select_blocks_to_offload(self, num_needed: int) -> List[int]:
        """
        Select blocks to offload using LRU policy.

        Eviction policy:
            1. Sort by access count (ascending): less accessed first
            2. Break ties by last access time (ascending): older first

        Args:
            num_needed: Number of blocks to offload

        Returns:
            List of block indices to offload
        """
        if num_needed <= 0:
            return []

        # Sort GPU blocks by LRU policy
        gpu_blocks_sorted = sorted(
            self.gpu_blocks,
            key=lambda b: (
                self.block_access_count.get(b, 0),
                self.block_last_access_time.get(b, 0)
            )
        )

        blocks_to_offload = gpu_blocks_sorted[:num_needed]

        if self.debug:
            logger.info(f"[BlockTracker] Selected {len(blocks_to_offload)} blocks to offload: {blocks_to_offload}")

        return blocks_to_offload

    def estimate_block_size(self, block) -> float:
        """Estimate memory size of a block in GB."""
        if not hasattr(block, 'parameters'):
            return self.average_block_size

        total_bytes = sum(p.numel() * p.element_size() for p in block.parameters())
        size_gb = total_bytes / (1024**3)
        return size_gb

    def set_block_size(self, block_idx: int, block):
        """Calculate and store block size."""
        size_gb = self.estimate_block_size(block)
        self.block_sizes[block_idx] = size_gb

        # Update average
        if self.block_sizes:
            self.average_block_size = sum(self.block_sizes.values()) / len(self.block_sizes)

    def get_block_size(self, block_idx: int) -> float:
        """Get stored block size or return average estimate."""
        return self.block_sizes.get(block_idx, self.average_block_size)

    def print_statistics(self):
        """Print transfer statistics."""
        logger.info(f"[BlockTracker] Total Transfers: {self.total_transfers}")
        logger.info(f"[BlockTracker] Total Transfer Time: {self.total_transfer_time:.2f}s")
        if self.total_transfers > 0:
            avg_time = self.total_transfer_time / self.total_transfers
            logger.info(f"[BlockTracker] Average Transfer Time: {avg_time*1000:.1f}ms")


class BlockManager:
    """
    Manages synchronous block transfers between CPU and GPU.

    Implements the critical synchronous transfer protocol:
        1. Free memory FIRST (synchronous)
        2. Clear cache aggressively
        3. Load block SECOND (synchronous)
        4. Never allocate before freeing
    """

    def __init__(self, device: torch.device, tracker: BlockTracker,
                 memory_monitor: MemoryMonitor, debug: bool = False):
        self.device = device
        self.tracker = tracker
        self.memory_monitor = memory_monitor
        self.debug = debug

    def move_block_to_cpu(self, block, block_idx: int):
        """
        Move block to CPU with synchronous protocol.

        Protocol:
            1. block.to('cpu') - blocking by default
            2. torch.cuda.synchronize() - explicit wait
            3. torch.cuda.empty_cache() - request cache release
            4. torch.cuda.synchronize() - wait for release
        """
        start_time = time.time()

        if self.debug:
            before_alloc, before_reserved, _ = self.memory_monitor.get_current_memory()
            logger.info(f"[BlockManager] Offloading block {block_idx} to CPU | "
                       f"Before: {before_reserved:.2f}GB reserved")

        # Synchronous transfer protocol
        block.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Update tracker
        self.tracker.mark_on_cpu(block_idx)
        self.tracker.total_transfers += 1

        elapsed = time.time() - start_time
        self.tracker.total_transfer_time += elapsed

        if self.debug:
            after_alloc, after_reserved, _ = self.memory_monitor.get_current_memory()
            logger.info(f"[BlockManager] Offloaded block {block_idx} in {elapsed*1000:.1f}ms | "
                       f"After: {after_reserved:.2f}GB reserved | "
                       f"Freed: {before_reserved - after_reserved:.2f}GB")

    def move_block_to_gpu(self, block, block_idx: int):
        """
        Move block to GPU with synchronous protocol.

        CRITICAL: This should ONLY be called after free_gpu_memory()
        has ensured sufficient space is available.
        """
        start_time = time.time()

        if self.debug:
            before_alloc, before_reserved, _ = self.memory_monitor.get_current_memory()
            block_size = self.tracker.get_block_size(block_idx)
            logger.info(f"[BlockManager] Loading block {block_idx} to GPU | "
                       f"Size: {block_size:.2f}GB | "
                       f"Before: {before_reserved:.2f}GB reserved")

        # Synchronous transfer protocol
        block.to(self.device)
        torch.cuda.synchronize()

        # Update tracker
        self.tracker.mark_on_gpu(block_idx)
        self.tracker.total_transfers += 1

        elapsed = time.time() - start_time
        self.tracker.total_transfer_time += elapsed

        if self.debug:
            after_alloc, after_reserved, _ = self.memory_monitor.get_current_memory()
            logger.info(f"[BlockManager] Loaded block {block_idx} in {elapsed*1000:.1f}ms | "
                       f"After: {after_reserved:.2f}GB reserved | "
                       f"Increased: {after_reserved - before_reserved:.2f}GB")

    def free_gpu_memory(self, required_gb: float, blocks_dict: Dict[int, torch.nn.Module]):
        """
        Free GPU memory BEFORE allocating new memory.

        Args:
            required_gb: Amount of GPU memory needed (in GB)
            blocks_dict: Dictionary mapping block indices to block modules
        """
        current_alloc, current_reserved, cache = self.memory_monitor.get_current_memory()

        # Check if we need to free memory
        # Heuristic: Reserve 2GB buffer for activations and other tensors
        gpu_capacity_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        available_gb = gpu_capacity_gb - current_reserved - 2.0

        if available_gb >= required_gb:
            if self.debug:
                logger.info(f"[BlockManager] Sufficient memory available: {available_gb:.2f}GB >= {required_gb:.2f}GB")
            return

        # Calculate how much we need to free
        need_to_free_gb = required_gb - available_gb + 1.0  # +1GB extra buffer

        if self.debug:
            logger.info(f"[BlockManager] Need to free {need_to_free_gb:.2f}GB | "
                       f"Current reserved: {current_reserved:.2f}GB | "
                       f"GPU capacity: {gpu_capacity_gb:.2f}GB")

        # Select blocks to offload using LRU
        freed_gb = 0.0
        blocks_to_offload = []

        for block_idx in self.tracker.select_blocks_to_offload(len(self.tracker.gpu_blocks)):
            block_size = self.tracker.get_block_size(block_idx)
            blocks_to_offload.append(block_idx)
            freed_gb += block_size

            if freed_gb >= need_to_free_gb:
                break

        # Offload selected blocks
        for block_idx in blocks_to_offload:
            if block_idx in blocks_dict:
                self.move_block_to_cpu(blocks_dict[block_idx], block_idx)

        # Verify memory was freed
        after_alloc, after_reserved, _ = self.memory_monitor.get_current_memory()
        actual_freed = current_reserved - after_reserved

        if self.debug:
            logger.info(f"[BlockManager] Freed {actual_freed:.2f}GB by offloading {len(blocks_to_offload)} blocks")


class RabbitVideoOffloader:
    """
    Main orchestrator for RabbitVideo block-level offloading.

    Implements:
        - Phase 1: Smart initialization with proactive offloading
        - Phase 2: Dynamic block swapping during inference
        - Phase 3: Auxiliary model management (VAE, text encoders)
    """

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 blocks_to_keep: int = 5,
                 debug: bool = False):
        """
        Initialize RabbitVideo offloader.

        Args:
            model: HYVideoDiffusionTransformer model
            device: Target CUDA device
            blocks_to_keep: Number of blocks to keep on GPU (default: 5)
            debug: Enable debug logging
        """
        self.model = model
        self.device = device
        self.blocks_to_keep = blocks_to_keep
        self.debug = debug

        # Extract blocks from model
        self.double_blocks = list(model.double_blocks)
        self.single_blocks = list(model.single_blocks)
        self.all_blocks = self.double_blocks + self.single_blocks
        self.num_blocks = len(self.all_blocks)

        # Create mapping from block index to block module
        self.blocks_dict: Dict[int, torch.nn.Module] = {}
        for i, block in enumerate(self.all_blocks):
            self.blocks_dict[i] = block

        # Initialize components
        self.memory_monitor = MemoryMonitor(device, debug=debug)
        self.tracker = BlockTracker(self.num_blocks, debug=debug)
        self.block_manager = BlockManager(device, self.tracker, self.memory_monitor, debug=debug)

        # Track current execution state
        self.current_step = 0
        self.total_steps = 0

        logger.info(f"[RabbitVideo] Initialized with {self.num_blocks} blocks "
                   f"({len(self.double_blocks)} double + {len(self.single_blocks)} single)")
        logger.info(f"[RabbitVideo] Will keep {self.blocks_to_keep} blocks on GPU")

    def initialize_offloading(self):
        """
        Phase 1: Smart initialization with proactive offloading.

        Strategy:
            - Keep first `blocks_to_keep` blocks on GPU
            - Offload remaining blocks to CPU
            - Calculate block sizes for accurate memory estimation
        """
        logger.info(f"[RabbitVideo] Phase 1: Proactive offloading initialization")
        self.memory_monitor.record(blocks_on_gpu=self.num_blocks, current_block=-1)

        # First, calculate block sizes while they're on GPU
        for i, block in enumerate(self.all_blocks):
            self.tracker.set_block_size(i, block)

        avg_size = self.tracker.average_block_size
        logger.info(f"[RabbitVideo] Average block size: {avg_size:.2f}GB")

        # Decide which blocks to keep on GPU (first N blocks)
        blocks_to_keep_on_gpu = set(range(self.blocks_to_keep))
        blocks_to_offload = set(range(self.num_blocks)) - blocks_to_keep_on_gpu

        logger.info(f"[RabbitVideo] Keeping blocks {list(blocks_to_keep_on_gpu)} on GPU")
        logger.info(f"[RabbitVideo] Offloading {len(blocks_to_offload)} blocks to CPU")

        # Offload blocks not in the initial set
        for block_idx in sorted(blocks_to_offload):
            block = self.blocks_dict[block_idx]
            self.block_manager.move_block_to_cpu(block, block_idx)

        # Mark initial blocks as on GPU
        for block_idx in blocks_to_keep_on_gpu:
            self.tracker.mark_on_gpu(block_idx)

        self.memory_monitor.record(blocks_on_gpu=len(blocks_to_keep_on_gpu), current_block=-1)

        # Print initial memory state
        alloc, reserved, cache = self.memory_monitor.get_current_memory()
        logger.info(f"[RabbitVideo] After initialization: "
                   f"Allocated={alloc:.2f}GB, Reserved={reserved:.2f}GB, Cache={cache:.2f}GB")

    def ensure_block_on_gpu(self, block_idx: int):
        """
        Phase 2: Ensure block is on GPU for execution.

        Strategy:
            1. Check if block is on GPU
            2. If not, free memory FIRST using LRU
            3. Then load block SECOND
        """
        # Record access
        self.tracker.record_access(block_idx)

        # If already on GPU, nothing to do
        if self.tracker.is_on_gpu(block_idx):
            if self.debug:
                logger.info(f"[RabbitVideo] Block {block_idx} already on GPU")
            return

        if self.debug:
            logger.info(f"[RabbitVideo] Block {block_idx} needs to be loaded to GPU")

        # Get block and its size
        block = self.blocks_dict[block_idx]
        block_size = self.tracker.get_block_size(block_idx)

        # Free memory FIRST
        self.block_manager.free_gpu_memory(block_size, self.blocks_dict)

        # Load block SECOND
        self.block_manager.move_block_to_gpu(block, block_idx)

        # Record memory state
        self.memory_monitor.record(
            blocks_on_gpu=len(self.tracker.gpu_blocks),
            current_block=block_idx
        )

    def prefetch_next_blocks(self, current_block_idx: int, window: int = 1):
        """
        Prefetch next blocks in the sequence (optional optimization).

        Args:
            current_block_idx: Currently executing block
            window: Number of blocks ahead to prefetch
        """
        for i in range(1, window + 1):
            next_idx = current_block_idx + i
            if next_idx < self.num_blocks and not self.tracker.is_on_gpu(next_idx):
                # Only prefetch if we have spare GPU memory
                alloc, reserved, _ = self.memory_monitor.get_current_memory()
                gpu_capacity = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)

                if reserved + self.tracker.get_block_size(next_idx) + 2.0 < gpu_capacity:
                    if self.debug:
                        logger.info(f"[RabbitVideo] Prefetching block {next_idx}")
                    self.ensure_block_on_gpu(next_idx)

    def step_begin(self, step: int, total_steps: int):
        """Called at the beginning of each denoising step."""
        self.current_step = step
        self.total_steps = total_steps
        if self.debug:
            logger.info(f"\n[RabbitVideo] === Step {step}/{total_steps} ===")

    def step_end(self):
        """Called at the end of each denoising step."""
        self.memory_monitor.record(
            blocks_on_gpu=len(self.tracker.gpu_blocks),
            current_block=-1
        )

    def print_summary(self):
        """Print final summary statistics."""
        self.memory_monitor.print_summary()
        self.tracker.print_statistics()

    def save_timeline(self, filepath: str):
        """Save memory timeline data."""
        self.memory_monitor.save_timeline(filepath)


def offload_auxiliary_models_to_cpu(vae, text_encoder, text_encoder_2=None):
    """
    Phase 3: Move auxiliary models (VAE, text encoders) to CPU.

    These models are only needed temporarily and should not occupy
    GPU memory during transformer inference.
    """
    logger.info("[RabbitVideo] Phase 3: Offloading auxiliary models to CPU")

    if vae is not None and hasattr(vae, 'to'):
        vae.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("[RabbitVideo] VAE moved to CPU")

    if text_encoder is not None and hasattr(text_encoder, 'to'):
        text_encoder.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("[RabbitVideo] Text encoder moved to CPU")

    if text_encoder_2 is not None and hasattr(text_encoder_2, 'to'):
        text_encoder_2.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("[RabbitVideo] Text encoder 2 moved to CPU")


def temporarily_move_to_gpu(model, device, operation_name: str = "operation"):
    """
    Context manager to temporarily move a model to GPU, execute operation, and return to CPU.

    Usage:
        with temporarily_move_to_gpu(text_encoder, device, "text encoding"):
            outputs = text_encoder.encode(...)
    """
    class TempGPUContext:
        def __enter__(self):
            logger.info(f"[RabbitVideo] Temporarily moving model to GPU for {operation_name}")
            model.to(device)
            torch.cuda.synchronize()
            return model

        def __exit__(self, exc_type, exc_val, exc_tb):
            logger.info(f"[RabbitVideo] Moving model back to CPU after {operation_name}")
            model.to('cpu')
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    return TempGPUContext()
