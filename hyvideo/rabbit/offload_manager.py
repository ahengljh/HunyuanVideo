"""
RabbitVideo Offload Manager
Manages CPU-GPU memory transfers for transformer blocks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque, defaultdict
import threading
import queue
import time
import numpy as np

class OffloadManager:
    """Manages offloading of model blocks between CPU and GPU."""

    def __init__(self, target_memory_gb: float = 24.0, prefetch_blocks: int = 2,
                 aggressive_mode: bool = False):
        self.target_memory_gb = target_memory_gb
        self.prefetch_blocks = prefetch_blocks
        self.aggressive_mode = aggressive_mode

        # Block tracking
        self.blocks: Dict[int, nn.Module] = {}
        self.block_devices: Dict[int, str] = {}
        self.block_access_counts: Dict[int, int] = defaultdict(int)
        self.block_last_access: Dict[int, float] = {}
        self.block_memory_size: Dict[int, float] = {}

        # Offloading state
        self.blocks_on_gpu: Set[int] = set()
        self.blocks_on_cpu: Set[int] = set()
        self.blocks_in_transfer: Set[int] = set()

        # Prefetching
        self.prefetch_queue: deque = deque(maxlen=prefetch_blocks * 2)
        self.prefetch_thread: Optional[threading.Thread] = None
        self.prefetch_thread_queue: queue.Queue = queue.Queue()
        self.stop_prefetch = threading.Event()

        # Statistics
        self.total_transfers = 0
        self.total_transfer_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        # Memory management
        self.gpu_memory_limit = target_memory_gb * 1024**3
        self.current_gpu_memory = 0

    def initialize(self, model: nn.Module):
        """Initialize offload manager with model blocks."""
        block_idx = 0

        # Find all transformer blocks
        for name, module in model.named_modules():
            if self._is_transformer_block(module):
                self.blocks[block_idx] = module
                self.block_devices[block_idx] = str(next(module.parameters()).device)

                # Calculate block memory size
                block_memory = self._calculate_module_memory(module)
                self.block_memory_size[block_idx] = block_memory

                # Initially all blocks are on their current device
                if 'cuda' in self.block_devices[block_idx]:
                    self.blocks_on_gpu.add(block_idx)
                else:
                    self.blocks_on_cpu.add(block_idx)

                block_idx += 1

        print(f"[RabbitVideo] OffloadManager initialized with {len(self.blocks)} blocks")
        self._print_memory_distribution()

        # PROACTIVE offloading: immediately offload most blocks to prevent peak memory
        # Only keep first few blocks on GPU, offload the rest
        if self.aggressive_mode and len(self.blocks) > 0:
            blocks_to_keep = max(2, len(self.blocks) // 10)  # Keep 10% or at least 2
            print(f"[RabbitVideo] Aggressive mode: offloading {len(self.blocks) - blocks_to_keep} blocks to CPU immediately")
            self._proactive_offload(blocks_to_keep)
        elif len(self.blocks) > 4:  # Normal mode: keep 25% on GPU
            blocks_to_keep = max(4, len(self.blocks) // 4)
            print(f"[RabbitVideo] Proactive offloading: keeping {blocks_to_keep} blocks on GPU, offloading {len(self.blocks) - blocks_to_keep} to CPU")
            self._proactive_offload(blocks_to_keep)

        # Start prefetch thread
        self._start_prefetch_thread()

    def _is_transformer_block(self, module: nn.Module) -> bool:
        """Check if module is a transformer block."""
        class_name = module.__class__.__name__
        return 'MMDoubleStreamBlock' in class_name or 'MMSingleStreamBlock' in class_name

    def _proactive_offload(self, blocks_to_keep_on_gpu: int):
        """Proactively offload blocks to CPU to prevent peak memory at initialization."""
        # Keep first N blocks on GPU, offload the rest
        gpu_blocks = list(self.blocks_on_gpu)

        for block_idx in gpu_blocks[blocks_to_keep_on_gpu:]:
            print(f"[RabbitVideo] Offloading block {block_idx} to CPU...", end='', flush=True)
            success = self._move_block_to_cpu(block_idx)
            if success:
                print(" ✓")
            else:
                print(" ✗")

        print(f"[RabbitVideo] Proactive offloading complete")
        self._print_memory_distribution()
        torch.cuda.empty_cache()  # Free up memory immediately

    def _calculate_module_memory(self, module: nn.Module) -> float:
        """Calculate memory usage of a module in bytes."""
        total_memory = 0
        for param in module.parameters():
            total_memory += param.element_size() * param.numel()

        # Add buffer memory
        for buffer in module.buffers():
            total_memory += buffer.element_size() * buffer.numel()

        return total_memory

    def _print_memory_distribution(self):
        """Print current memory distribution."""
        gpu_memory = sum(self.block_memory_size[idx] for idx in self.blocks_on_gpu)
        cpu_memory = sum(self.block_memory_size[idx] for idx in self.blocks_on_cpu)

        print(f"[RabbitVideo] Memory Distribution:")
        print(f"  GPU blocks: {len(self.blocks_on_gpu)} ({gpu_memory/1024**3:.2f} GB)")
        print(f"  CPU blocks: {len(self.blocks_on_cpu)} ({cpu_memory/1024**3:.2f} GB)")

    def prepare_inference(self, num_timesteps: int):
        """Prepare for inference phase."""
        # Reset statistics
        self.block_access_counts.clear()
        self.block_last_access.clear()
        self.cache_hits = 0
        self.cache_misses = 0

        # Determine initial blocks to keep on GPU based on memory budget
        available_memory = self.gpu_memory_limit
        blocks_to_keep = set()

        # Strategy: Keep as many blocks as possible on GPU initially
        # Prioritize first blocks since they're accessed first
        total_block_memory = sum(self.block_memory_size.values())

        if total_block_memory <= available_memory:
            # All blocks can fit on GPU
            blocks_to_keep = set(self.blocks.keys())
        else:
            # Keep blocks until we reach memory limit
            # Reserve some memory for activations (30% of limit)
            usable_memory = available_memory * 0.7
            current_memory = 0

            # First, keep initial blocks (they're accessed first)
            for block_idx in range(len(self.blocks)):
                if block_idx in self.blocks:
                    block_memory = self.block_memory_size[block_idx]
                    if current_memory + block_memory <= usable_memory:
                        blocks_to_keep.add(block_idx)
                        current_memory += block_memory
                    else:
                        break

        # Move blocks to their target devices
        for block_idx in self.blocks:
            if block_idx in blocks_to_keep:
                if block_idx not in self.blocks_on_gpu:
                    self._move_block_to_gpu(block_idx)
            else:
                if block_idx not in self.blocks_on_cpu:
                    self._move_block_to_cpu(block_idx)

        print(f"[RabbitVideo] Initial distribution: {len(self.blocks_on_gpu)} blocks on GPU, {len(self.blocks_on_cpu)} blocks on CPU")

    def ensure_on_device(self, block_idx: int, device: str) -> bool:
        """Ensure a block is on the specified device."""
        if block_idx not in self.blocks:
            return False

        current_device = self.block_devices[block_idx]

        # Update access tracking
        self.block_access_counts[block_idx] += 1
        self.block_last_access[block_idx] = time.time()

        # Check if already on correct device
        if device in current_device:
            self.cache_hits += 1
            return True

        self.cache_misses += 1

        # Move block to device
        if device == 'cuda':
            return self._move_block_to_gpu(block_idx)
        else:
            return self._move_block_to_cpu(block_idx)

    def _move_block_to_gpu(self, block_idx: int) -> bool:
        """Move a block from CPU to GPU."""
        if block_idx in self.blocks_on_gpu:
            return True

        if block_idx in self.blocks_in_transfer:
            # Wait for transfer to complete
            while block_idx in self.blocks_in_transfer:
                time.sleep(0.001)
            return block_idx in self.blocks_on_gpu

        self.blocks_in_transfer.add(block_idx)

        try:
            start_time = time.time()
            block = self.blocks[block_idx]

            # Check if we need to free memory first
            block_memory = self.block_memory_size[block_idx]
            if self._get_current_gpu_memory() + block_memory > self.gpu_memory_limit:
                self._free_gpu_memory(block_memory)

            # Move block to GPU (non-blocking for better performance)
            with torch.cuda.stream(torch.cuda.Stream()):
                block.to('cuda', non_blocking=True)
            self.block_devices[block_idx] = 'cuda'

            # Update tracking
            self.blocks_on_cpu.discard(block_idx)
            self.blocks_on_gpu.add(block_idx)

            transfer_time = time.time() - start_time
            self.total_transfers += 1
            self.total_transfer_time += transfer_time

            if transfer_time > 3.0:  # Only log very slow transfers (>3s)
                print(f"[RabbitVideo] Warning: Block {block_idx} CPU->GPU transfer took {transfer_time:.3f}s")

            return True

        finally:
            self.blocks_in_transfer.discard(block_idx)

    def _move_block_to_cpu(self, block_idx: int) -> bool:
        """Move a block from GPU to CPU."""
        if block_idx in self.blocks_on_cpu:
            return True

        if block_idx in self.blocks_in_transfer:
            return False

        self.blocks_in_transfer.add(block_idx)

        try:
            start_time = time.time()
            block = self.blocks[block_idx]

            # Move block to CPU
            block.to('cpu')
            self.block_devices[block_idx] = 'cpu'

            # Update tracking
            self.blocks_on_gpu.discard(block_idx)
            self.blocks_on_cpu.add(block_idx)

            transfer_time = time.time() - start_time
            self.total_transfers += 1
            self.total_transfer_time += transfer_time

            # Clear GPU cache
            torch.cuda.empty_cache()

            return True

        finally:
            self.blocks_in_transfer.discard(block_idx)

    def _get_current_gpu_memory(self) -> float:
        """Get current GPU memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0

    def _free_gpu_memory(self, required_memory: float):
        """Free GPU memory by offloading blocks to CPU."""
        # Sort blocks by access frequency (least recently used first)
        gpu_blocks = list(self.blocks_on_gpu)
        gpu_blocks.sort(key=lambda x: (self.block_access_counts[x],
                                       self.block_last_access.get(x, 0)))

        freed_memory = 0
        for block_idx in gpu_blocks:
            if freed_memory >= required_memory:
                break

            if block_idx not in self.blocks_in_transfer:
                self._move_block_to_cpu(block_idx)
                freed_memory += self.block_memory_size[block_idx]

    def offload_cold_blocks(self, current_block_idx: int):
        """Offload blocks that are unlikely to be used soon."""
        # Only offload if we're actually running low on memory
        current_gpu_memory = self._get_current_gpu_memory()
        if current_gpu_memory < self.gpu_memory_limit * 0.8:  # Still have 20% headroom
            return

        # Determine which blocks to keep on GPU
        keep_on_gpu = set()

        # Keep current and nearby blocks (wider window)
        for offset in range(-3, self.prefetch_blocks + 3):
            idx = current_block_idx + offset
            if 0 <= idx < len(self.blocks):
                keep_on_gpu.add(idx)

        # In aggressive mode, offload more aggressively
        if self.aggressive_mode:
            max_gpu_blocks = min(15, len(self.blocks) // 3)
        else:
            max_gpu_blocks = min(30, len(self.blocks) // 2)

        # Keep frequently accessed blocks
        access_sorted = sorted(self.blocks_on_gpu,
                             key=lambda x: (self.block_access_counts[x],
                                          -abs(x - current_block_idx)),  # Prefer blocks close to current
                             reverse=True)

        for idx in access_sorted[:max_gpu_blocks]:
            keep_on_gpu.add(idx)

        # Only offload blocks that are far from current position
        blocks_to_offload = []
        for block_idx in self.blocks_on_gpu:
            if block_idx not in keep_on_gpu and abs(block_idx - current_block_idx) > 5:
                blocks_to_offload.append(block_idx)

        # Offload blocks in batches to reduce overhead
        for block_idx in blocks_to_offload[:3]:  # Offload at most 3 blocks at a time
            self._move_block_to_cpu(block_idx)

    def prefetch_block(self, block_idx: int):
        """Add block to prefetch queue."""
        if block_idx in self.blocks and block_idx not in self.blocks_on_gpu:
            self.prefetch_queue.append(block_idx)
            self.prefetch_thread_queue.put(block_idx)

    def _start_prefetch_thread(self):
        """Start background thread for prefetching."""
        self.stop_prefetch.clear()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _prefetch_worker(self):
        """Background worker for prefetching blocks."""
        while not self.stop_prefetch.is_set():
            try:
                block_idx = self.prefetch_thread_queue.get(timeout=0.1)
                if block_idx in self.blocks_on_cpu:
                    # Prefetch to GPU in background
                    self._move_block_to_gpu(block_idx)
            except queue.Empty:
                continue

    def set_aggressive_mode(self, aggressive: bool):
        """Set aggressive offloading mode."""
        self.aggressive_mode = aggressive
        if aggressive:
            print("[RabbitVideo] Aggressive offloading mode enabled")

    def emergency_offload(self):
        """Emergency offload all non-essential blocks."""
        print("[RabbitVideo] Emergency offload triggered!")

        # Keep only the bare minimum on GPU
        essential_blocks = set(list(self.blocks.keys())[:5])  # Keep first 5 blocks

        for block_idx in list(self.blocks_on_gpu):
            if block_idx not in essential_blocks:
                self._move_block_to_cpu(block_idx)

        torch.cuda.empty_cache()

    def get_statistics(self) -> Dict[str, Any]:
        """Get offloading statistics."""
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses + 1e-6)

        return {
            'total_transfers': self.total_transfers,
            'avg_transfer_time': self.total_transfer_time / max(1, self.total_transfers),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'blocks_on_gpu': len(self.blocks_on_gpu),
            'blocks_on_cpu': len(self.blocks_on_cpu),
            'gpu_memory_gb': sum(self.block_memory_size[idx]
                                for idx in self.blocks_on_gpu) / 1024**3
        }

    def cleanup(self):
        """Cleanup offload manager."""
        self.stop_prefetch.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1.0)

        # Print statistics
        stats = self.get_statistics()
        print("\n[RabbitVideo] Offload Manager Statistics:")
        print(f"  Total transfers: {stats['total_transfers']}")
        print(f"  Avg transfer time: {stats['avg_transfer_time']:.3f}s")
        print(f"  Cache hit rate: {stats['hit_rate']:.1%}")
        print(f"  Final GPU memory: {stats['gpu_memory_gb']:.2f} GB")