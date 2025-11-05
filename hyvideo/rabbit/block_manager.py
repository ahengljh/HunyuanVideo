"""
RabbitVideo Block Manager
Manages transformer block execution and memory tracking.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import time
import numpy as np

class BlockManager:
    """Manages execution and tracking of transformer blocks."""

    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks

        # Block references
        self.blocks: Dict[int, nn.Module] = {}
        self.block_types: Dict[int, str] = {}  # 'double' or 'single'

        # Execution tracking
        self.block_execution_times: Dict[int, List[float]] = defaultdict(list)
        self.block_memory_usage: Dict[int, List[float]] = defaultdict(list)
        self.block_call_counts: Dict[int, int] = defaultdict(int)

        # Hooks
        self.pre_hooks: Dict[int, List[Callable]] = defaultdict(list)
        self.post_hooks: Dict[int, List[Callable]] = defaultdict(list)
        self.forward_handles: Dict[int, Any] = {}

        # Current state
        self.current_block_idx = -1
        self.current_timestep = 0

    def initialize(self, model: nn.Module):
        """Initialize block manager with model blocks."""
        block_idx = 0

        # Find and register all transformer blocks
        for name, module in model.named_modules():
            if self._is_transformer_block(module):
                self.blocks[block_idx] = module

                # Determine block type
                if 'Double' in module.__class__.__name__:
                    self.block_types[block_idx] = 'double'
                else:
                    self.block_types[block_idx] = 'single'

                # Register hooks
                self._register_block_hooks(block_idx, module)

                block_idx += 1

        print(f"[RabbitVideo] BlockManager initialized: {len(self.blocks)} blocks")
        double_blocks = sum(1 for t in self.block_types.values() if t == 'double')
        single_blocks = len(self.blocks) - double_blocks
        print(f"  Double blocks: {double_blocks}, Single blocks: {single_blocks}")

    def _is_transformer_block(self, module: nn.Module) -> bool:
        """Check if module is a transformer block."""
        class_name = module.__class__.__name__
        return 'MMDoubleStreamBlock' in class_name or 'MMSingleStreamBlock' in class_name

    def _register_block_hooks(self, block_idx: int, module: nn.Module):
        """Register forward hooks for a block."""

        def pre_forward_hook(module, inputs):
            """Hook called before forward pass."""
            # Execute custom pre-hooks
            for hook in self.pre_hooks[block_idx]:
                hook(block_idx, inputs)

            # Start timing
            self._block_forward_start = time.time()
            self._block_start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            return inputs

        def post_forward_hook(module, inputs, outputs):
            """Hook called after forward pass."""
            # Record execution time
            exec_time = time.time() - self._block_forward_start
            self.block_execution_times[block_idx].append(exec_time)

            # Record memory usage
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_delta = end_memory - self._block_start_memory
            self.block_memory_usage[block_idx].append(memory_delta)

            # Update call count
            self.block_call_counts[block_idx] += 1

            # Execute custom post-hooks
            for hook in self.post_hooks[block_idx]:
                hook(block_idx, outputs)

            return outputs

        # Register hooks
        handle_pre = module.register_forward_pre_hook(pre_forward_hook)
        handle_post = module.register_forward_hook(post_forward_hook)
        self.forward_handles[block_idx] = (handle_pre, handle_post)

    def add_pre_hook(self, block_idx: int, hook: Callable):
        """Add a pre-forward hook for a block."""
        self.pre_hooks[block_idx].append(hook)

    def add_post_hook(self, block_idx: int, hook: Callable):
        """Add a post-forward hook for a block."""
        self.post_hooks[block_idx].append(hook)

    def on_block_start(self, block_idx: int):
        """Called when starting to execute a block."""
        self.current_block_idx = block_idx

    def on_block_end(self, block_idx: int):
        """Called when finished executing a block."""
        # Could add additional tracking here
        pass

    def get_block_statistics(self, block_idx: int) -> Dict[str, Any]:
        """Get statistics for a specific block."""
        exec_times = self.block_execution_times.get(block_idx, [])
        memory_usage = self.block_memory_usage.get(block_idx, [])

        stats = {
            'block_idx': block_idx,
            'block_type': self.block_types.get(block_idx, 'unknown'),
            'call_count': self.block_call_counts[block_idx],
            'avg_exec_time': np.mean(exec_times) if exec_times else 0,
            'max_exec_time': np.max(exec_times) if exec_times else 0,
            'total_exec_time': np.sum(exec_times) if exec_times else 0,
            'avg_memory_delta': np.mean(memory_usage) / 1024**3 if memory_usage else 0,  # GB
            'max_memory_delta': np.max(memory_usage) / 1024**3 if memory_usage else 0,  # GB
        }

        return stats

    def get_bottleneck_blocks(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Identify blocks that are performance bottlenecks."""
        bottlenecks = []

        for block_idx in range(self.total_blocks):
            stats = self.get_block_statistics(block_idx)
            if stats['call_count'] > 0:
                # Score based on execution time and memory usage
                score = stats['total_exec_time'] + stats['max_memory_delta'] * 10
                bottlenecks.append({
                    **stats,
                    'bottleneck_score': score
                })

        # Sort by bottleneck score
        bottlenecks.sort(key=lambda x: x['bottleneck_score'], reverse=True)

        return bottlenecks[:top_k]

    def should_checkpoint(self, block_idx: int) -> bool:
        """Determine if gradient checkpointing should be used for this block."""
        # Use checkpointing for blocks with high memory usage
        stats = self.get_block_statistics(block_idx)

        # Checkpoint if average memory delta > 500MB
        if stats['avg_memory_delta'] > 0.5:
            return True

        # Checkpoint middle blocks (accessed less frequently)
        if self.total_blocks > 20:
            middle_start = self.total_blocks // 3
            middle_end = 2 * self.total_blocks // 3
            if middle_start <= block_idx <= middle_end:
                return True

        return False

    def optimize_block_execution(self, block_idx: int) -> Dict[str, Any]:
        """Get optimization recommendations for a block."""
        stats = self.get_block_statistics(block_idx)
        recommendations = []

        # Check if block is a bottleneck
        if stats['avg_exec_time'] > 0.1:  # >100ms
            recommendations.append("Consider using Flash Attention")

        if stats['avg_memory_delta'] > 1.0:  # >1GB
            recommendations.append("High memory usage - consider gradient checkpointing")

        if stats['call_count'] < 5 and self.current_timestep > 10:
            recommendations.append("Rarely used block - good candidate for CPU offloading")

        return {
            'block_idx': block_idx,
            'recommendations': recommendations,
            'stats': stats
        }

    def get_memory_profile(self) -> Dict[str, Any]:
        """Get memory profile of all blocks."""
        total_memory = 0
        block_memory_map = {}

        for block_idx, block in self.blocks.items():
            block_memory = 0
            for param in block.parameters():
                block_memory += param.element_size() * param.numel()

            block_memory_mb = block_memory / (1024 * 1024)
            block_memory_map[block_idx] = block_memory_mb
            total_memory += block_memory_mb

        return {
            'total_memory_mb': total_memory,
            'total_memory_gb': total_memory / 1024,
            'block_memory_map': block_memory_map,
            'avg_block_memory_mb': total_memory / len(self.blocks) if self.blocks else 0
        }

    def print_summary(self):
        """Print summary of block execution."""
        print("\n" + "="*60)
        print("Block Manager Summary")
        print("="*60)

        # Overall statistics
        total_calls = sum(self.block_call_counts.values())
        total_time = sum(sum(times) for times in self.block_execution_times.values())

        print(f"Total blocks: {self.total_blocks}")
        print(f"Total calls: {total_calls}")
        print(f"Total execution time: {total_time:.2f}s")

        # Memory profile
        memory_profile = self.get_memory_profile()
        print(f"Total block memory: {memory_profile['total_memory_gb']:.2f} GB")

        # Bottleneck blocks
        print("\nTop 5 Bottleneck Blocks:")
        bottlenecks = self.get_bottleneck_blocks(5)
        for block in bottlenecks:
            print(f"  Block {block['block_idx']} ({block['block_type']}): "
                  f"Time={block['total_exec_time']:.2f}s, "
                  f"Memory={block['max_memory_delta']:.2f}GB")

        # Recommendations
        print("\nOptimization Recommendations:")
        for block_idx in range(min(5, self.total_blocks)):
            opt = self.optimize_block_execution(block_idx)
            if opt['recommendations']:
                print(f"  Block {block_idx}: {', '.join(opt['recommendations'])}")

        print("="*60)

    def cleanup(self):
        """Cleanup block manager."""
        # Remove hooks
        for handles in self.forward_handles.values():
            for handle in handles:
                handle.remove()

        self.forward_handles.clear()

        # Print summary
        self.print_summary()