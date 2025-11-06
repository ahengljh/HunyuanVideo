"""
RabbitVideo Memory Profiler
Comprehensive GPU memory tracking and profiling for video generation optimization.
"""

import torch
import gc
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime
import json
import os
import threading
from contextlib import contextmanager

class MemoryProfiler:
    """Advanced GPU memory profiler for tracking component-level memory usage."""

    def __init__(self, enable_profiling: bool = True, log_dir: str = "./memory_logs"):
        self.enable_profiling = enable_profiling
        self.log_dir = log_dir
        self.memory_timeline = []
        self.component_memory = defaultdict(list)
        self.peak_memory = {}
        self.current_phase = "initialization"
        self.start_time = time.time()

        # Create log directory
        if enable_profiling:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"memory_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Memory thresholds for warnings
        self.warning_threshold_gb = 20.0
        self.critical_threshold_gb = 23.0

        # Component tracking
        self.active_components = set()
        self.component_start_memory = {}

        # Real-time monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 0.1  # seconds

    def start_monitoring(self):
        """Start real-time memory monitoring in background thread."""
        if not self.enable_profiling:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop real-time memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Background thread for continuous memory monitoring."""
        while self.monitoring:
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024**3
                self.memory_timeline.append({
                    'timestamp': time.time() - self.start_time,
                    'memory_gb': memory_gb,
                    'phase': self.current_phase
                })
            time.sleep(self.monitor_interval)

    @contextmanager
    def track_component(self, component_name: str, log_details: bool = True):
        """Context manager to track memory usage of a specific component."""
        if not self.enable_profiling:
            yield
            return

        # Force garbage collection before measurement
        gc.collect()
        torch.cuda.empty_cache()

        # Record start memory
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        start_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        start_time = time.time()

        self.active_components.add(component_name)
        self.component_start_memory[component_name] = start_memory

        if log_details:
            print(f"[RabbitVideo] Entering {component_name} - Memory: {start_memory/1024**3:.2f}GB")

        try:
            yield
        finally:
            # Record end memory
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            end_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            end_time = time.time()

            memory_delta = end_memory - start_memory
            reserved_delta = end_reserved - start_reserved
            duration = end_time - start_time

            # Track component memory
            component_data = {
                'component': component_name,
                'start_gb': start_memory / 1024**3,
                'end_gb': end_memory / 1024**3,
                'delta_gb': memory_delta / 1024**3,
                'reserved_delta_gb': reserved_delta / 1024**3,
                'duration': duration,
                'timestamp': time.time() - self.start_time
            }

            self.component_memory[component_name].append(component_data)

            # Update peak memory
            if component_name not in self.peak_memory or end_memory > self.peak_memory[component_name]:
                self.peak_memory[component_name] = end_memory

            # Check thresholds
            memory_gb = end_memory / 1024**3
            if memory_gb > self.critical_threshold_gb:
                print(f"[RabbitVideo] ⚠️ CRITICAL: {component_name} using {memory_gb:.2f}GB (>{self.critical_threshold_gb}GB)")
            elif memory_gb > self.warning_threshold_gb:
                print(f"[RabbitVideo] ⚠️ WARNING: {component_name} using {memory_gb:.2f}GB (>{self.warning_threshold_gb}GB)")

            if log_details:
                print(f"[RabbitVideo] Exiting {component_name} - Memory: {end_memory/1024**3:.2f}GB (Δ{memory_delta/1024**3:+.2f}GB) Time: {duration:.2f}s")

            self.active_components.discard(component_name)

    def set_phase(self, phase: str):
        """Set the current processing phase for tracking."""
        self.current_phase = phase
        if self.enable_profiling:
            print(f"[RabbitVideo] Phase: {phase}")

    def get_current_memory_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        summary = {
            'current_memory_gb': self.get_current_memory_gb(),
            'peak_memory_gb': max([m/1024**3 for m in self.peak_memory.values()]) if self.peak_memory else 0,
            'component_peaks': {k: v/1024**3 for k, v in self.peak_memory.items()},
            'total_duration': time.time() - self.start_time,
            'active_components': list(self.active_components)
        }

        # Add component statistics
        component_stats = {}
        for component, measurements in self.component_memory.items():
            if measurements:
                deltas = [m['delta_gb'] for m in measurements]
                durations = [m['duration'] for m in measurements]
                component_stats[component] = {
                    'calls': len(measurements),
                    'avg_delta_gb': np.mean(deltas),
                    'max_delta_gb': np.max(deltas),
                    'total_time': np.sum(durations),
                    'avg_time': np.mean(durations)
                }
        summary['component_stats'] = component_stats

        return summary

    def analyze_memory_pattern(self) -> Dict[str, Any]:
        """Analyze memory usage patterns to identify optimization opportunities."""
        if not self.memory_timeline:
            return {}

        timeline_array = np.array([(t['timestamp'], t['memory_gb']) for t in self.memory_timeline])

        analysis = {
            'peak_memory_gb': np.max(timeline_array[:, 1]),
            'avg_memory_gb': np.mean(timeline_array[:, 1]),
            'min_memory_gb': np.min(timeline_array[:, 1]),
            'std_memory_gb': np.std(timeline_array[:, 1]),
            'total_duration': timeline_array[-1, 0] if len(timeline_array) > 0 else 0
        }

        # Identify memory spikes
        memory_values = timeline_array[:, 1]
        mean_memory = np.mean(memory_values)
        std_memory = np.std(memory_values)
        spike_threshold = mean_memory + 2 * std_memory

        spikes = []
        for i, (timestamp, memory) in enumerate(timeline_array):
            if memory > spike_threshold:
                phase = self.memory_timeline[i]['phase']
                spikes.append({
                    'timestamp': timestamp,
                    'memory_gb': memory,
                    'phase': phase,
                    'spike_size_gb': memory - mean_memory
                })

        analysis['memory_spikes'] = spikes

        # Identify components that could be offloaded
        offload_candidates = []
        for component, stats in self.get_memory_summary()['component_stats'].items():
            if stats['max_delta_gb'] > 2.0:  # Components using >2GB
                offload_candidates.append({
                    'component': component,
                    'max_memory_gb': stats['max_delta_gb'],
                    'avg_memory_gb': stats['avg_delta_gb'],
                    'potential_savings_gb': stats['max_delta_gb'] * 0.8  # Assume 80% can be offloaded
                })

        analysis['offload_candidates'] = sorted(offload_candidates,
                                                key=lambda x: x['potential_savings_gb'],
                                                reverse=True)

        return analysis

    def save_profile(self):
        """Save profiling data to disk."""
        if not self.enable_profiling:
            return

        profile_data = {
            'summary': self.get_memory_summary(),
            'analysis': self.analyze_memory_pattern(),
            'timeline': self.memory_timeline,
            'component_memory': {k: list(v) for k, v in self.component_memory.items()},
            'peak_memory': {k: v/1024**3 for k, v in self.peak_memory.items()}
        }

        with open(self.log_file, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)

        print(f"[RabbitVideo] Memory profile saved to {self.log_file}")

        # Print summary
        summary = profile_data['summary']
        analysis = profile_data['analysis']

        print("\n" + "="*60)
        print("RabbitVideo Memory Profile Summary")
        print("="*60)
        print(f"Peak Memory: {summary['peak_memory_gb']:.2f} GB")
        print(f"Current Memory: {summary['current_memory_gb']:.2f} GB")
        print(f"Total Duration: {summary['total_duration']:.2f} seconds")

        if analysis.get('offload_candidates'):
            print("\nTop Offloading Candidates:")
            for candidate in analysis['offload_candidates'][:5]:
                print(f"  - {candidate['component']}: {candidate['potential_savings_gb']:.2f} GB potential savings")

        print("="*60)

    def log_tensor_info(self, name: str, tensor: torch.Tensor):
        """Log information about a tensor including memory usage."""
        if not self.enable_profiling or tensor is None:
            return

        memory_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)
        print(f"[RabbitVideo] Tensor '{name}': shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, memory={memory_mb:.2f}MB")

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get current GPU memory information."""
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3,
            'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }

    def log_allocated_tensors(self, tag: str = "snapshot", top_n: int = 20):
        """Log all allocated tensors on GPU with detailed information."""
        if not self.enable_profiling or not torch.cuda.is_available():
            return

        print(f"\n{'='*80}")
        print(f"[RabbitVideo] GPU Memory Snapshot: {tag}")
        print(f"Time: {time.time() - self.start_time:.2f}s | Phase: {self.current_phase}")
        print(f"{'='*80}")

        # Get GPU memory info
        mem_info = self.get_gpu_memory_info()
        print(f"Total GPU Memory: {mem_info.get('total_gb', 0):.2f} GB")
        print(f"Allocated: {mem_info.get('allocated_gb', 0):.2f} GB  (memory actively used by tensors)")
        print(f"Reserved: {mem_info.get('reserved_gb', 0):.2f} GB  (memory reserved from CUDA)")
        print(f"Cached: {(mem_info.get('reserved_gb', 0) - mem_info.get('allocated_gb', 0)):.2f} GB  (reserved but not in use, for fast reallocation)")
        print(f"Free: {mem_info.get('free_gb', 0):.2f} GB")
        print(f"{'-'*80}")

        # Collect all tensors
        tensor_info = []
        param_memory_mb = 0
        activation_memory_mb = 0

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        memory_mb = obj.element_size() * obj.numel() / (1024 * 1024)
                        tensor_info.append({
                            'shape': tuple(obj.shape),
                            'dtype': str(obj.dtype),
                            'memory_mb': memory_mb,
                            'device': str(obj.device),
                            'requires_grad': obj.requires_grad
                        })

                        # Categorize tensors
                        if obj.requires_grad:
                            param_memory_mb += memory_mb
                        else:
                            activation_memory_mb += memory_mb
            except Exception:
                pass

        # Sort by memory usage
        tensor_info.sort(key=lambda x: x['memory_mb'], reverse=True)

        # Calculate total memory from tracked tensors
        total_tracked_mb = sum(t['memory_mb'] for t in tensor_info)
        allocated_gb = mem_info.get('allocated_gb', 0)
        tracked_gb = total_tracked_mb / 1024
        untracked_gb = allocated_gb - tracked_gb

        # Log memory breakdown
        print(f"Memory Breakdown:")
        print(f"  Total Tracked Tensors: {len(tensor_info):,}")
        print(f"  Total Tracked Memory: {tracked_gb:.2f} GB ({tracked_gb/allocated_gb*100:.1f}% of allocated)")
        print(f"  Parameters (requires_grad=True): {param_memory_mb/1024:.2f} GB ({len([t for t in tensor_info if t['requires_grad']]):,} tensors)")
        print(f"  Activations (requires_grad=False): {activation_memory_mb/1024:.2f} GB ({len([t for t in tensor_info if not t['requires_grad']]):,} tensors)")
        if untracked_gb > 0.1:  # Only show if significant
            print(f"  Untracked/Overhead: {untracked_gb:.2f} GB (CUDA allocator overhead, fragmentation)")
        print(f"{'-'*80}")

        # Log top N tensors
        print(f"Top {top_n} Tensors by Memory Usage:")
        print(f"{'Shape':<30} {'DType':<15} {'Memory (MB)':<15} {'Grad':<8} {'Device'}")
        print(f"{'-'*80}")

        top_n_memory = 0
        for i, info in enumerate(tensor_info[:top_n]):
            top_n_memory += info['memory_mb']
            shape_str = str(info['shape'])[:28]
            dtype_str = info['dtype'].replace('torch.', '')[:13]
            print(f"{shape_str:<30} {dtype_str:<15} {info['memory_mb']:>10.2f} MB   {str(info['requires_grad']):<8} {info['device']}")

        print(f"{'-'*80}")
        remaining_tensors = len(tensor_info) - top_n
        remaining_memory_gb = (total_tracked_mb - top_n_memory) / 1024
        print(f"Top {top_n} tensors: {top_n_memory/1024:.2f} GB ({top_n_memory/total_tracked_mb*100:.1f}% of tracked)")
        print(f"Remaining {remaining_tensors:,} tensors: {remaining_memory_gb:.2f} GB ({remaining_memory_gb/tracked_gb*100:.1f}% of tracked)")
        print(f"{'='*80}\n")

        # Save snapshot to timeline
        snapshot_data = {
            'timestamp': time.time() - self.start_time,
            'tag': tag,
            'phase': self.current_phase,
            'memory_info': mem_info,
            'top_tensors': tensor_info[:top_n],
            'total_tensors': len(tensor_info)
        }

        # Store in memory timeline with special marker
        self.memory_timeline.append(snapshot_data)

    def log_model_memory(self, model, model_name: str):
        """Log memory usage of a model and its parameters."""
        if not self.enable_profiling or model is None:
            return

        print(f"\n[RabbitVideo] Model Memory: {model_name}")
        print(f"{'-'*60}")

        total_params = 0
        total_memory_mb = 0
        param_memory = {}

        for name, param in model.named_parameters():
            if param is not None:
                num_params = param.numel()
                memory_mb = param.element_size() * num_params / (1024 * 1024)
                total_params += num_params
                total_memory_mb += memory_mb

                # Group by module
                module_name = name.split('.')[0] if '.' in name else name
                if module_name not in param_memory:
                    param_memory[module_name] = {'params': 0, 'memory_mb': 0}
                param_memory[module_name]['params'] += num_params
                param_memory[module_name]['memory_mb'] += memory_mb

        # Print by module
        print(f"{'Module':<30} {'Parameters':<20} {'Memory (MB)'}")
        print(f"{'-'*60}")
        for module_name, stats in sorted(param_memory.items(), key=lambda x: x[1]['memory_mb'], reverse=True)[:10]:
            print(f"{module_name[:28]:<30} {stats['params']:>15,}      {stats['memory_mb']:>10.2f}")

        print(f"{'-'*60}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Total Model Memory: {total_memory_mb:.2f} MB ({total_memory_mb/1024:.2f} GB)")
        print(f"{'-'*60}\n")

    def log_timestep_memory(self, step: int, total_steps: int, extra_info: dict = None):
        """Log memory at each denoising timestep with detailed breakdown."""
        if not self.enable_profiling:
            return

        mem_info = self.get_gpu_memory_info()
        elapsed = time.time() - self.start_time

        # Create compact log entry
        log_msg = f"[RabbitVideo] Step {step:3d}/{total_steps} | "
        log_msg += f"Mem: {mem_info.get('allocated_gb', 0):5.2f}GB | "
        log_msg += f"Reserved: {mem_info.get('reserved_gb', 0):5.2f}GB | "
        log_msg += f"Time: {elapsed:6.1f}s"

        if extra_info:
            for key, value in extra_info.items():
                log_msg += f" | {key}: {value}"

        print(log_msg)

        # Store detailed timestep info
        timestep_data = {
            'timestamp': elapsed,
            'step': step,
            'total_steps': total_steps,
            'phase': f"denoising_step_{step}",
            'memory_info': mem_info,
            'extra_info': extra_info or {}
        }
        self.memory_timeline.append(timestep_data)

    def print_memory_explanation(self):
        """Print detailed explanation of PyTorch memory management."""
        if not self.enable_profiling or not torch.cuda.is_available():
            return

        print(f"\n{'='*80}")
        print("[RabbitVideo] PyTorch Memory Management Explanation")
        print(f"{'='*80}")

        mem_info = self.get_gpu_memory_info()
        allocated = mem_info.get('allocated_gb', 0)
        reserved = mem_info.get('reserved_gb', 0)
        cached = reserved - allocated
        total = mem_info.get('total_gb', 0)
        free = mem_info.get('free_gb', 0)

        print("\nMemory Hierarchy:")
        print(f"  1. Total GPU Memory:     {total:6.2f} GB  (Physical VRAM on your GPU)")
        print(f"  2. Reserved by PyTorch:  {reserved:6.2f} GB  (Requested from CUDA driver)")
        print(f"     ├─ Allocated:         {allocated:6.2f} GB  (Actually used by tensors)")
        print(f"     └─ Cached:            {cached:6.2f} GB  (Reserved but free, for fast reuse)")
        print(f"  3. Free GPU Memory:      {free:6.2f} GB  (Not reserved by PyTorch)")

        print("\nWho marks memory as 'used'?")
        print("  • PyTorch Caching Allocator: Reserves large chunks from CUDA")
        print("  • Tensor Creation: Allocates from reserved memory pool")
        print("  • Model Parameters: Weights marked as allocated when model loads")
        print("  • Activations: Intermediate tensors during forward/backward pass")

        print("\nWhy is 'Allocated' different from sum of visible tensors?")
        print("  • Many tensors exist beyond the top 20 shown")
        print("  • Model parameters (weights/biases) across all layers")
        print("  • Optimizer states (if training)")
        print("  • Attention caches, embeddings, intermediate activations")
        print("  • Small overhead for CUDA memory management")

        print(f"\nCurrent Status:")
        print(f"  • Memory Utilization: {allocated/total*100:.1f}% of total GPU")
        print(f"  • Cache Efficiency: {cached/reserved*100:.1f}% of reserved is cached")
        print(f"  • Memory Pressure: {'HIGH' if allocated/total > 0.8 else 'MODERATE' if allocated/total > 0.6 else 'LOW'}")

        print(f"{'='*80}\n")

    @staticmethod
    def force_cleanup():
        """Force garbage collection and empty CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# Global profiler instance
_global_profiler: Optional[MemoryProfiler] = None

def get_memory_profiler() -> Optional[MemoryProfiler]:
    """Get the global memory profiler instance."""
    return _global_profiler

def set_memory_profiler(profiler: MemoryProfiler):
    """Set the global memory profiler instance."""
    global _global_profiler
    _global_profiler = profiler

@contextmanager
def profile_memory(component_name: str, create_if_none: bool = True):
    """Convenience context manager for memory profiling."""
    profiler = get_memory_profiler()
    if profiler is None and create_if_none:
        profiler = MemoryProfiler(enable_profiling=True)
        set_memory_profiler(profiler)

    if profiler:
        with profiler.track_component(component_name):
            yield profiler
    else:
        yield None