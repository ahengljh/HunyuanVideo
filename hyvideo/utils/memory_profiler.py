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