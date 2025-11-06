#!/usr/bin/env python3
"""
Memory Profiling Visualization for Research Papers

Generates publication-quality figures from memory profiling data.
Usage:
    python visualize_memory.py memory_logs/memory_profile_*.json --output figures/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Color scheme for consistency
COLORS = {
    'allocated': '#2E86AB',
    'reserved': '#A23B72',
    'cached': '#F18F01',
    'peak': '#C73E1D',
    'parameter': '#06A77D',
    'activation': '#D90368',
    'free': '#CCCCCC'
}


class MemoryVisualizer:
    """Generate publication-quality memory profiling visualizations."""

    def __init__(self, profile_path: str, output_dir: str = 'figures'):
        self.profile_path = Path(profile_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load profile data
        with open(self.profile_path) as f:
            self.data = json.load(f)

        self.summary = self.data.get('summary', {})
        self.analysis = self.data.get('analysis', {})
        self.timeline = self.data.get('timeline', [])
        self.component_memory = self.data.get('component_memory', {})

    def generate_all_figures(self):
        """Generate all visualization figures."""
        print("Generating memory profiling visualizations...")

        # Figure 1: Memory timeline overview
        self.plot_memory_timeline()

        # Figure 2: Component memory breakdown
        self.plot_component_breakdown()

        # Figure 3: Denoising memory progression
        self.plot_denoising_memory()

        # Figure 4: Memory phases
        self.plot_phase_memory()

        # Figure 5: Allocated vs Reserved comparison
        self.plot_allocated_vs_reserved()

        # Figure 6: Combined multi-panel figure for paper
        self.plot_combined_figure()

        print(f"\nAll figures saved to: {self.output_dir}")

    def plot_memory_timeline(self):
        """Plot complete memory timeline with allocated and reserved."""
        fig, ax = plt.subplots(figsize=(10, 4))

        # Extract timeline data
        times = []
        allocated = []
        reserved = []

        for entry in self.timeline:
            if 'memory_gb' in entry:  # Background monitoring data
                times.append(entry['timestamp'])
                allocated.append(entry['memory_gb'])
                if 'reserved_gb' in entry:
                    reserved.append(entry['reserved_gb'])
                else:
                    reserved.append(entry['memory_gb'])

        if not times:
            print("Warning: No timeline data available")
            return

        times = np.array(times)
        allocated = np.array(allocated)
        reserved = np.array(reserved)

        # Plot
        ax.plot(times / 60, allocated, label='Allocated',
                color=COLORS['allocated'], linewidth=2)
        ax.plot(times / 60, reserved, label='Reserved',
                color=COLORS['reserved'], linewidth=2, linestyle='--')
        ax.fill_between(times / 60, allocated, reserved,
                        alpha=0.3, color=COLORS['cached'],
                        label='Cached (Reserved - Allocated)')

        # Mark peak
        peak_idx = np.argmax(reserved)
        ax.plot(times[peak_idx] / 60, reserved[peak_idx],
                'r*', markersize=15, label=f'Peak: {reserved[peak_idx]:.1f} GB')

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('GPU Memory (GB)')
        ax.set_title('GPU Memory Usage Timeline')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add annotations for high cache
        if len(reserved) > 0:
            avg_cache = np.mean(reserved - allocated)
            if avg_cache > 10:
                ax.axhline(y=np.mean(allocated), color='gray',
                          linestyle=':', alpha=0.5)
                ax.text(times[-1] / 60 * 0.7, np.mean(allocated) * 1.1,
                       f'Avg Cache: {avg_cache:.1f} GB',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_timeline.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'memory_timeline.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: memory_timeline.pdf/png")

    def plot_component_breakdown(self):
        """Plot memory breakdown by component."""
        component_stats = self.summary.get('component_stats', {})
        if not component_stats:
            print("Warning: No component stats available")
            return

        # Extract data
        components = []
        memory_deltas = []

        for comp, stats in component_stats.items():
            components.append(comp.replace('_', ' ').title())
            memory_deltas.append(stats.get('max_delta_gb', 0))

        # Sort by memory usage
        sorted_idx = np.argsort(memory_deltas)[::-1]
        components = [components[i] for i in sorted_idx[:10]]  # Top 10
        memory_deltas = [memory_deltas[i] for i in sorted_idx[:10]]

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.barh(components, memory_deltas, color=COLORS['parameter'])

        # Add value labels
        for i, (comp, mem) in enumerate(zip(components, memory_deltas)):
            ax.text(mem + 0.2, i, f'{mem:.1f} GB',
                   va='center', fontsize=9)

        ax.set_xlabel('Memory Usage (GB)')
        ax.set_title('Memory Usage by Component')
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_breakdown.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'component_breakdown.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: component_breakdown.pdf/png")

    def plot_denoising_memory(self):
        """Plot memory usage during denoising steps."""
        # Extract denoising steps from timeline
        steps = []
        memory = []
        reserved_mem = []

        for entry in self.timeline:
            if entry.get('phase', '').startswith('denoising_step_'):
                step_num = entry.get('step', -1)
                if step_num >= 0:
                    steps.append(step_num)
                    mem_info = entry.get('memory_info', {})
                    memory.append(mem_info.get('allocated_gb', 0))
                    reserved_mem.append(mem_info.get('reserved_gb', 0))

        if not steps:
            print("Warning: No denoising step data available")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                       sharex=True, height_ratios=[3, 1])

        # Top panel: Memory usage
        ax1.plot(steps, memory, 'o-', label='Allocated',
                color=COLORS['allocated'], linewidth=2, markersize=4)
        ax1.plot(steps, reserved_mem, 's--', label='Reserved',
                color=COLORS['reserved'], linewidth=2, markersize=4)

        ax1.set_ylabel('GPU Memory (GB)')
        ax1.set_title('Memory Usage During Denoising Loop')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Bottom panel: Cache size
        cache = np.array(reserved_mem) - np.array(memory)
        ax2.fill_between(steps, cache, color=COLORS['cached'], alpha=0.6)
        ax2.plot(steps, cache, color=COLORS['cached'], linewidth=2)
        ax2.set_xlabel('Denoising Step')
        ax2.set_ylabel('Cache (GB)')
        ax2.grid(True, alpha=0.3)

        # Highlight if cache is excessive
        if np.mean(cache) > 10:
            ax2.axhline(y=10, color='red', linestyle=':', alpha=0.5)
            ax2.text(steps[-1] * 0.7, 10 * 1.1,
                    'High Cache Threshold',
                    color='red', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'denoising_memory.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'denoising_memory.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: denoising_memory.pdf/png")

    def plot_phase_memory(self):
        """Plot memory usage across different phases."""
        # Extract phase information
        phases_data = {}

        for entry in self.timeline:
            phase = entry.get('phase', 'unknown')
            if 'memory_gb' in entry or 'memory_info' in entry:
                if phase not in phases_data:
                    phases_data[phase] = []

                mem = entry.get('memory_gb') or entry.get('memory_info', {}).get('allocated_gb', 0)
                phases_data[phase].append(mem)

        # Calculate statistics per phase
        phases = []
        mean_memory = []
        max_memory = []

        phase_order = [
            'initialization',
            'loading_transformer',
            'loading_vae',
            'loading_text_encoder',
            'models_loaded_complete',
            'denoising_loop',
            'after_denoising',
            'vae_decode',
            'pipeline_complete'
        ]

        for phase in phase_order:
            if phase in phases_data and phases_data[phase]:
                phases.append(phase.replace('_', ' ').title())
                mean_memory.append(np.mean(phases_data[phase]))
                max_memory.append(np.max(phases_data[phase]))

        if not phases:
            print("Warning: No phase data available")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(phases))
        width = 0.35

        bars1 = ax.bar(x - width/2, mean_memory, width,
                      label='Average', color=COLORS['allocated'])
        bars2 = ax.bar(x + width/2, max_memory, width,
                      label='Peak', color=COLORS['peak'])

        ax.set_xlabel('Processing Phase')
        ax.set_ylabel('GPU Memory (GB)')
        ax.set_title('Memory Usage by Processing Phase')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase_memory.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'phase_memory.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: phase_memory.pdf/png")

    def plot_allocated_vs_reserved(self):
        """Plot allocated vs reserved memory comparison."""
        # Calculate statistics
        times = []
        allocated = []
        reserved = []

        for entry in self.timeline:
            if 'memory_gb' in entry:
                times.append(entry['timestamp'])
                allocated.append(entry['memory_gb'])
                reserved.append(entry.get('reserved_gb', entry['memory_gb']))

        if not times:
            print("Warning: No data for allocated vs reserved plot")
            return

        allocated = np.array(allocated)
        reserved = np.array(reserved)
        cache = reserved - allocated

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Top-left: Scatter plot
        ax1.scatter(allocated, reserved, alpha=0.3, s=10, color=COLORS['allocated'])
        ax1.plot([0, max(allocated)], [0, max(allocated)],
                'r--', label='Allocated = Reserved', linewidth=1)
        ax1.set_xlabel('Allocated Memory (GB)')
        ax1.set_ylabel('Reserved Memory (GB)')
        ax1.set_title('Allocated vs Reserved Memory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Top-right: Cache distribution
        ax2.hist(cache, bins=50, color=COLORS['cached'], alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(cache), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(cache):.1f} GB')
        ax2.set_xlabel('Cached Memory (GB)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Cached Memory')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Bottom-left: Time series of efficiency
        efficiency = (allocated / np.maximum(reserved, 0.001)) * 100
        ax3.plot(np.array(times) / 60, efficiency,
                color=COLORS['parameter'], linewidth=2)
        ax3.axhline(y=80, color='green', linestyle=':', alpha=0.5, label='80% threshold')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Memory Efficiency (%)')
        ax3.set_title('Memory Allocation Efficiency Over Time')
        ax3.set_ylim([0, 105])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Bottom-right: Summary statistics
        ax4.axis('off')
        stats_text = f"""
        Memory Statistics:

        Allocated Memory:
          Mean:  {np.mean(allocated):.2f} GB
          Peak:  {np.max(allocated):.2f} GB

        Reserved Memory:
          Mean:  {np.mean(reserved):.2f} GB
          Peak:  {np.max(reserved):.2f} GB

        Cached Memory:
          Mean:  {np.mean(cache):.2f} GB
          Peak:  {np.max(cache):.2f} GB

        Efficiency:
          Mean:  {np.mean(efficiency):.1f}%
          Min:   {np.min(efficiency):.1f}%

        Wasted Memory: {np.mean(cache):.2f} GB avg
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'allocated_vs_reserved.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'allocated_vs_reserved.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: allocated_vs_reserved.pdf/png")

    def plot_combined_figure(self):
        """Create a combined multi-panel figure suitable for papers."""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Panel A: Memory timeline
        ax1 = fig.add_subplot(gs[0, :])
        times = []
        allocated = []
        reserved = []

        for entry in self.timeline:
            if 'memory_gb' in entry:
                times.append(entry['timestamp'])
                allocated.append(entry['memory_gb'])
                reserved.append(entry.get('reserved_gb', entry['memory_gb']))

        if times:
            times = np.array(times) / 60
            allocated = np.array(allocated)
            reserved = np.array(reserved)

            ax1.plot(times, allocated, label='Allocated',
                    color=COLORS['allocated'], linewidth=2)
            ax1.plot(times, reserved, label='Reserved',
                    color=COLORS['reserved'], linewidth=2, linestyle='--')
            ax1.fill_between(times, allocated, reserved,
                            alpha=0.3, color=COLORS['cached'])

            peak_idx = np.argmax(reserved)
            ax1.plot(times[peak_idx], reserved[peak_idx],
                    'r*', markersize=15, label=f'Peak: {reserved[peak_idx]:.1f} GB')

            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('GPU Memory (GB)')
            ax1.set_title('(A) GPU Memory Timeline During Video Generation')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

        # Panel B: Denoising steps
        ax2 = fig.add_subplot(gs[1, 0])
        steps = []
        step_memory = []

        for entry in self.timeline:
            if entry.get('phase', '').startswith('denoising_step_'):
                step_num = entry.get('step', -1)
                if step_num >= 0:
                    steps.append(step_num)
                    mem_info = entry.get('memory_info', {})
                    step_memory.append(mem_info.get('allocated_gb', 0))

        if steps:
            ax2.plot(steps, step_memory, 'o-',
                    color=COLORS['allocated'], linewidth=2, markersize=4)
            ax2.set_xlabel('Denoising Step')
            ax2.set_ylabel('Memory (GB)')
            ax2.set_title('(B) Memory per Denoising Step')
            ax2.grid(True, alpha=0.3)

        # Panel C: Component breakdown
        ax3 = fig.add_subplot(gs[1, 1])
        component_stats = self.summary.get('component_stats', {})

        if component_stats:
            components = []
            memory_deltas = []

            for comp, stats in list(component_stats.items())[:6]:
                components.append(comp.replace('_', ' ').title()[:20])
                memory_deltas.append(stats.get('max_delta_gb', 0))

            sorted_idx = np.argsort(memory_deltas)[::-1]
            components = [components[i] for i in sorted_idx]
            memory_deltas = [memory_deltas[i] for i in sorted_idx]

            ax3.barh(components, memory_deltas, color=COLORS['parameter'])
            ax3.set_xlabel('Memory (GB)')
            ax3.set_title('(C) Memory by Component')
            ax3.grid(True, axis='x', alpha=0.3)

        # Panel D: Memory efficiency
        ax4 = fig.add_subplot(gs[2, :])

        if times.size > 0:
            cache = reserved - allocated
            efficiency = (allocated / np.maximum(reserved, 0.001)) * 100

            ax4_twin = ax4.twinx()

            line1 = ax4.plot(times, cache, color=COLORS['cached'],
                           linewidth=2, label='Cached Memory')
            line2 = ax4_twin.plot(times, efficiency, color=COLORS['parameter'],
                                linewidth=2, linestyle='--', label='Efficiency')

            ax4.axhline(y=10, color='red', linestyle=':', alpha=0.5)
            ax4_twin.axhline(y=80, color='green', linestyle=':', alpha=0.5)

            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('Cached Memory (GB)', color=COLORS['cached'])
            ax4_twin.set_ylabel('Efficiency (%)', color=COLORS['parameter'])
            ax4.set_title('(D) Memory Efficiency and Cache Analysis')
            ax4.grid(True, alpha=0.3)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left')

        plt.savefig(self.output_dir / 'combined_figure.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'combined_figure.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: combined_figure.pdf/png (multi-panel for paper)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality memory profiling visualizations'
    )
    parser.add_argument('profile', type=str,
                       help='Path to memory profile JSON file')
    parser.add_argument('--output', '-o', type=str, default='figures',
                       help='Output directory for figures (default: figures/)')
    parser.add_argument('--format', '-f', type=str, default='both',
                       choices=['pdf', 'png', 'both'],
                       help='Output format (default: both)')

    args = parser.parse_args()

    visualizer = MemoryVisualizer(args.profile, args.output)
    visualizer.generate_all_figures()

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Output directory: {args.output}")
    print("\nGenerated figures:")
    print("  1. memory_timeline.pdf/png - Complete memory timeline")
    print("  2. component_breakdown.pdf/png - Memory by component")
    print("  3. denoising_memory.pdf/png - Denoising loop analysis")
    print("  4. phase_memory.pdf/png - Memory by processing phase")
    print("  5. allocated_vs_reserved.pdf/png - Detailed comparison")
    print("  6. combined_figure.pdf/png - Multi-panel figure for paper")
    print("="*60)


if __name__ == '__main__':
    main()
