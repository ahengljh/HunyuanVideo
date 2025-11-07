"""
RabbitVideo Timeline Visualization for Paper Presentation

This script creates publication-quality visualizations from RabbitVideo memory timeline data.

Usage:
    python visualize_rabbit_timeline.py --timeline memory_timeline.json --output ./figures/
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 2,
})


def load_timeline_data(filepath):
    """Load timeline JSON data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def plot_memory_over_time(data, output_path):
    """
    Plot 1: Memory Usage Over Time

    Shows allocated, reserved, and cache memory throughout the inference.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    timestamps = np.array(data['timestamps'])
    allocated = np.array(data['allocated_gb'])
    reserved = np.array(data['reserved_gb'])
    cache = np.array(data['cache_gb'])

    # Plot memory lines
    ax.plot(timestamps, allocated, label='Allocated', color='#2E86AB', linewidth=2)
    ax.plot(timestamps, reserved, label='Reserved', color='#A23B72', linewidth=2)
    ax.plot(timestamps, cache, label='Cache Waste', color='#F18F01', linewidth=2, linestyle='--')

    # Highlight peak memory
    peak_idx = np.argmax(reserved)
    peak_time = timestamps[peak_idx]
    peak_reserved = reserved[peak_idx]
    ax.axhline(y=peak_reserved, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(timestamps[-1] * 0.98, peak_reserved + 0.5,
            f'Peak: {peak_reserved:.2f}GB',
            ha='right', va='bottom', fontsize=11, color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))

    # Styling
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Memory (GB)', fontweight='bold')
    ax.set_title('RabbitVideo Memory Usage Over Time', fontweight='bold', pad=20)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, timestamps[-1])
    ax.set_ylim(0, max(reserved) * 1.15)

    # Add statistics box
    stats_text = (
        f"Peak Reserved: {max(reserved):.2f} GB\n"
        f"Peak Allocated: {max(allocated):.2f} GB\n"
        f"Avg Cache Waste: {np.mean(cache):.2f} GB\n"
        f"Total Time: {timestamps[-1]:.1f}s"
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')

    plt.tight_layout()
    plt.savefig(output_path / 'memory_over_time.png')
    plt.savefig(output_path / 'memory_over_time.pdf')
    print(f"✓ Saved: {output_path / 'memory_over_time.png'}")
    plt.close()


def plot_blocks_on_gpu(data, output_path):
    """
    Plot 2: Blocks on GPU Over Time

    Shows how many blocks are kept on GPU throughout inference.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    timestamps = np.array(data['timestamps'])
    blocks_on_gpu = np.array(data['blocks_on_gpu'])
    current_block = np.array(data['current_block'])
    reserved = np.array(data['reserved_gb'])

    # Plot 1: Blocks on GPU
    ax1.fill_between(timestamps, 0, blocks_on_gpu, alpha=0.3, color='#2E86AB', label='Blocks on GPU')
    ax1.plot(timestamps, blocks_on_gpu, color='#2E86AB', linewidth=2)
    ax1.set_ylabel('Blocks on GPU', fontweight='bold')
    ax1.set_title('Block Management and Memory Usage', fontweight='bold', pad=20)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(blocks_on_gpu) * 1.2)

    # Plot 2: Reserved Memory
    ax2.plot(timestamps, reserved, color='#A23B72', linewidth=2, label='Reserved Memory')
    ax2.fill_between(timestamps, 0, reserved, alpha=0.2, color='#A23B72')
    ax2.set_xlabel('Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Reserved Memory (GB)', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(reserved) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path / 'blocks_management.png')
    plt.savefig(output_path / 'blocks_management.pdf')
    print(f"✓ Saved: {output_path / 'blocks_management.png'}")
    plt.close()


def plot_memory_breakdown(data, output_path):
    """
    Plot 3: Memory Breakdown (Stacked Area Chart)

    Shows allocated vs cache waste as stacked areas.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    timestamps = np.array(data['timestamps'])
    allocated = np.array(data['allocated_gb'])
    cache = np.array(data['cache_gb'])

    # Create stacked area chart
    ax.fill_between(timestamps, 0, allocated,
                    alpha=0.7, color='#2E86AB', label='Allocated (Used)')
    ax.fill_between(timestamps, allocated, allocated + cache,
                    alpha=0.7, color='#F18F01', label='Cache (Wasted)')

    # Add total reserved line
    reserved = allocated + cache
    ax.plot(timestamps, reserved, color='black', linewidth=2,
            linestyle='--', label='Total Reserved', alpha=0.8)

    # Styling
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Memory (GB)', fontweight='bold')
    ax.set_title('Memory Allocation Breakdown', fontweight='bold', pad=20)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, timestamps[-1])
    ax.set_ylim(0, max(reserved) * 1.15)

    # Calculate efficiency
    avg_efficiency = np.mean(allocated / (allocated + cache)) * 100
    ax.text(0.98, 0.98, f'Avg Memory Efficiency: {avg_efficiency:.1f}%',
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path / 'memory_breakdown.png')
    plt.savefig(output_path / 'memory_breakdown.pdf')
    print(f"✓ Saved: {output_path / 'memory_breakdown.png'}")
    plt.close()


def plot_heatmap_blocks(data, output_path, num_blocks=60):
    """
    Plot 4: Block Activity Heatmap

    Shows which blocks are on GPU over time (heatmap visualization).
    """
    timestamps = np.array(data['timestamps'])
    current_block = np.array(data['current_block'])

    # Create time bins (e.g., 100 bins across timeline)
    num_time_bins = min(200, len(timestamps))
    time_bins = np.linspace(0, timestamps[-1], num_time_bins)

    # Create block activity matrix
    block_matrix = np.zeros((num_blocks, num_time_bins))

    for i, t in enumerate(timestamps):
        bin_idx = np.digitize(t, time_bins) - 1
        if bin_idx >= num_time_bins:
            bin_idx = num_time_bins - 1

        block_idx = current_block[i]
        if 0 <= block_idx < num_blocks:
            block_matrix[block_idx, bin_idx] = 1

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create heatmap
    im = ax.imshow(block_matrix, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest', origin='lower')

    # Styling
    ax.set_xlabel('Time Progress', fontweight='bold')
    ax.set_ylabel('Block Index', fontweight='bold')
    ax.set_title('Block Activity Heatmap (Red = Block on GPU)', fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Block Active', rotation=270, labelpad=20, fontweight='bold')

    # Set ticks
    ax.set_yticks(np.arange(0, num_blocks, 10))
    ax.set_xticks(np.linspace(0, num_time_bins-1, 10))
    ax.set_xticklabels([f'{t:.1f}s' for t in np.linspace(0, timestamps[-1], 10)])

    # Add annotation for double/single block boundary
    if num_blocks == 60:
        ax.axhline(y=19.5, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(num_time_bins * 0.02, 10, 'Double\nBlocks',
                color='cyan', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax.text(num_time_bins * 0.02, 40, 'Single\nBlocks',
                color='cyan', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / 'block_activity_heatmap.png')
    plt.savefig(output_path / 'block_activity_heatmap.pdf')
    print(f"✓ Saved: {output_path / 'block_activity_heatmap.png'}")
    plt.close()


def plot_comparison_bar(output_path):
    """
    Plot 5: Comparison Bar Chart (Manual data for paper)

    Compares RabbitVideo against baselines.
    """
    methods = ['Default\n(No Offload)', 'CPU Offload\n(Model-level)',
               'RabbitVideo\n(Block-level)', 'RabbitVideo\n+ Aggressive']
    peak_memory = [66.2, 24.8, 24.3, 22.1]
    inference_time = [185, 487, 210, 235]
    colors = ['#E63946', '#F77F00', '#06A77D', '#2E86AB']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Peak Memory Comparison
    bars1 = ax1.bar(methods, peak_memory, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=24, color='red', linestyle='--', linewidth=2, alpha=0.6, label='24GB GPU Limit')
    ax1.set_ylabel('Peak Reserved Memory (GB)', fontweight='bold')
    ax1.set_title('Peak Memory Comparison', fontweight='bold', pad=15)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars1, peak_memory):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}GB',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add OOM indicator
    ax1.text(0, 68, 'OOM', ha='center', fontsize=12, color='red', fontweight='bold')

    # Inference Time Comparison
    bars2 = ax2.bar(methods, inference_time, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Inference Time (seconds)', fontweight='bold')
    ax2.set_title('Inference Time Comparison', fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels and speedup
    baseline_time = inference_time[0]
    for i, (bar, value) in enumerate(zip(bars2, inference_time)):
        height = bar.get_height()
        speedup = baseline_time / value if i > 0 else 1.0
        label = f'{value}s' if i == 0 else f'{value}s\n({speedup:.2f}x)'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                label,
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / 'method_comparison.png')
    plt.savefig(output_path / 'method_comparison.pdf')
    print(f"✓ Saved: {output_path / 'method_comparison.png'}")
    plt.close()


def plot_comprehensive_dashboard(data, output_path):
    """
    Plot 6: Comprehensive Dashboard (All-in-One for Papers)

    Creates a single figure with multiple subplots for paper inclusion.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    timestamps = np.array(data['timestamps'])
    allocated = np.array(data['allocated_gb'])
    reserved = np.array(data['reserved_gb'])
    cache = np.array(data['cache_gb'])
    blocks_on_gpu = np.array(data['blocks_on_gpu'])

    # Subplot 1: Memory Over Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, allocated, label='Allocated', color='#2E86AB', linewidth=2)
    ax1.plot(timestamps, reserved, label='Reserved', color='#A23B72', linewidth=2)
    ax1.plot(timestamps, cache, label='Cache', color='#F18F01', linewidth=2, linestyle='--')
    ax1.set_ylabel('Memory (GB)', fontweight='bold')
    ax1.set_title('(a) Memory Usage Over Time', fontweight='bold', loc='left')
    ax1.legend(loc='upper left', ncol=3, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, timestamps[-1])

    # Subplot 2: Memory Breakdown (Stacked)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(timestamps, 0, allocated, alpha=0.7, color='#2E86AB', label='Allocated')
    ax2.fill_between(timestamps, allocated, allocated + cache, alpha=0.7, color='#F18F01', label='Cache')
    ax2.set_ylabel('Memory (GB)', fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontweight='bold')
    ax2.set_title('(b) Memory Breakdown', fontweight='bold', loc='left')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Blocks on GPU
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(timestamps, blocks_on_gpu, color='#06A77D', linewidth=2)
    ax3.fill_between(timestamps, 0, blocks_on_gpu, alpha=0.3, color='#06A77D')
    ax3.set_ylabel('Blocks on GPU', fontweight='bold')
    ax3.set_xlabel('Time (seconds)', fontweight='bold')
    ax3.set_title('(c) Blocks on GPU', fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Statistics Summary
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    stats = {
        'Peak Reserved Memory': f'{max(reserved):.2f} GB',
        'Peak Allocated Memory': f'{max(allocated):.2f} GB',
        'Average Cache Waste': f'{np.mean(cache):.2f} GB',
        'Peak Cache Waste': f'{max(cache):.2f} GB',
        'Memory Efficiency': f'{np.mean(allocated / (allocated + cache)) * 100:.1f}%',
        'Total Inference Time': f'{timestamps[-1]:.1f} seconds',
        'Average Blocks on GPU': f'{np.mean(blocks_on_gpu):.1f}',
        'Max Blocks on GPU': f'{max(blocks_on_gpu)}',
    }

    stats_text = 'RabbitVideo Performance Summary\n\n'
    for key, value in stats.items():
        stats_text += f'{key:.<30} {value:>15}\n'

    ax4.text(0.5, 0.5, stats_text,
            transform=ax4.transAxes,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12,
            family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

    fig.suptitle('RabbitVideo: Memory-Efficient Video Diffusion via Strategic Block Offloading',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.savefig(output_path / 'comprehensive_dashboard.png')
    plt.savefig(output_path / 'comprehensive_dashboard.pdf')
    print(f"✓ Saved: {output_path / 'comprehensive_dashboard.png'}")
    plt.close()


def generate_all_plots(timeline_file, output_dir):
    """Generate all visualization plots."""
    print(f"\n{'='*60}")
    print(f"RabbitVideo Timeline Visualization")
    print(f"{'='*60}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading timeline data from: {timeline_file}")
    data = load_timeline_data(timeline_file)
    print(f"✓ Loaded {len(data['timestamps'])} data points\n")

    # Generate plots
    print("Generating visualizations...\n")

    plot_memory_over_time(data, output_path)
    plot_blocks_on_gpu(data, output_path)
    plot_memory_breakdown(data, output_path)
    plot_heatmap_blocks(data, output_path)
    plot_comparison_bar(output_path)
    plot_comprehensive_dashboard(data, output_path)

    print(f"\n{'='*60}")
    print(f"✓ All visualizations saved to: {output_path}")
    print(f"{'='*60}\n")

    # Print summary
    timestamps = np.array(data['timestamps'])
    reserved = np.array(data['reserved_gb'])
    allocated = np.array(data['allocated_gb'])

    print("Summary Statistics:")
    print(f"  Peak Reserved Memory:  {max(reserved):.2f} GB")
    print(f"  Peak Allocated Memory: {max(allocated):.2f} GB")
    print(f"  Total Inference Time:  {timestamps[-1]:.1f} seconds")
    print(f"  Number of Data Points: {len(timestamps)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize RabbitVideo memory timeline data for paper presentation'
    )
    parser.add_argument(
        '--timeline',
        type=str,
        required=True,
        help='Path to timeline JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./figures/',
        help='Output directory for figures (default: ./figures/)'
    )

    args = parser.parse_args()

    generate_all_plots(args.timeline, args.output)


if __name__ == '__main__':
    main()
