#!/usr/bin/env python3
"""
Compare Memory Profiles - Before and After Optimization

Generates comparison figures showing the impact of memory optimizations.
Usage:
    python compare_memory_profiles.py baseline.json optimized.json --output comparison/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

COLORS = {
    'baseline': '#C73E1D',
    'optimized': '#06A77D',
    'improvement': '#2E86AB'
}


class ProfileComparator:
    """Compare two memory profiles."""

    def __init__(self, baseline_path: str, optimized_path: str, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load both profiles
        with open(baseline_path) as f:
            self.baseline = json.load(f)
        with open(optimized_path) as f:
            self.optimized = json.load(f)

    def generate_comparison(self):
        """Generate all comparison figures."""
        print("Generating comparison visualizations...")

        self.plot_timeline_comparison()
        self.plot_peak_comparison()
        self.plot_efficiency_comparison()
        self.plot_combined_comparison()

        print(f"\nAll comparison figures saved to: {self.output_dir}")

    def extract_timeline(self, profile):
        """Extract timeline data from profile."""
        timeline = profile.get('timeline', [])
        times = []
        allocated = []
        reserved = []

        for entry in timeline:
            if 'memory_gb' in entry:
                times.append(entry['timestamp'])
                allocated.append(entry['memory_gb'])
                reserved.append(entry.get('reserved_gb', entry['memory_gb']))

        return np.array(times), np.array(allocated), np.array(reserved)

    def plot_timeline_comparison(self):
        """Compare memory timelines."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Baseline
        times_b, alloc_b, res_b = self.extract_timeline(self.baseline)
        if len(times_b) > 0:
            ax1.plot(times_b / 60, alloc_b, label='Allocated',
                    color=COLORS['baseline'], linewidth=2)
            ax1.plot(times_b / 60, res_b, label='Reserved',
                    color=COLORS['baseline'], linewidth=2, linestyle='--', alpha=0.5)
            ax1.fill_between(times_b / 60, alloc_b, res_b,
                            alpha=0.2, color=COLORS['baseline'])

            ax1.set_ylabel('Memory (GB)')
            ax1.set_title('Baseline (Without Optimization)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # Add peak annotation
            peak_idx = np.argmax(res_b)
            ax1.annotate(f'Peak: {res_b[peak_idx]:.1f} GB',
                        xy=(times_b[peak_idx] / 60, res_b[peak_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # Optimized
        times_o, alloc_o, res_o = self.extract_timeline(self.optimized)
        if len(times_o) > 0:
            ax2.plot(times_o / 60, alloc_o, label='Allocated',
                    color=COLORS['optimized'], linewidth=2)
            ax2.plot(times_o / 60, res_o, label='Reserved',
                    color=COLORS['optimized'], linewidth=2, linestyle='--', alpha=0.5)
            ax2.fill_between(times_o / 60, alloc_o, res_o,
                            alpha=0.2, color=COLORS['optimized'])

            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Memory (GB)')
            ax2.set_title('Optimized (With RabbitVideo)')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            # Add peak annotation
            peak_idx = np.argmax(res_o)
            ax2.annotate(f'Peak: {res_o[peak_idx]:.1f} GB',
                        xy=(times_o[peak_idx] / 60, res_o[peak_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'timeline_comparison.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'timeline_comparison.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: timeline_comparison.pdf/png")

    def plot_peak_comparison(self):
        """Compare peak memory usage."""
        baseline_summary = self.baseline.get('summary', {})
        optimized_summary = self.optimized.get('summary', {})

        peak_b = baseline_summary.get('peak_memory_gb', 0)
        peak_o = optimized_summary.get('peak_memory_gb', 0)
        current_b = baseline_summary.get('current_memory_gb', 0)
        current_o = optimized_summary.get('current_memory_gb', 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Peak memory comparison
        categories = ['Peak Memory', 'Current Memory']
        baseline_vals = [peak_b, current_b]
        optimized_vals = [peak_o, current_o]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(x - width/2, baseline_vals, width,
                       label='Baseline', color=COLORS['baseline'])
        bars2 = ax1.bar(x + width/2, optimized_vals, width,
                       label='Optimized', color=COLORS['optimized'])

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f} GB',
                        ha='center', va='bottom', fontsize=9)

        ax1.set_ylabel('Memory (GB)')
        ax1.set_title('Peak Memory Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)

        # Memory reduction
        peak_reduction = ((peak_b - peak_o) / peak_b * 100) if peak_b > 0 else 0
        current_reduction = ((current_b - current_o) / current_b * 100) if current_b > 0 else 0

        categories_red = ['Peak Memory\nReduction', 'Current Memory\nReduction']
        reductions = [peak_reduction, current_reduction]

        bars = ax2.bar(categories_red, reductions, color=COLORS['improvement'])

        for bar, val in zip(bars, reductions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_ylabel('Memory Reduction (%)')
        ax2.set_title('Memory Savings from Optimization')
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.set_ylim([0, max(reductions) * 1.2])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'peak_comparison.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'peak_comparison.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: peak_comparison.pdf/png")

    def plot_efficiency_comparison(self):
        """Compare memory efficiency metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Extract data
        times_b, alloc_b, res_b = self.extract_timeline(self.baseline)
        times_o, alloc_o, res_o = self.extract_timeline(self.optimized)

        # Cache over time
        if len(times_b) > 0:
            cache_b = res_b - alloc_b
            ax1.plot(times_b / 60, cache_b, color=COLORS['baseline'],
                    linewidth=2, label='Baseline')
        if len(times_o) > 0:
            cache_o = res_o - alloc_o
            ax1.plot(times_o / 60, cache_o, color=COLORS['optimized'],
                    linewidth=2, label='Optimized')

        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Cached Memory (GB)')
        ax1.set_title('Cached Memory Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Efficiency over time
        if len(times_b) > 0:
            eff_b = (alloc_b / np.maximum(res_b, 0.001)) * 100
            ax2.plot(times_b / 60, eff_b, color=COLORS['baseline'],
                    linewidth=2, label='Baseline')
        if len(times_o) > 0:
            eff_o = (alloc_o / np.maximum(res_o, 0.001)) * 100
            ax2.plot(times_o / 60, eff_o, color=COLORS['optimized'],
                    linewidth=2, label='Optimized')

        ax2.axhline(y=80, color='green', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('Memory Allocation Efficiency')
        ax2.set_ylim([0, 105])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Statistics comparison
        metrics = ['Mean\nAllocated', 'Peak\nAllocated', 'Mean\nCache', 'Peak\nCache']
        baseline_stats = []
        optimized_stats = []

        if len(times_b) > 0:
            baseline_stats = [
                np.mean(alloc_b),
                np.max(alloc_b),
                np.mean(res_b - alloc_b),
                np.max(res_b - alloc_b)
            ]

        if len(times_o) > 0:
            optimized_stats = [
                np.mean(alloc_o),
                np.max(alloc_o),
                np.mean(res_o - alloc_o),
                np.max(res_o - alloc_o)
            ]

        x = np.arange(len(metrics))
        width = 0.35

        if baseline_stats:
            ax3.bar(x - width/2, baseline_stats, width,
                   label='Baseline', color=COLORS['baseline'])
        if optimized_stats:
            ax3.bar(x + width/2, optimized_stats, width,
                   label='Optimized', color=COLORS['optimized'])

        ax3.set_ylabel('Memory (GB)')
        ax3.set_title('Memory Statistics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, axis='y', alpha=0.3)

        # Summary text
        ax4.axis('off')
        if baseline_stats and optimized_stats:
            improvement = {
                'peak': ((baseline_stats[1] - optimized_stats[1]) / baseline_stats[1] * 100),
                'mean': ((baseline_stats[0] - optimized_stats[0]) / baseline_stats[0] * 100),
                'cache': ((baseline_stats[3] - optimized_stats[3]) / baseline_stats[3] * 100),
            }

            summary = f"""
            Optimization Impact Summary

            Peak Allocated Memory:
              Baseline:   {baseline_stats[1]:.2f} GB
              Optimized:  {optimized_stats[1]:.2f} GB
              Reduction:  {improvement['peak']:.1f}%

            Mean Allocated Memory:
              Baseline:   {baseline_stats[0]:.2f} GB
              Optimized:  {optimized_stats[0]:.2f} GB
              Reduction:  {improvement['mean']:.1f}%

            Peak Cached Memory:
              Baseline:   {baseline_stats[3]:.2f} GB
              Optimized:  {optimized_stats[3]:.2f} GB
              Reduction:  {improvement['cache']:.1f}%

            Overall Assessment:
              {"✓ Optimization EFFECTIVE" if improvement['peak'] > 10 else "⚠ Limited improvement"}
            """
            ax4.text(0.1, 0.5, summary, fontsize=10, family='monospace',
                    verticalalignment='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_comparison.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'efficiency_comparison.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: efficiency_comparison.pdf/png")

    def plot_combined_comparison(self):
        """Create combined comparison figure for paper."""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Panel A: Overlaid timelines
        ax1 = fig.add_subplot(gs[0, :])

        times_b, alloc_b, res_b = self.extract_timeline(self.baseline)
        times_o, alloc_o, res_o = self.extract_timeline(self.optimized)

        if len(times_b) > 0:
            ax1.plot(times_b / 60, alloc_b, label='Baseline Allocated',
                    color=COLORS['baseline'], linewidth=2)
            ax1.plot(times_b / 60, res_b, label='Baseline Reserved',
                    color=COLORS['baseline'], linewidth=1.5, linestyle='--', alpha=0.5)

        if len(times_o) > 0:
            ax1.plot(times_o / 60, alloc_o, label='Optimized Allocated',
                    color=COLORS['optimized'], linewidth=2)
            ax1.plot(times_o / 60, res_o, label='Optimized Reserved',
                    color=COLORS['optimized'], linewidth=1.5, linestyle='--', alpha=0.5)

        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('GPU Memory (GB)')
        ax1.set_title('(A) Memory Usage Comparison: Baseline vs Optimized')
        ax1.legend(loc='best', ncol=2)
        ax1.grid(True, alpha=0.3)

        # Panels B & C: Peak comparison
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        baseline_summary = self.baseline.get('summary', {})
        optimized_summary = self.optimized.get('summary', {})

        peak_b = baseline_summary.get('peak_memory_gb', 0)
        peak_o = optimized_summary.get('peak_memory_gb', 0)

        # Bar chart
        ax2.bar(['Baseline', 'Optimized'], [peak_b, peak_o],
               color=[COLORS['baseline'], COLORS['optimized']])
        ax2.set_ylabel('Peak Memory (GB)')
        ax2.set_title('(B) Peak Memory Reduction')
        ax2.grid(True, axis='y', alpha=0.3)

        for i, (label, val) in enumerate([('Baseline', peak_b), ('Optimized', peak_o)]):
            ax2.text(i, val, f'{val:.1f} GB', ha='center', va='bottom', fontweight='bold')

        # Reduction percentage
        reduction = ((peak_b - peak_o) / peak_b * 100) if peak_b > 0 else 0
        ax3.bar(['Memory\nSavings'], [reduction], color=COLORS['improvement'])
        ax3.set_ylabel('Reduction (%)')
        ax3.set_title('(C) Memory Savings')
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.text(0, reduction, f'{reduction:.1f}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold')

        # Panel D: Efficiency comparison
        ax4 = fig.add_subplot(gs[2, :])

        if len(times_b) > 0 and len(times_o) > 0:
            cache_b = res_b - alloc_b
            cache_o = res_o - alloc_o

            ax4.plot(times_b / 60, cache_b, color=COLORS['baseline'],
                    linewidth=2, label='Baseline Cached', alpha=0.7)
            ax4.plot(times_o / 60, cache_o, color=COLORS['optimized'],
                    linewidth=2, label='Optimized Cached', alpha=0.7)

            ax4.fill_between(times_b / 60, cache_b, alpha=0.2, color=COLORS['baseline'])
            ax4.fill_between(times_o / 60, cache_o, alpha=0.2, color=COLORS['optimized'])

            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('Cached Memory (GB)')
            ax4.set_title('(D) Cached Memory Comparison (Lower is Better)')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)

        plt.savefig(self.output_dir / 'combined_comparison.pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / 'combined_comparison.png', bbox_inches='tight')
        plt.close()
        print("✓ Saved: combined_comparison.pdf/png (multi-panel for paper)")


def main():
    parser = argparse.ArgumentParser(
        description='Compare two memory profiles (baseline vs optimized)'
    )
    parser.add_argument('baseline', type=str,
                       help='Path to baseline profile JSON')
    parser.add_argument('optimized', type=str,
                       help='Path to optimized profile JSON')
    parser.add_argument('--output', '-o', type=str, default='comparison',
                       help='Output directory (default: comparison/)')

    args = parser.parse_args()

    comparator = ProfileComparator(args.baseline, args.optimized, args.output)
    comparator.generate_comparison()

    print("\n" + "="*60)
    print("Comparison complete!")
    print(f"Output directory: {args.output}")
    print("\nGenerated figures:")
    print("  1. timeline_comparison.pdf/png")
    print("  2. peak_comparison.pdf/png")
    print("  3. efficiency_comparison.pdf/png")
    print("  4. combined_comparison.pdf/png - Multi-panel for paper")
    print("="*60)


if __name__ == '__main__':
    main()
