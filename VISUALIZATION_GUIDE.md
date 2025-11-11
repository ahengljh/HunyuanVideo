# RabbitVideo Visualization Guide

This guide shows how to visualize RabbitVideo memory timeline data for paper presentations.

## Quick Start

### 1. Generate Timeline Data

Run RabbitVideo with timeline saving enabled:

```bash
python sample_video.py \
    --rabbit-mode \
    --rabbit-debug \
    --rabbit-save-timeline ./memory_timeline.json \
    --prompt "A cat walking in the garden" \
    --video-size 540 960 \
    --video-length 129 \
    --infer-steps 30
```

### 2. Visualize Timeline Data

Generate all visualization plots:

```bash
python visualize_rabbit_timeline.py \
    --timeline memory_timeline.json \
    --output ./figures/
```

This will create:
- `memory_over_time.png/pdf` - Memory usage over time
- `blocks_management.png/pdf` - Block swapping visualization
- `memory_breakdown.png/pdf` - Allocated vs cache breakdown
- `block_activity_heatmap.png/pdf` - Block activity heatmap
- `method_comparison.png/pdf` - Comparison with baselines
- `comprehensive_dashboard.png/pdf` - All-in-one dashboard

## Generated Plots

### Plot 1: Memory Over Time
Shows allocated, reserved, and cache memory throughout inference.

**Best for:** Demonstrating memory efficiency and cache management.

**Key features:**
- Three lines: Allocated (used), Reserved (total), Cache (wasted)
- Peak memory highlighted
- Statistics box with key metrics
- Publication-quality formatting

### Plot 2: Block Management
Shows blocks on GPU and memory usage in synchronized subplots.

**Best for:** Illustrating the relationship between block swapping and memory.

**Key features:**
- Upper plot: Number of blocks on GPU over time
- Lower plot: Reserved memory over time
- Shows correlation between block count and memory

### Plot 3: Memory Breakdown
Stacked area chart showing allocated vs cache waste.

**Best for:** Visualizing memory efficiency.

**Key features:**
- Blue area: Actually used memory (allocated)
- Orange area: Wasted memory (cache)
- Memory efficiency percentage displayed

### Plot 4: Block Activity Heatmap
Heatmap showing which blocks are active over time.

**Best for:** Understanding block execution patterns.

**Key features:**
- Red = block on GPU
- Y-axis: Block index (0-59)
- X-axis: Time progress
- Clearly shows sequential execution pattern
- Distinguishes double blocks (0-19) vs single blocks (20-59)

### Plot 5: Method Comparison
Bar chart comparing RabbitVideo with baselines.

**Best for:** Main results figure in paper.

**Key features:**
- Compares 4 methods: Default, CPU Offload, RabbitVideo, RabbitVideo+Aggressive
- Two metrics: Peak memory and inference time
- 24GB GPU limit highlighted
- Speedup annotations

### Plot 6: Comprehensive Dashboard
All-in-one figure with multiple subplots.

**Best for:** Supplementary material or detailed presentation.

**Key features:**
- 4 subplots in one figure: Memory over time, breakdown, blocks on GPU, statistics
- Complete summary of all metrics
- Ready for direct paper inclusion

## Customization

### Change Plot Style

Edit the `plt.rcParams` at the top of `visualize_rabbit_timeline.py`:

```python
plt.rcParams.update({
    'font.size': 14,              # Increase font size
    'figure.dpi': 600,            # Higher resolution
    'font.family': 'sans-serif',  # Change font family
})
```

### Adjust Figure Size

Modify the `figsize` parameter in each plot function:

```python
fig, ax = plt.subplots(figsize=(14, 7))  # Width, Height in inches
```

### Add Custom Comparisons

Edit `plot_comparison_bar()` to add your own baseline data:

```python
methods = ['Default', 'Baseline1', 'Baseline2', 'RabbitVideo']
peak_memory = [66.2, 45.0, 32.0, 24.3]
inference_time = [185, 250, 300, 210]
```

## Output Formats

All plots are saved in two formats:
- **PNG**: For presentations, websites, and quick viewing
- **PDF**: For paper submission (vector graphics, scalable)

## Example Output

After running the visualization script, you'll see:

```
============================================================
RabbitVideo Timeline Visualization
============================================================

Loading timeline data from: memory_timeline.json
✓ Loaded 1234 data points

Generating visualizations...

✓ Saved: ./figures/memory_over_time.png
✓ Saved: ./figures/blocks_management.png
✓ Saved: ./figures/memory_breakdown.png
✓ Saved: ./figures/block_activity_heatmap.png
✓ Saved: ./figures/method_comparison.png
✓ Saved: ./figures/comprehensive_dashboard.png

============================================================
✓ All visualizations saved to: ./figures/
============================================================

Summary Statistics:
  Peak Reserved Memory:  24.32 GB
  Peak Allocated Memory: 20.75 GB
  Total Inference Time:  210.3 seconds
  Number of Data Points: 1234
```

## Tips for Paper Presentation

### Main Paper Figure
Use **method_comparison.png** as your main results figure. It clearly shows:
1. RabbitVideo enables 24GB GPU usage (vs 66GB default)
2. Only 1.13x slowdown (vs 2.6x for CPU offload)

### Supplementary Material
Include:
1. **comprehensive_dashboard.pdf** - Full system overview
2. **block_activity_heatmap.pdf** - Sequential execution pattern
3. **memory_over_time.pdf** - Detailed memory trace

### Conference Presentation
- Use **memory_over_time.png** for animated explanation
- Show **blocks_management.png** to explain the pipeline
- Conclude with **method_comparison.png** for results

## Advanced: Comparing Multiple Runs

To compare different configurations:

```python
# Load multiple timelines
data_default = load_timeline_data('timeline_default.json')
data_aggressive = load_timeline_data('timeline_aggressive.json')

# Plot comparison
plt.plot(data_default['timestamps'], data_default['reserved_gb'], label='Default')
plt.plot(data_aggressive['timestamps'], data_aggressive['reserved_gb'], label='Aggressive')
plt.legend()
plt.savefig('comparison.png')
```

## Dependencies

The visualization script requires:
```bash
pip install matplotlib numpy
```

Already included in HunyuanVideo's environment.

## Troubleshooting

### Font warnings
If you see font warnings, install Times New Roman or change font family:
```python
'font.serif': ['DejaVu Serif'],  # Instead of Times New Roman
```

### Memory error when loading large JSON
For very long runs, the JSON might be large. Use chunking:
```python
import json
with open('timeline.json', 'r') as f:
    data = json.load(f)
    # Downsample if needed
    data['timestamps'] = data['timestamps'][::10]  # Every 10th point
```

### PDF not rendering
If PDFs don't render properly, use PNG only:
```bash
# Comment out PDF saving lines in the script
# plt.savefig(output_path / 'figure.pdf')
```

## Citation

If you use these visualizations in your paper, please cite:

```bibtex
@article{rabbitvideo2025,
  title={RabbitVideo: Memory-Efficient Video Diffusion via Strategic Block Offloading},
  author={},
  journal={},
  year={2025}
}
```
