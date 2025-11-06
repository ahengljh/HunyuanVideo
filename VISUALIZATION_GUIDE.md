# Memory Profiling Visualization Guide

This guide explains how to generate publication-quality figures from memory profiling data for research papers.

## Quick Start

### 1. Collect Memory Profile Data

Run your video generation with debug logging enabled:

```bash
python sample_video.py \
    --rabbit-mode \
    --rabbit-debug \
    --prompt "Your test prompt" \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50
```

This generates: `memory_logs/memory_profile_YYYYMMDD_HHMMSS.json`

### 2. Generate Visualizations

```bash
python visualize_memory.py memory_logs/memory_profile_20250107_123456.json --output figures/
```

### 3. Compare Before/After Optimization

First collect baseline (without optimization):
```bash
python sample_video.py --prompt "test" --rabbit-debug  # baseline
```

Then with optimization:
```bash
python sample_video.py --prompt "test" --rabbit-mode --rabbit-debug  # optimized
```

Compare:
```bash
python compare_memory_profiles.py \
    memory_logs/baseline.json \
    memory_logs/optimized.json \
    --output comparison/
```

## Generated Figures

### Single Profile Visualization

Running `visualize_memory.py` generates 6 figures:

1. **memory_timeline.pdf** - Complete memory timeline
   - Shows allocated vs reserved memory over time
   - Highlights cache size
   - Marks peak memory usage
   - **Use for**: Showing overall memory behavior

2. **component_breakdown.pdf** - Memory by component
   - Bar chart of memory usage per component
   - Shows transformer, VAE, text encoder, etc.
   - **Use for**: Identifying which components use most memory

3. **denoising_memory.pdf** - Denoising loop analysis
   - Memory at each denoising step
   - Cache size progression
   - **Use for**: Showing per-step memory consumption

4. **phase_memory.pdf** - Memory by processing phase
   - Average and peak memory per phase
   - Covers: initialization, model loading, denoising, VAE decode
   - **Use for**: Identifying which phase has highest memory

5. **allocated_vs_reserved.pdf** - Detailed comparison
   - 4-panel figure with scatter plot, histogram, efficiency, stats
   - **Use for**: Understanding memory allocation efficiency

6. **combined_figure.pdf** - Multi-panel figure for paper ⭐
   - Publication-ready 3x2 panel layout
   - Panels A-D covering key insights
   - **Use for**: Main figure in your paper

### Comparison Visualization

Running `compare_memory_profiles.py` generates 4 figures:

1. **timeline_comparison.pdf**
   - Side-by-side baseline vs optimized timelines
   - **Use for**: Showing optimization impact over time

2. **peak_comparison.pdf**
   - Bar charts comparing peak memory
   - Shows percentage reduction
   - **Use for**: Highlighting memory savings

3. **efficiency_comparison.pdf**
   - 4-panel detailed efficiency comparison
   - Cache, efficiency, statistics, summary
   - **Use for**: Detailed optimization analysis

4. **combined_comparison.pdf** - Multi-panel comparison ⭐
   - Publication-ready comparison figure
   - **Use for**: Main comparison figure in paper

## Example Use Cases for Papers

### Use Case 1: Showing Memory Problem

**Goal**: Demonstrate that video generation models need optimization

**Figures to use**:
- `combined_figure.pdf` - Shows high reserved memory (66 GB) vs allocated (20 GB)
- Caption: "GPU memory usage during video generation shows significant wasted cache (46 GB), indicating need for optimization"

**Key metrics to highlight**:
- Peak reserved memory: 66 GB
- Average cache: 46 GB
- Memory efficiency: ~30%

### Use Case 2: Demonstrating Optimization Effectiveness

**Goal**: Show RabbitVideo reduces memory usage

**Figures to use**:
- `combined_comparison.pdf` - Shows before/after
- Caption: "RabbitVideo optimization reduces peak memory by X% while maintaining generation quality"

**Key metrics to highlight**:
- Peak memory reduction: XX%
- Cache reduction: XX%
- Efficiency improvement: XX%

### Use Case 3: Detailed Analysis

**Goal**: Explain where memory is used

**Figures to use**:
- `component_breakdown.pdf` - Shows transformer uses most memory
- `phase_memory.pdf` - Shows denoising has highest peak
- `denoising_memory.pdf` - Shows per-step variation

**Key insights**:
- "Transformer blocks account for 60-70% of memory"
- "Denoising loop shows consistent memory usage"
- "VAE decode causes temporary spike"

## Customizing Figures

### Change Figure Size

Edit the scripts and modify `figsize` parameters:

```python
fig, ax = plt.subplots(figsize=(10, 6))  # Width, Height in inches
```

### Change Colors

Modify the `COLORS` dictionary:

```python
COLORS = {
    'allocated': '#2E86AB',    # Blue
    'reserved': '#A23B72',     # Purple
    'cached': '#F18F01',       # Orange
    ...
}
```

### Add More Metrics

Add custom plots to the scripts:

```python
def plot_custom_metric(self):
    # Your custom visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... plotting code ...
    plt.savefig(self.output_dir / 'custom_metric.pdf')
```

## Tips for Publication Quality

### 1. Vector Format (PDF)
- Always use PDF for publications (vector graphics)
- PNG is provided for quick preview

### 2. Font Sizes
- Current defaults work for most papers
- For presentations, increase font sizes by 2-3 points

### 3. Color Schemes
- Current colors are colorblind-friendly
- For black & white printing, use patterns instead of colors

### 4. Multi-panel Figures
- Use `combined_figure.pdf` or `combined_comparison.pdf`
- Panels labeled (A), (B), (C), (D) for easy reference
- All panels use consistent styling

### 5. Figure Captions

Example captions:

**For memory timeline**:
```
Figure 1: GPU memory usage during video generation (720×1280, 129 frames).
(A) Memory timeline shows reserved memory (dashed line) significantly exceeds
allocated memory (solid line), resulting in 46 GB cached. (B) Per-step memory
during denoising loop remains stable around 20 GB. (C) Component breakdown
identifies transformer blocks as primary memory consumer. (D) Memory efficiency
analysis reveals 30% utilization, indicating optimization potential.
```

**For comparison**:
```
Figure 2: Impact of RabbitVideo optimization on memory usage. (A) Overlaid
timelines show optimized version maintains lower memory throughout execution.
(B) Peak memory reduced from 66 GB to 45 GB (32% reduction). (C) Memory
savings quantification. (D) Cached memory comparison shows significant
reduction in wasted memory.
```

## Advanced Usage

### Extract Specific Metrics

Use Python to extract specific values:

```python
import json

with open('memory_logs/profile.json') as f:
    data = json.load(f)

peak_memory = data['summary']['peak_memory_gb']
print(f"Peak memory: {peak_memory:.2f} GB")

# Get component with highest memory
component_stats = data['summary']['component_stats']
for comp, stats in component_stats.items():
    print(f"{comp}: {stats['max_delta_gb']:.2f} GB")
```

### Batch Processing

Process multiple profiles:

```bash
for profile in memory_logs/*.json; do
    python visualize_memory.py "$profile" --output "figures/$(basename $profile .json)/"
done
```

### Create Animated Timeline

Use generated PNG files:

```bash
# Generate frame-by-frame snapshots (modify script to save per-step)
ffmpeg -framerate 10 -pattern_type glob -i 'figures/step_*.png' \
       -c:v libx264 -pix_fmt yuv420p memory_animation.mp4
```

## Dependencies

Install required packages:

```bash
pip install matplotlib numpy seaborn
```

## Troubleshooting

### "No timeline data available"

**Cause**: Profile was collected without background monitoring

**Solution**: Ensure profiling was enabled with `--rabbit-debug`

### "No component stats available"

**Cause**: Profile saved before component tracking completed

**Solution**: Let the full pipeline complete before interrupting

### Figures look pixelated

**Cause**: Using PNG at default resolution

**Solution**: Use PDF files for publication (vector graphics)

### Colors don't print well

**Cause**: Some printers don't handle colors well

**Solution**: Modify script to use grayscale + patterns

## Citation

If you use these visualization tools in your research, please cite:

```bibtex
@software{hunyuanvideo_memory_profiling,
  title={Memory Profiling and Visualization Tools for HunyuanVideo},
  author={...},
  year={2025},
  url={https://github.com/...}
}
```

## Support

For issues or questions:
- Check existing GitHub issues
- Review memory profiling documentation: `MEMORY_LOGGING.md`
- Open new issue with example profile JSON
