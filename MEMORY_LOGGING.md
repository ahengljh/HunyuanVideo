# GPU Memory Logging and Profiling Guide

This document explains how to use the comprehensive GPU memory logging system to understand and optimize memory usage during video generation.

## Overview

The memory logging system tracks GPU memory usage at various points during model execution:

- **Model Loading**: Memory usage when loading transformer, VAE, and text encoders
- **Denoising Loop**: Per-timestep memory tracking during the diffusion process
- **Tensor Snapshots**: Detailed information about all allocated tensors at key points
- **VAE Operations**: Memory usage during encoding/decoding

## How to Enable Memory Logging

Memory logging is controlled by the `--rabbit-debug` flag. Simply add it to your command line:

```bash
python sample_video.py \
    --video-size 512 768 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walking in the rain" \
    --seed 42 \
    --rabbit-mode \
    --rabbit-debug
```

## Key Flags

- `--rabbit-debug`: Enables comprehensive memory logging and profiling
- `--rabbit-mode`: Enables RabbitVideo memory optimizations (recommended to use together)
- `--rabbit-profile`: Alternative flag for profiling (used internally)

## What Gets Logged

### 1. Model Loading Phase

When models are loaded, you'll see:

```
[RabbitVideo] Model Memory: Transformer
----------------------------------------------------------
Module                         Parameters           Memory (MB)
----------------------------------------------------------
double_blocks                    2,456,789,012          9,872.45
single_blocks                    1,234,567,890          4,936.23
...
Total Parameters: 13,123,456,789
Total Model Memory: 52,493.83 MB (51.26 GB)
```

### 2. Tensor Snapshots

Detailed snapshots show all GPU tensors:

```
================================================================================
[RabbitVideo] GPU Memory Snapshot: after_transformer_loaded
Time: 45.23s | Phase: loading_transformer
================================================================================
Total GPU Memory: 48.00 GB
Allocated: 15.43 GB
Reserved: 16.50 GB
Free: 32.57 GB
--------------------------------------------------------------------------------
Top 20 Tensors by Memory Usage:
Shape                          DType           Memory (MB)      Grad     Device
--------------------------------------------------------------------------------
(2, 16, 33, 120, 208)         bfloat16            2048.00 MB   False    cuda:0
(42, 4096, 3072)              bfloat16            1024.00 MB   False    cuda:0
...
```

### 3. Denoising Loop Progress

During generation, compact per-step logging:

```
[RabbitVideo] Step   0/50 | Mem:  15.43GB | Reserved:  16.50GB | Time:   45.1s | latent_shape: [1, 16, 33, 64, 112] | timestep: 1000.0
[RabbitVideo] Step   1/50 | Mem:  15.87GB | Reserved:  16.50GB | Time:   47.3s | latent_shape: [1, 16, 33, 64, 112] | timestep: 980.0
[RabbitVideo] Step   2/50 | Mem:  15.91GB | Reserved:  16.50GB | Time:   49.5s | latent_shape: [1, 16, 33, 64, 112] | timestep: 960.0
```

### 4. Phase Transitions

The profiler logs when entering different phases:

```
[RabbitVideo] Phase: initialization
[RabbitVideo] Phase: loading_transformer
[RabbitVideo] Phase: loading_vae
[RabbitVideo] Phase: loading_text_encoder
[RabbitVideo] Phase: models_loaded_complete
[RabbitVideo] Phase: denoising_loop
[RabbitVideo] Phase: after_denoising
[RabbitVideo] Phase: vae_decode
[RabbitVideo] Phase: pipeline_complete
```

## Memory Profile Output

At the end of execution, a comprehensive JSON file is saved to `./memory_logs/memory_profile_YYYYMMDD_HHMMSS.json` containing:

### Profile Structure

```json
{
  "summary": {
    "current_memory_gb": 15.43,
    "peak_memory_gb": 23.87,
    "component_peaks": {
      "Transformer": 18.45,
      "VAE": 8.23,
      "TextEncoder": 3.45
    },
    "total_duration": 245.67,
    "component_stats": {
      "loading_transformer": {
        "calls": 1,
        "avg_delta_gb": 12.34,
        "max_delta_gb": 12.34,
        "total_time": 15.67,
        "avg_time": 15.67
      }
    }
  },
  "analysis": {
    "peak_memory_gb": 23.87,
    "avg_memory_gb": 18.45,
    "min_memory_gb": 2.34,
    "std_memory_gb": 4.56,
    "memory_spikes": [
      {
        "timestamp": 78.45,
        "memory_gb": 23.87,
        "phase": "denoising_step_5",
        "spike_size_gb": 5.42
      }
    ],
    "offload_candidates": [
      {
        "component": "double_blocks",
        "max_memory_gb": 8.45,
        "avg_memory_gb": 7.23,
        "potential_savings_gb": 6.76
      }
    ]
  },
  "timeline": [
    {
      "timestamp": 0.0,
      "memory_gb": 2.34,
      "phase": "initialization"
    }
  ]
}
```

## Using Memory Data for Optimization

### 1. Identify Memory Bottlenecks

Check the `offload_candidates` section to see which components use the most memory:

```python
{
  "component": "double_blocks",
  "max_memory_gb": 8.45,
  "potential_savings_gb": 6.76
}
```

### 2. Analyze Memory Spikes

Look for sudden increases in memory usage:

```python
{
  "timestamp": 78.45,
  "memory_gb": 23.87,
  "phase": "denoising_step_5",
  "spike_size_gb": 5.42
}
```

This indicates that step 5 of denoising caused a 5.42 GB spike.

### 3. Optimize Based on Findings

Based on the profiling data, you can:

#### Enable More Aggressive Offloading

```bash
--rabbit-aggressive-offload \
--rabbit-offload-threshold 18.0
```

#### Adjust Block Prefetching

```bash
--rabbit-prefetch-blocks 1  # Reduce from default 2
```

#### Enable VAE Tiling

Add to your script:
```python
enable_tiling=True  # Reduces VAE peak memory
```

#### Use Gradient Checkpointing

```bash
--rabbit-gradient-checkpointing
```

## Real-time Monitoring

The memory profiler also runs a background monitoring thread that samples memory usage every 0.1 seconds. This creates a detailed timeline for visualizing memory patterns.

## Example: Analyzing a Memory Profile

After running with `--rabbit-debug`, check the logs:

1. **Model Loading**: Did transformer loading exceed your target memory?
2. **Peak Memory**: What was the peak during denoising?
3. **Per-Step Memory**: Are there specific steps with higher memory?
4. **VAE Decode**: How much memory did VAE decoding use?

Example workflow:

```bash
# Run with logging
python sample_video.py --rabbit-mode --rabbit-debug --prompt "test"

# Check the generated profile
cat memory_logs/memory_profile_*.json | jq '.summary.peak_memory_gb'

# Adjust parameters and re-run
python sample_video.py \
    --rabbit-mode \
    --rabbit-debug \
    --rabbit-target-memory 20.0 \
    --rabbit-aggressive-offload \
    --prompt "test"
```

## Tips for Memory Optimization

1. **Start with profiling**: Always profile first to understand your baseline
2. **Check peak memory**: Focus on reducing peak memory, not average
3. **Identify spikes**: Look for sudden memory increases and investigate their cause
4. **Iterative optimization**: Make one change at a time and measure impact
5. **Balance performance**: Lower memory usually means slower inference

## Troubleshooting

### Memory profiling slows down execution

The tensor enumeration (via `gc.get_objects()`) can be slow. This is only done at snapshot points (every 5 steps), not every step.

### Profile file is very large

The timeline contains many samples (10 per second). For production runs, you can reduce `monitor_interval` in the profiler.

### OOM even with profiling enabled

Profiling adds minimal overhead, but if you're at the edge of OOM, try:
- Reducing `--video-size` or `--video-length`
- Enabling `--rabbit-aggressive-offload`
- Lowering `--rabbit-target-memory`

## Advanced Usage

### Programmatic Access

You can access the profiler in your own code:

```python
from hyvideo.utils.memory_profiler import get_memory_profiler

profiler = get_memory_profiler()
if profiler:
    # Log current state
    profiler.log_allocated_tensors("custom_checkpoint")

    # Track a custom component
    with profiler.track_component("my_component"):
        # Your code here
        pass
```

### Custom Snapshots

Add your own snapshot points:

```python
if mem_profiler:
    mem_profiler.log_allocated_tensors(tag="before_custom_operation", top_n=10)
```

## Summary

The memory logging system provides comprehensive visibility into GPU memory usage during video generation. Use it to:

1. Understand where memory is being used
2. Identify optimization opportunities
3. Tune RabbitVideo parameters for your specific hardware
4. Debug out-of-memory issues

For questions or issues, please check the main README or open an issue on GitHub.
