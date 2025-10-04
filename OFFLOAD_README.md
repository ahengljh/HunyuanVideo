# HunyuanVideo CPU Offloading

This modification adds CPU memory offloading capabilities to HunyuanVideo for research experiments with limited GPU memory.

## Features

1. **Layer-by-Layer CPU Offloading**: Selectively offload specific transformer blocks to CPU during inference
2. **Metrics Collection**: Track GPU/CPU memory usage and inference timing
3. **Benchmark Script**: Compare performance with and without offloading
4. **Weight Sharing (Experimental)**: Reuse transformer block weights across layers to shrink resident model memory

## New Command-Line Arguments

- `--layer-offload`: Enable layer-by-layer CPU offloading during inference
- `--offload-blocks`: Comma-separated block indices to offload (e.g., "0,1,2,3,4")
- `--collect-metrics`: Collect and display memory and timing metrics
- `--layer-share-map`: Experimental weight sharing map (e.g., `"25->5,26->6"`) to reuse earlier block weights
- `--prefetch-offload`: Enable asynchronous GPU prefetching of offloaded blocks for better overlap
- `--int8-cache-offload`: Keep int8 caches of offloaded weights on GPU to cut transfer volume
- `--share-adapter-rank`: Optional low-rank residual adapter size for shared blocks (set to >0 to enable)
- `--share-adapter-scale`: Scale factor applied to residual adapters on shared blocks

## Usage

### Basic Inference with Offloading

```bash
python sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --layer-offload \
    --offload-blocks "0,1,2,3,4" \
    --collect-metrics \
    --save-path ./results
```

### Run Benchmark Comparison

```bash
python benchmark_offload.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --prompt "A cat walks on the grass, realistic style." \
    --offload-blocks "0,1,2,3,4" \
    --save-path ./benchmark_results
```

This will:
1. Run baseline inference (no offloading)
2. Run inference with offloading
3. Generate comparison table with metrics
4. Save videos and metrics JSON

### Benchmark Options

- `--skip-baseline`: Skip the baseline run
- `--skip-offload`: Skip the offload run
- `--offload-blocks`: Which blocks to offload (default: "0,1,2,3,4")
- `--layer-share-map`: Weight sharing map applied during both runs

## Understanding Block Offloading

HunyuanVideo has two types of transformer blocks:
- **Double blocks** (0-19): Dual-stream blocks for image and text processing
- **Single blocks** (20-59): Single-stream blocks for fused processing

When specifying `--offload-blocks`:
- Indices 0-19 refer to double blocks
- Indices 20+ refer to single blocks (e.g., 20 is the first single block)

Example configurations:
- `--offload-blocks "0,1,2,3,4"`: Offload first 5 double blocks
- `--offload-blocks "0-9"`: Not supported yet (use explicit list)
- `--offload-blocks "20,21,22"`: Offload first 3 single blocks

## Expected Performance

**Memory Savings**: Offloading 5-10 blocks typically saves 10-30% GPU memory

**Speed Impact**: Expect 10-50% slowdown depending on:
- Number of offloaded blocks
- PCIe bandwidth
- Model precision (fp16/bf16)

## Tips for Experimentation

1. **Start small**: Try offloading 2-3 blocks first
2. **Monitor metrics**: Use `--collect-metrics` to see exact memory/time tradeoffs
3. **Adjust based on GPU**: More VRAM → offload fewer blocks
4. **Lower resolution**: Combine with smaller `--video-size` for maximum memory savings
5. **Leverage weight sharing**: Use `--layer-share-map` to tie late blocks to earlier ones when minor quality loss is acceptable

### Weight Sharing Example (Experimental)

```bash
python sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --layer-offload \
    --offload-blocks "0,1,2,3,4" \
    --layer-share-map "25->5,26->6,27->7" \
    --collect-metrics
```

The example above reuses the weights of double blocks 5-7 for blocks 25-27, trimming resident parameters while keeping the latency benefits of layer offloading.

Set `--share-adapter-rank` to a small value (e.g., 32) to attach low-rank residual adapters to the shared blocks, with `--share-adapter-scale` controlling their contribution. Adapter energy statistics are reported under `adapter_energy` in the collected metrics.

## Example Workflow

```bash
# 1. Test baseline (full GPU)
python sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --collect-metrics

# 2. Test with light offloading
python sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --layer-offload \
    --offload-blocks "0,1,2" \
    --collect-metrics

# 3. Test with aggressive offloading
python sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --layer-offload \
    --offload-blocks "0,1,2,3,4,5,6,7,8,9" \
    --collect-metrics

# 4. Run automated benchmark
python benchmark_offload.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --offload-blocks "0,1,2,3,4"
```

## Output Files

### Benchmark Results

The benchmark script creates:
- `benchmark_results/baseline_<timestamp>.mp4`: Baseline video
- `benchmark_results/offload_<timestamp>.mp4`: Offloaded video
- `benchmark_results/metrics_<timestamp>.json`: Detailed metrics

### Metrics JSON Format

```json
[
  {
    "run_name": "baseline_20250103_120000",
    "load_time": 45.2,
    "inference_time": 120.5,
    "total_time": 165.7,
    "gpu_memory_peak_gb": 58.3,
    "gpu_memory_reserved_gb": 60.0,
    ...
  },
  {
    "run_name": "offload_20250103_120300",
    "load_time": 45.1,
    "inference_time": 145.2,
    "total_time": 190.3,
    "gpu_memory_peak_gb": 42.1,
    "gpu_memory_reserved_gb": 44.0,
    ...
  }
]
```

## Technical Details

### Implementation

The offloading mechanism:
1. Keeps specified blocks on CPU initially
2. Moves each block to GPU before its forward pass
3. Moves block back to CPU immediately after
4. Clears GPU cache after each offload

This minimizes peak GPU memory while maintaining functionality.

### Modified Files

- `hyvideo/config.py`: Added command-line arguments
- `hyvideo/inference.py`: Added metrics tracking
- `hyvideo/modules/models.py`: Implemented layer offloading logic
- `hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py`: Added metrics collection
- `benchmark_offload.py`: New benchmark script

### Preservation of Original Functionality

When offloading is disabled (default), the model behaves identically to the original implementation. All modifications are opt-in via command-line flags.

## Troubleshooting

**Out of Memory (OOM) errors**: Offload more blocks or reduce video size/length

**Very slow inference**: Offloading too many blocks; reduce the number

**CUDA errors**: Ensure blocks to offload are within valid range (0-59 for HYVideo-T/2)

## Notes

- This is a research implementation focused on functionality over optimization
- Offloading adds PCIe transfer overhead; not recommended for production
- Quality should be identical between baseline and offloaded runs
- Use the same seed for direct comparison
