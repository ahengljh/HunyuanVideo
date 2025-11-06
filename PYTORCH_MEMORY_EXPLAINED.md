# Understanding PyTorch GPU Memory Management

This document explains how PyTorch manages GPU memory and why memory numbers may seem confusing at first.

## The Question

When running memory profiling with `--rabbit-debug`, you might see output like:

```
Total GPU Memory: 79.25 GB
Allocated: 38.61 GB
Reserved: 38.83 GB
Free: 40.64 GB
--------------------------------------------------------------------------------
Total tracked tensors: 1,638
Memory in top 20: 3.32 GB
```

**Why does the top 20 tensors (3.32 GB) not add up to allocated (38.61 GB)?**

## The Answer: PyTorch's Memory Hierarchy

### Memory Levels

PyTorch has several layers of memory management:

```
┌─────────────────────────────────────────────────────────┐
│  Physical GPU VRAM (e.g., 80 GB)                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Reserved by PyTorch (38.83 GB)                   │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Allocated (38.61 GB)                       │  │  │
│  │  │  - Model parameters: ~20-25 GB              │  │  │
│  │  │  - Activations: ~10-15 GB                   │  │  │
│  │  │  - Embeddings/intermediate: ~3-8 GB         │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Cached (0.22 GB)                           │  │  │
│  │  │  - Free but reserved for fast reuse         │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Free GPU Memory (40.64 GB)                       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 1. Total GPU Memory (79.25 GB)
- Physical VRAM on your GPU
- Fixed by hardware
- Shown by `nvidia-smi`

### 2. Reserved Memory (38.83 GB)
- Memory requested from CUDA driver by PyTorch
- PyTorch's **caching allocator** manages this pool
- Includes both in-use and cached memory
- `torch.cuda.memory_reserved()`

### 3. Allocated Memory (38.61 GB)
- Memory **actually used** by tensors right now
- This is what tensors occupy
- `torch.cuda.memory_allocated()`

### 4. Cached Memory (0.22 GB)
- Reserved - Allocated = 38.83 - 38.61 = 0.22 GB
- Memory reserved but not currently in use
- Kept for **fast reallocation** without asking CUDA again
- Prevents fragmentation

## Why Only Top 20 Tensors Show 3.32 GB?

In your example:
- **1,638 total tensors** on GPU
- **Top 20 show**: 3.32 GB
- **Remaining 1,618 tensors**: ~35 GB!

Let's break down where the 38.61 GB actually goes:

### Typical Memory Distribution

```python
# Example breakdown for HunyuanVideo at 720x1280x129

1. Model Parameters (requires_grad=True)
   - Transformer blocks: ~20-25 GB
     • Double blocks: ~15 GB
     • Single blocks: ~5 GB
     • Attention layers: ~3-5 GB
     • MLP layers: ~2-3 GB
   - Text encoder: ~2-4 GB
   - VAE: ~2-3 GB
   Total Parameters: ~24-32 GB

2. Activations (requires_grad=False)
   - Text embeddings: ~1-2 GB
     • Your example: (128320, 4096) = 1 GB in top 20
   - Latent representations: ~0.5-2 GB
   - Intermediate activations: ~3-8 GB
     • Your example: 19x (21504, 3072) = 2.4 GB in top 20
   - Attention outputs: ~1-2 GB
   Total Activations: ~5-14 GB

3. PyTorch Overhead
   - CUDA allocator metadata: ~0.1-0.5 GB
   - Memory alignment padding: ~0.1-0.3 GB
   Total Overhead: ~0.2-0.8 GB
```

## Who Marks Memory as "Used"?

### 1. PyTorch Caching Allocator
```python
# When PyTorch needs memory
torch.cuda.malloc(size)
  ↓
Check if size available in cache
  ↓
If not: cudaMalloc(large_block)  # Reserve from CUDA
  ↓
Return pointer to tensor
```

The caching allocator:
- Requests large chunks from CUDA (e.g., 2 GB at a time)
- Manages sub-allocation within those chunks
- Keeps freed memory cached for reuse

### 2. Model Loading
```python
model.load_state_dict(checkpoint)
  ↓
For each parameter tensor:
  - Allocate memory from cache
  - Copy weights from CPU/disk
  - Mark as allocated
```

For a 20 GB model:
- 20 GB immediately shows as "allocated"
- This is your base memory usage

### 3. Forward Pass
```python
output = model(input)
  ↓
Create intermediate tensors:
  - Layer outputs
  - Attention scores
  - Activation results
  ↓
Each allocation increases "allocated" counter
```

## Your Specific Example Analysis

From your log:
```
Allocated: 38.61 GB
Top 20 tensors: 3.32 GB
Total tensors: 1,638
```

**Where's the other 35.29 GB?**

### The Remaining 1,618 Tensors

Let's estimate:

```python
# If top 20 average 166 MB each (3320 MB / 20)
# Remaining tensors might average:
35,290 MB / 1,618 tensors ≈ 21.8 MB per tensor

# These could be:
- Small parameter tensors: biases, norms (1-10 MB each)
- Intermediate layer outputs (10-50 MB each)
- Attention caches (20-100 MB each)
- Embedding tables (50-200 MB each)
- Optimizer states if training (matches param size)
```

### Typical Large Tensors (seen in your top 20)

1. **(128320, 4096) float16 = 1002 MB**
   - Text embeddings from CLIP/LLM
   - Shape suggests: 128k tokens × 4096 dims
   - Likely: Concatenated text features for batch

2. **19× (21504, 3072) bfloat16 = 126 MB each**
   - Transformer block outputs
   - Shape: 21504 spatial-temporal tokens × 3072 hidden dims
   - These are activations from each layer
   - Total: 19 layers × 126 MB = 2.4 GB

## Improved Logging Output

With the enhanced profiler, you'll now see:

```
================================================================================
[RabbitVideo] GPU Memory Snapshot: before_denoising_loop
Time: 27.68s | Phase: denoising_loop
================================================================================
Total GPU Memory: 79.25 GB
Allocated: 38.61 GB  (memory actively used by tensors)
Reserved: 38.83 GB  (memory reserved from CUDA)
Cached: 0.22 GB  (reserved but not in use, for fast reallocation)
Free: 40.64 GB
--------------------------------------------------------------------------------
Memory Breakdown:
  Total Tracked Tensors: 1,638
  Total Tracked Memory: 38.45 GB (99.6% of allocated)
  Parameters (requires_grad=True): 24.23 GB (856 tensors)
  Activations (requires_grad=False): 14.22 GB (782 tensors)
  Untracked/Overhead: 0.16 GB (CUDA allocator overhead, fragmentation)
--------------------------------------------------------------------------------
Top 20 Tensors by Memory Usage:
Shape                          DType           Memory (MB)     Grad     Device
--------------------------------------------------------------------------------
(128320, 4096)                 float16            1002.50 MB   False    cuda:0
(21504, 3072)                  bfloat16            126.00 MB   True     cuda:0
... (18 more)
--------------------------------------------------------------------------------
Top 20 tensors: 3.32 GB (8.6% of tracked)
Remaining 1,618 tensors: 35.13 GB (91.4% of tracked)
================================================================================
```

## Key Takeaways

1. **Many tensors exist**: Your model has 1,638 tensors, not just 20
2. **Model parameters dominate**: 20-30 GB is typical for large transformers
3. **Activations add up**: Hundreds of intermediate tensors during forward pass
4. **Reserved ≠ Allocated**: PyTorch caches memory for performance
5. **Top N is just a sample**: Use it to find the largest consumers, not total

## How to Investigate Further

### 1. Check all tensor sizes
```python
# In profiler output, look for:
"Remaining X tensors: Y GB"
```

### 2. Profile with more top tensors
```python
# See top 50 instead of 20
mem_profiler.log_allocated_tensors(tag="snapshot", top_n=50)
```

### 3. Check the JSON profile
```bash
cat memory_logs/memory_profile_*.json | jq '.summary.component_stats'
```

### 4. Use the explanation feature
The profiler now automatically prints:
```
[RabbitVideo] PyTorch Memory Management Explanation
Memory Hierarchy:
  1. Total GPU Memory:     79.25 GB  (Physical VRAM on your GPU)
  2. Reserved by PyTorch:  38.83 GB  (Requested from CUDA driver)
     ├─ Allocated:         38.61 GB  (Actually used by tensors)
     └─ Cached:             0.22 GB  (Reserved but free, for fast reuse)
  3. Free GPU Memory:      40.64 GB  (Not reserved by PyTorch)
```

## Common Misconceptions

### ❌ "Top 20 tensors should equal allocated memory"
No - there are typically hundreds or thousands of tensors. Top 20 is just the largest.

### ❌ "Reserved memory is wasted"
No - it includes both used (allocated) and cached memory. Caching improves performance.

### ❌ "I should see all model parameters in top 20"
No - model parameters are split across layers. Each layer's parameters are separate tensors.

### ✅ "Most memory is in parameters and activations"
Yes! For inference:
- ~60-70% model parameters
- ~30-40% activations and embeddings

## Memory Optimization Strategies

Based on understanding where memory goes:

### 1. Model Parameters (if too large)
```bash
--rabbit-enable-offloading  # Offload unused blocks to CPU
--rabbit-aggressive-offload  # More aggressive offloading
```

### 2. Activations (if spikes occur)
```bash
--rabbit-gradient-checkpointing  # Recompute activations instead of storing
--enable-tiling  # VAE tiling for decoding
```

### 3. Cache Management
```bash
torch.cuda.empty_cache()  # Free cached memory
# Note: Usually not needed, PyTorch manages this well
```

## References

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Understanding GPU Memory](https://pytorch.org/docs/stable/notes/cuda.html#memory-semantics)
- [Caching Allocator](https://pytorch.org/docs/stable/notes/cuda.html#caching-allocator)

## Summary

The "mystery" of why top 20 tensors (3.32 GB) doesn't equal allocated (38.61 GB) is simple:

**You have 1,618 more tensors that contain 35 GB of model parameters, activations, embeddings, and intermediate computations!**

The improved logging now shows this clearly with a complete breakdown of where all memory is allocated.
