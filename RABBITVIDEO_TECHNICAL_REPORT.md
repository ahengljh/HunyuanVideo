# RabbitVideo: Memory-Efficient Video Diffusion Technical Report

**Author:** Jinheng Li
**Date:** November 2025
**Version:** 1.1 (with KV Cache Optimization)

---

## Executive Summary

RabbitVideo is a memory optimization system designed for large-scale video diffusion transformers that enables high-quality video generation on consumer-grade GPUs (24GB) by reducing peak memory usage from ~66GB to ~23GB with only 15% time overhead. The system achieves this through strategic block-level offloading, synchronous memory management, and auxiliary model orchestration.

**Key Achievements:**
- **Memory Reduction:** 66GB → 23GB (65% reduction)
- **Performance:** 1.15x slowdown vs baseline (2.6x better than naive CPU offload)
- **Compatibility:** Works with existing model architectures without retraining
- **Simplicity:** Only 3-4 user-facing flags, no complex tuning

This report provides complete implementation details suitable for adapting RabbitVideo to other large-scale diffusion models.

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Core Design Principles](#2-core-design-principles)
3. [System Architecture](#3-system-architecture)
4. [Implementation Details](#4-implementation-details)
5. [Phase-by-Phase Implementation](#5-phase-by-phase-implementation)
6. [Advanced: KV Cache Optimization](#6-advanced-kv-cache-optimization)
7. [Integration Guide](#7-integration-guide)
8. [Performance Characteristics](#8-performance-characteristics)
9. [Limitations and Constraints](#9-limitations-and-constraints)
10. [Code Reference](#10-code-reference)

---

## 1. Problem Analysis

### 1.1 Memory Bottleneck in Video Diffusion

HunyuanVideo's architecture consists of:
- **Transformer blocks:** 60 blocks (20 double-stream + 40 single-stream)
- **Block size:** ~648MB per block
- **Total transformer:** 60 × 648MB = ~38.8GB
- **Auxiliary models:**
  - VAE: ~8GB
  - Text encoders (CLIP + T5): ~4GB
- **Activations and buffers:** ~10GB
- **PyTorch cache overhead:** Additional 46GB reserved memory

**Total peak memory:** ~66GB (exceeds 24GB consumer GPUs)

### 1.2 Why Existing Solutions Fall Short

| Approach | Granularity | Memory | Speed | Issues |
|----------|-------------|---------|-------|--------|
| **Model-level offload** | Entire 38GB model | ✓ Low | ✗ 2.6x slower | Massive transfer overhead |
| **Layer-level offload** | ~72MB per layer | ✓ Low | ✗ 3x+ slower | Tight coupling, activation sync |
| **Gradient checkpointing** | N/A | Partial | ✗ Training only | Requires backward pass |
| **Quantization** | N/A | Partial | ✓ Fast | Quality degradation |

### 1.3 Key Insight: Block-Level Granularity

**Observation:** In transformer inference, only ONE block is active at any moment due to sequential execution.

**Solution:** Keep only the active block on GPU, offload the rest to CPU.

**Why blocks?**
1. **Functional completeness:** One attention + MLP pass is atomic
2. **Transfer efficiency:** 648MB fits PCIe bandwidth sweet spot (~40ms/transfer)
3. **Activation locality:** Activations stay on GPU, no extra transfer
4. **Independence:** Blocks don't share parameters, clean separation

---

## 2. Core Design Principles

### 2.1 Proactive Memory Management

**Principle:** Prevent peak memory rather than react to OOM.

**Implementation:**
- Initialize with most blocks already on CPU
- Free memory BEFORE allocating new blocks
- Never allocate without first checking capacity

### 2.2 Synchronous Transfer Protocol

**Problem:** PyTorch's `.to('cpu')` doesn't immediately free GPU memory cache.

**Solution:** Explicit synchronization + aggressive cache clearing:

```python
def synchronous_offload(block):
    block.to('cpu')                 # Move parameters
    torch.cuda.synchronize()        # Wait for transfer completion
    torch.cuda.empty_cache()        # Request cache release
    torch.cuda.synchronize()        # Wait for cache clearing
    torch.cuda.empty_cache()        # Double-clear for aggressive release
```

**Critical:** This is the difference between 45GB and 23GB peak memory.

### 2.3 Three-Phase Architecture

1. **Phase 1 - Smart Initialization:** Proactively offload 55/60 blocks at startup
2. **Phase 2 - Dynamic Block Swapping:** Load/offload blocks on-demand during inference
3. **Phase 3 - Auxiliary Model Management:** Offload VAE and text encoders when idle

### 2.4 LRU Eviction Policy

**Problem:** When GPU is full, which blocks to offload?

**Solution:** Least Recently Used (LRU) with access count:

```python
def select_blocks_to_offload(num_needed):
    # Sort by: (1) access count, (2) last access time
    return sorted(gpu_blocks, key=lambda b: (
        access_count[b],      # Less accessed first
        last_access_time[b]   # Older first (tie-breaker)
    ))[:num_needed]
```

**Insight:** Exploits sequential execution pattern where blocks are rarely reused.

---

## 3. System Architecture

### 3.1 Component Hierarchy

```
RabbitVideoOffloader (Orchestrator)
├── MemoryMonitor (Tracking)
│   ├── Timeline recording
│   ├── Peak memory tracking
│   └── Statistics collection
│
├── BlockTracker (State Management)
│   ├── GPU/CPU location tracking
│   ├── LRU eviction policy
│   ├── Access pattern recording
│   └── Block size estimation
│
├── BlockManager (Transfer Operations)
│   ├── Synchronous CPU→GPU transfer
│   ├── Synchronous GPU→CPU transfer
│   └── Memory-aware transfer scheduling
│
└── [Optional] KV Cache System (v1.1+)
    ├── StaticRegionDetector
    └── SelectiveKVCache
```

### 3.2 Data Flow

```
Initialization:
  Model Loading → Extract Blocks → Calculate Sizes → Proactive Offload

Inference Loop:
  For each denoising step:
    For each block[i]:
      1. ensure_block_on_gpu(i)
         ├─→ Check if block[i] on GPU
         ├─→ If not: free_gpu_memory()
         │           ├─→ Select LRU blocks
         │           └─→ Synchronous offload
         └─→ Load block[i] synchronously

      2. Execute block forward()

      3. [Stateless mode only] offload_block_after_execution(i)

      4. clear_cache_after_block()

Post-inference:
  Print statistics → Save timeline → Cleanup
```

### 3.3 Memory Timeline Example

```
Time  |  Reserved  |  Allocated  |  Cache  |  Blocks on GPU  |  Action
------|------------|-------------|---------|-----------------|------------------
0s    |  45.2 GB   |  38.8 GB    |  6.4 GB |  60             |  Initial load
1s    |  24.3 GB   |  20.1 GB    |  4.2 GB |  5              |  After Phase 1
5s    |  24.8 GB   |  20.7 GB    |  4.1 GB |  6              |  Block swap
10s   |  24.5 GB   |  20.3 GB    |  4.2 GB |  5              |  Steady state
200s  |  23.9 GB   |  19.8 GB    |  4.1 GB |  5              |  Completed
```

---

## 4. Implementation Details

### 4.1 MemoryMonitor Class

**Purpose:** Track GPU memory usage with detailed statistics.

**Key Methods:**

```python
class MemoryMonitor:
    def __init__(self, device, debug=False, logger=None):
        self.device = device
        self.timeline_data = {
            "timestamps": [],
            "allocated_gb": [],
            "reserved_gb": [],
            "cache_gb": [],
            "blocks_on_gpu": [],
            "current_block": []
        }
        self.peak_allocated = 0
        self.peak_reserved = 0

    def record(self, blocks_on_gpu=0, current_block=-1):
        """Record current memory snapshot to timeline."""
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        cache = reserved - allocated

        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

        self.timeline_data["timestamps"].append(time.time() - self.start_time)
        self.timeline_data["allocated_gb"].append(round(allocated, 2))
        self.timeline_data["reserved_gb"].append(round(reserved, 2))
        self.timeline_data["cache_gb"].append(round(cache, 2))
        # ... store other metrics

    def get_current_memory(self):
        """Returns (allocated_gb, reserved_gb, cache_gb)."""
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        return allocated, reserved, reserved - allocated
```

**Design Notes:**
- Records memory snapshots at key points (initialization, block swaps, step boundaries)
- Distinguishes between allocated (actual usage) and reserved (PyTorch cache)
- Timeline data enables post-hoc analysis and visualization

### 4.2 BlockTracker Class

**Purpose:** Track block locations and implement LRU eviction.

**Key Data Structures:**

```python
class BlockTracker:
    def __init__(self, num_blocks, debug=False, logger=None):
        # Location tracking
        self.gpu_blocks: Set[int] = set()
        self.cpu_blocks: Set[int] = set(range(num_blocks))

        # LRU policy data
        self.block_access_count: Dict[int, int] = defaultdict(int)
        self.block_last_access_time: Dict[int, float] = {}

        # Size estimation
        self.block_sizes: Dict[int, float] = {}  # GB
        self.average_block_size: float = 0.65  # Initial estimate

        # Statistics
        self.total_transfers = 0
        self.total_transfer_time = 0.0
```

**Key Methods:**

```python
def record_access(self, block_idx):
    """Update LRU tracking."""
    self.block_access_count[block_idx] += 1
    self.block_last_access_time[block_idx] = time.time()

def select_blocks_to_offload(self, num_needed):
    """Select blocks using LRU policy."""
    gpu_blocks_sorted = sorted(
        self.gpu_blocks,
        key=lambda b: (
            self.block_access_count.get(b, 0),     # Primary: access count
            self.block_last_access_time.get(b, 0)  # Secondary: recency
        )
    )
    return gpu_blocks_sorted[:num_needed]

def estimate_block_size(self, block):
    """Calculate actual block memory size."""
    total_bytes = sum(p.numel() * p.element_size()
                      for p in block.parameters())
    return total_bytes / (1024**3)
```

**Design Notes:**
- Maintains separate sets for GPU/CPU blocks (O(1) lookup)
- LRU policy exploits sequential execution pattern
- Dynamic size calculation accounts for architecture variations

### 4.3 BlockManager Class

**Purpose:** Handle synchronous block transfers with memory awareness.

**Critical Implementation:**

```python
class BlockManager:
    def move_block_to_cpu(self, block, block_idx):
        """Synchronous CPU offload with aggressive cache clearing."""
        start_time = time.time()

        # Synchronous transfer protocol
        block.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Double-clear for aggressive release

        # Update tracking
        self.tracker.mark_on_cpu(block_idx)
        self.tracker.total_transfers += 1
        self.tracker.total_transfer_time += time.time() - start_time

    def move_block_to_gpu(self, block, block_idx):
        """Synchronous GPU load (called AFTER free_gpu_memory)."""
        block.to(self.device)
        torch.cuda.synchronize()

        self.tracker.mark_on_gpu(block_idx)
        self.tracker.total_transfers += 1

    def free_gpu_memory(self, required_gb, blocks_dict):
        """Free memory BEFORE allocating (critical for preventing OOM)."""
        current_alloc, current_reserved, cache = self.memory_monitor.get_current_memory()

        # Calculate available space
        gpu_capacity_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        available_gb = gpu_capacity_gb - current_reserved - 1.5  # 1.5GB buffer

        if available_gb >= required_gb:
            return  # Sufficient space

        # Calculate deficit
        need_to_free_gb = required_gb - available_gb + 0.5  # +0.5GB extra buffer

        # Select and offload blocks using LRU
        freed_gb = 0.0
        for block_idx in self.tracker.select_blocks_to_offload(len(self.tracker.gpu_blocks)):
            block_size = self.tracker.get_block_size(block_idx)

            if block_idx in blocks_dict:
                self.move_block_to_cpu(blocks_dict[block_idx], block_idx)
                freed_gb += block_size

            if freed_gb >= need_to_free_gb:
                break

        # Final aggressive cache clear
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

**Critical Design Decision:**
The `free_gpu_memory()` method is called BEFORE `move_block_to_gpu()`. This ensures we never trigger OOM by allocating without first making space.

### 4.4 RabbitVideoOffloader Class

**Purpose:** Main orchestrator coordinating all components.

**Initialization:**

```python
class RabbitVideoOffloader:
    def __init__(self, model, device, blocks_to_keep=1, debug=False,
                 logger=None, stateless=False):
        # Extract blocks from model
        self.double_blocks = list(model.double_blocks)
        self.single_blocks = list(model.single_blocks)
        self.all_blocks = self.double_blocks + self.single_blocks
        self.num_blocks = len(self.all_blocks)

        # Create block index mapping
        self.blocks_dict = {i: block for i, block in enumerate(self.all_blocks)}

        # Initialize components
        self.memory_monitor = MemoryMonitor(device, debug, logger)
        self.tracker = BlockTracker(self.num_blocks, debug, logger)
        self.block_manager = BlockManager(device, self.tracker,
                                         self.memory_monitor, debug, logger)

        # Configuration
        self.blocks_to_keep = blocks_to_keep
        self.stateless = stateless
        self.prefetch_enabled = not stateless
```

**Core Algorithm - ensure_block_on_gpu:**

```python
def ensure_block_on_gpu(self, block_idx):
    """Phase 2: Ensure block is on GPU for execution."""

    # 1. Record access for LRU
    self.tracker.record_access(block_idx)

    # 2. AGGRESSIVE MODE: Offload previous block immediately
    #    (Keeps only 1 block on GPU at a time)
    if self.tracker.previous_block is not None and \
       self.tracker.previous_block != block_idx:
        if self.tracker.is_on_gpu(self.tracker.previous_block):
            prev_block = self.blocks_dict[self.tracker.previous_block]
            self.block_manager.move_block_to_cpu(prev_block,
                                                self.tracker.previous_block)
            torch.cuda.empty_cache()

    # 3. Check if already on GPU
    if self.tracker.is_on_gpu(block_idx):
        self.tracker.current_executing_block = block_idx
        self.tracker.previous_block = block_idx
        return

    # 4. Get block and size
    block = self.blocks_dict[block_idx]
    block_size = self.tracker.get_block_size(block_idx)

    # 5. FREE MEMORY FIRST (critical!)
    self.block_manager.free_gpu_memory(block_size, self.blocks_dict)

    # 6. LOAD BLOCK SECOND
    self.block_manager.move_block_to_gpu(block, block_idx)

    # 7. Update tracker
    self.tracker.current_executing_block = block_idx
    self.tracker.previous_block = block_idx

    # 8. Record memory state
    self.memory_monitor.record(
        blocks_on_gpu=len(self.tracker.gpu_blocks),
        current_block=block_idx
    )

    # 9. Optional: Prefetch next block if memory available
    if self.prefetch_enabled and block_idx + 1 < self.num_blocks:
        self.try_prefetch_block(block_idx + 1)
```

---

## 5. Phase-by-Phase Implementation

### Phase 1: Smart Initialization

**Goal:** Start with minimal GPU memory footprint.

**Strategy:**
1. Load full model to GPU (temporary)
2. Calculate each block's actual size
3. Immediately offload 55/60 blocks to CPU
4. Keep only 5 blocks on GPU (or 1 in aggressive mode)

**Implementation:**

```python
def initialize_offloading(self):
    """Phase 1: Proactive offloading initialization."""

    # Record initial state (all blocks on GPU)
    self.memory_monitor.record(blocks_on_gpu=self.num_blocks, current_block=-1)

    # Calculate block sizes while on GPU
    for i, block in enumerate(self.all_blocks):
        self.tracker.set_block_size(i, block)

    avg_size = self.tracker.average_block_size
    self.logger.info(f"Average block size: {avg_size:.2f}GB")

    # Decide which blocks to keep on GPU
    if self.stateless:
        # STATELESS: Offload ALL blocks
        blocks_to_keep_on_gpu = set()
        blocks_to_offload = set(range(self.num_blocks))
    else:
        # MINIMAL: Keep only first block
        blocks_to_keep_on_gpu = set(range(min(self.blocks_to_keep, 1)))
        blocks_to_offload = set(range(self.num_blocks)) - blocks_to_keep_on_gpu

    # Offload selected blocks
    for block_idx in sorted(blocks_to_offload):
        block = self.blocks_dict[block_idx]
        self.block_manager.move_block_to_cpu(block, block_idx)

    # Mark kept blocks as on GPU
    for block_idx in blocks_to_keep_on_gpu:
        self.tracker.mark_on_gpu(block_idx)

    # Aggressive cache clearing
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Record final state
    self.memory_monitor.record(blocks_on_gpu=len(blocks_to_keep_on_gpu),
                               current_block=-1)
```

**Memory Impact:**
- Before Phase 1: ~45GB reserved
- After Phase 1: ~24GB reserved (saves ~21GB)

### Phase 2: Dynamic Block Swapping

**Goal:** Load blocks on-demand during inference with minimal memory overhead.

**Integration Pattern:**

```python
def wrap_transformer_with_rabbit_video(model, rabbit_offloader, logger):
    """Intercept block forward passes to enable swapping."""

    stateless_mode = rabbit_offloader.stateless

    # Wrap double blocks
    for block_idx, block in enumerate(model.double_blocks):
        original_forward = block.forward

        def make_wrapped_forward(orig_forward, idx):
            @functools.wraps(orig_forward)
            def wrapped_forward(*args, **kwargs):
                # LOAD: Ensure block is on GPU
                rabbit_offloader.ensure_block_on_gpu(idx)

                # EXECUTE: Run original forward
                result = orig_forward(*args, **kwargs)

                # OFFLOAD: In stateless mode, immediately offload
                if stateless_mode:
                    rabbit_offloader.offload_block_after_execution(idx)

                # CLEAR CACHE: Aggressive cleanup
                rabbit_offloader.clear_cache_after_block()

                return result
            return wrapped_forward

        block.forward = make_wrapped_forward(original_forward, block_idx)

    # Wrap single blocks (same pattern, offset index)
    num_double_blocks = len(model.double_blocks)
    for block_idx, block in enumerate(model.single_blocks):
        original_forward = block.forward
        global_block_idx = num_double_blocks + block_idx
        # ... same wrapping logic with global_block_idx
```

**Key Insight:** Using closure (`make_wrapped_forward`) ensures each block captures its own index correctly.

**Execution Flow:**

```
Step 0:
  Block 0: ensure_on_gpu(0) → already on GPU → execute → keep on GPU
  Block 1: ensure_on_gpu(1) → load to GPU → execute → keep on GPU
  ...
  Block 5: ensure_on_gpu(5) → load, offload block 0 → execute
  Block 6: ensure_on_gpu(6) → load, offload block 1 → execute
  ...
```

### Phase 3: Auxiliary Model Management

**Goal:** Offload VAE and text encoders when they're not needed.

**Timeline:**

```
1. Text Encoding Phase:
   - Text encoders ON GPU
   - Encode prompts
   - Offload text encoders to CPU ← Phase 3a

2. Denoising Loop:
   - VAE ON CPU (not needed)    ← Phase 3b
   - Text encoders ON CPU
   - Transformer blocks swap dynamically (Phase 2)

3. VAE Decoding Phase:
   - Load VAE to GPU temporarily  ← Phase 3c
   - Decode latents
   - Offload VAE back to CPU
```

**Implementation (in pipeline):**

```python
# Phase 3a: After text encoding
if rabbit_mode:
    if self.text_encoder is not None:
        self.text_encoder.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if self.text_encoder_2 is not None:
        self.text_encoder_2.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Phase 3b: Before denoising loop
if rabbit_mode:
    if self.vae is not None:
        self.vae.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Denoising loop runs here with Phase 2 block swapping

# Phase 3c: VAE decoding
if rabbit_mode:
    # Load VAE temporarily
    if self.vae is not None:
        self.vae.to(device)
        torch.cuda.synchronize()

# Decode latents
with torch.autocast(...):
    image = self.vae.decode(latents)

if rabbit_mode:
    # Offload VAE back to CPU
    if self.vae is not None:
        self.vae.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

**Memory Impact:**
- Phase 3a: Saves ~12GB (text encoders)
- Phase 3b: Saves ~8GB (VAE)
- Total additional savings: ~20GB during denoising loop

---

## 6. Advanced: KV Cache Optimization

**Note:** This is an experimental feature added in version 1.1.

### 6.1 Motivation

**Observation:** During later denoising steps, some spatial regions become static and don't change much. Attention KV values for these regions can be cached and reused.

**Benefit:** Reduces redundant computation, potentially saves ~5-10% inference time.

**Risk:** Aggressive caching can degrade quality if static detection is too sensitive.

### 6.2 Static Region Detection

**Algorithm:**

```python
class StaticRegionDetector:
    def detect_static_regions(self, current_latent, timestep,
                             total_steps, block_idx):
        """Detect which spatial regions are static."""

        # Adaptive threshold based on progress
        progress = 1.0 - (timestep / total_steps)
        adaptive_threshold = self.base_threshold * (1.0 + progress**2 * 3.0)

        # Get previous latent from history
        if timestep not in self.latent_history:
            return torch.zeros_like(current_latent), 0.0

        prev_latent = self.latent_history[timestep]

        # Compute change map
        l2_change = torch.abs(current_latent - prev_latent)
        l2_norm = l2_change / (torch.abs(current_latent) +
                               torch.abs(prev_latent) + 1e-6)

        cos_sim = F.cosine_similarity(current_latent, prev_latent,
                                     dim=1, eps=1e-6)
        cos_change = 1.0 - cos_sim.unsqueeze(1)

        # Combined change metric
        change_map = 0.7 * l2_norm + 0.3 * cos_change

        # Create static mask with windowed averaging
        avg_pool = nn.AvgPool3d((1, 8, 8), stride=1, padding=(0, 4, 4))
        windowed_change = avg_pool(change_map)
        static_mask = (windowed_change < adaptive_threshold).float()

        # Morphological cleanup (erosion + dilation)
        static_mask = self._morphological_cleanup(static_mask)

        reuse_ratio = static_mask.float().mean().item()
        return static_mask, reuse_ratio
```

**Key Parameters:**
- `base_threshold=0.05`: Lower = more conservative (fewer static regions)
- `window_size=8`: Spatial coherence window
- Adaptive scaling: Threshold increases with denoising progress

### 6.3 Selective KV Cache

**Implementation:**

```python
class SelectiveKVCache:
    def get_cached_kv(self, block_idx, timestep, static_mask):
        """Retrieve cached KV if available."""
        if block_idx not in self.kv_cache:
            return None
        if timestep not in self.kv_cache[block_idx]:
            return None

        cache_entry = self.kv_cache[block_idx][timestep]
        return cache_entry['k'], cache_entry['v']

    def update_cache(self, block_idx, timestep, k, v, static_mask):
        """Store KV for static regions only."""

        # Only cache static regions to save memory
        static_ratio = static_mask.float().mean().item()
        kv_size_gb = self._estimate_kv_memory(k, v) * static_ratio

        # Check memory limit
        while self.memory_used_gb + kv_size_gb > self.max_cache_gb:
            if not self._evict_lru_entry():
                return  # Can't cache

        # Store detached copies
        if block_idx not in self.kv_cache:
            self.kv_cache[block_idx] = {}

        self.kv_cache[block_idx][timestep] = {
            'k': k.detach().clone(),
            'v': v.detach().clone(),
            'mask': static_mask.detach().clone(),
            'timestamp': time.time()
        }

        self.memory_used_gb += kv_size_gb
```

**Integration in Transformer Blocks:**

```python
# In MMDoubleStreamBlock and MMSingleStreamBlock forward()

# After computing Q, K, V
q, k, v = self.compute_qkv(...)

# Apply KV cache if available
if self.kv_cache_manager is not None:
    q, k, v = self.kv_cache_manager.process_kv_with_cache(
        self.block_idx, q, k, v,
        latent, timestep, total_steps
    )

# Continue with attention
attn = attention(q, k, v, ...)
```

**Conservative Strategy:**

```python
def process_kv_with_cache(self, block_idx, q, k, v):
    """Very conservative caching strategy."""

    # Only use cache if:
    # 1. Progress > 80% (final 20% of denoising)
    progress = 1.0 - (timestep / total_steps)
    if progress < 0.8:
        return q, k, v

    # 2. Block depth > 50% (deeper blocks more stable)
    block_depth_ratio = block_idx / self.num_blocks
    if block_depth_ratio < 0.5:
        return q, k, v

    # 3. Available memory > 3GB
    available = gpu_capacity - reserved
    if available < 3.0:
        return q, k, v

    # 4. Static ratio > 20%
    static_mask, reuse_ratio = self.static_detector.detect_static_regions(...)
    if reuse_ratio < 0.2:
        return q, k, v

    # Try to use cache
    cached_kv = self.kv_cache.get_cached_kv(block_idx, timestep, static_mask)
    if cached_kv is not None:
        k_cached, v_cached = cached_kv
        # Blend cached and new KV
        static_mask_expanded = static_mask.unsqueeze(-1).expand_as(k)
        k = k * (1 - static_mask_expanded) + k_cached * static_mask_expanded
        v = v * (1 - static_mask_expanded) + v_cached * static_mask_expanded

    return q, k, v
```

### 6.4 KV Cache Configuration

**Command-line flags:**

```bash
--rabbit-kv-cache                # Enable KV cache
--rabbit-kv-threshold 0.05       # Static detection threshold (lower = more conservative)
--rabbit-kv-max-gb 0.5          # Max memory for cache (GB)
```

**Recommendation:** Start with conservative settings and tune based on your model:
- `threshold=0.05` for high quality (fewer cached regions)
- `threshold=0.1` for more aggressive caching (faster, possible quality loss)
- `max_gb=0.5` should be safe for most 24GB GPUs

---

## 7. Integration Guide

### 7.1 Prerequisites

**Requirements for your model:**
1. Transformer-based architecture with sequential blocks
2. Blocks are functionally independent (no shared parameters)
3. Forward pass processes blocks sequentially (not parallel)
4. PyTorch-based implementation

**Model structures that work well:**
- DiT (Diffusion Transformer)
- U-ViT (U-Net Vision Transformer)
- Any sequential transformer blocks (encoder, decoder, or both)

### 7.2 Step-by-Step Integration

#### Step 1: Extract Blocks

Identify your model's transformer blocks:

```python
# Example: Your model structure
class YourDiffusionModel(nn.Module):
    def __init__(self):
        self.input_projection = ...
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(...) for _ in range(num_blocks)
        ])
        self.output_projection = ...
```

Extract blocks for RabbitVideo:

```python
all_blocks = list(model.transformer_blocks)
num_blocks = len(all_blocks)

# Create index mapping
blocks_dict = {i: block for i, block in enumerate(all_blocks)}
```

#### Step 2: Initialize RabbitVideoOffloader

```python
from rabbit_video import RabbitVideoOffloader

rabbit_offloader = RabbitVideoOffloader(
    model=model,
    device=torch.device('cuda'),
    blocks_to_keep=1,        # Start with 1 for minimal memory
    debug=True,              # Enable logging for first runs
    logger=logger,
    stateless=False          # Start with False, try True for absolute minimal memory
)

# Phase 1: Initialize offloading
rabbit_offloader.initialize_offloading()
```

#### Step 3: Wrap Block Forward Passes

Create wrapper function:

```python
import functools

def wrap_blocks_with_rabbit(model, rabbit_offloader, logger):
    """Wrap transformer blocks for RabbitVideo."""

    for block_idx, block in enumerate(model.transformer_blocks):
        original_forward = block.forward

        def make_wrapped_forward(orig_forward, idx):
            @functools.wraps(orig_forward)
            def wrapped_forward(*args, **kwargs):
                # Load block to GPU
                rabbit_offloader.ensure_block_on_gpu(idx)

                # Execute
                result = orig_forward(*args, **kwargs)

                # Optional: Stateless mode offload
                if rabbit_offloader.stateless:
                    rabbit_offloader.offload_block_after_execution(idx)

                # Clear cache
                rabbit_offloader.clear_cache_after_block()

                return result
            return wrapped_forward

        block.forward = make_wrapped_forward(original_forward, block_idx)

    logger.info(f"Wrapped {len(model.transformer_blocks)} blocks")

# Apply wrapper
wrap_blocks_with_rabbit(model, rabbit_offloader, logger)
```

#### Step 4: Add Phase 3 (Auxiliary Models)

Offload VAE and text encoders when idle:

```python
# After text encoding, before denoising
if text_encoder is not None:
    text_encoder.to('cpu')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

if vae is not None:
    vae.to('cpu')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Run denoising loop with block swapping

# Before VAE decoding
if vae is not None:
    vae.to(device)
    torch.cuda.synchronize()

# Decode
image = vae.decode(latents)

# After VAE decoding
if vae is not None:
    vae.to('cpu')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

#### Step 5: Add Memory Monitoring

```python
# Initialize memory monitor
from rabbit_video import MemoryMonitor

memory_monitor = MemoryMonitor(device, debug=True, logger=logger)

# Record at key points
memory_monitor.record(blocks_on_gpu=0, current_block=-1)  # Before inference

# During inference (inside denoising loop)
for i, timestep in enumerate(timesteps):
    memory_monitor.record(blocks_on_gpu=len(rabbit_offloader.tracker.gpu_blocks),
                         current_block=i)
    # ... run step

# After inference
memory_monitor.print_summary()
memory_monitor.save_timeline("memory_timeline.json")
```

### 7.3 Configuration Flags

Add command-line arguments:

```python
parser.add_argument("--rabbit-mode", action="store_true",
                    help="Enable RabbitVideo block-level offloading")
parser.add_argument("--rabbit-aggressive-offload", action="store_true",
                    help="Keep only 1 block on GPU (minimal memory)")
parser.add_argument("--rabbit-stateless", action="store_true",
                    help="Zero persistence mode (absolute minimal memory)")
parser.add_argument("--rabbit-debug", action="store_true",
                    help="Enable detailed memory logging")
parser.add_argument("--save-memory-timeline", type=str, default="",
                    help="Path to save memory timeline JSON")
```

Usage:

```bash
# Standard RabbitVideo (minimal memory, 1.15x time)
python your_script.py --rabbit-mode

# Aggressive mode (absolute minimal memory, 1.2x time)
python your_script.py --rabbit-mode --rabbit-aggressive-offload

# Stateless mode (lowest possible memory, 1.3x time)
python your_script.py --rabbit-mode --rabbit-stateless

# Debug mode with timeline
python your_script.py --rabbit-mode --rabbit-debug \
                      --save-memory-timeline timeline.json
```

### 7.4 Validation Checklist

After integration, verify:

1. **Memory usage:** Peak reserved < 25GB on 24GB GPU
2. **Correctness:** Output matches baseline (no quality degradation)
3. **Performance:** ~1.15-1.3x slowdown (acceptable range)
4. **Stability:** No OOM errors across multiple runs

**Debugging tips:**

```python
# Check memory at any point
alloc, reserved, cache = memory_monitor.get_current_memory()
print(f"Memory: {reserved:.2f}GB reserved, {alloc:.2f}GB allocated")

# Check block locations
print(f"GPU blocks: {rabbit_offloader.tracker.gpu_blocks}")
print(f"CPU blocks: {rabbit_offloader.tracker.cpu_blocks}")

# Check transfer statistics
rabbit_offloader.tracker.print_statistics()
```

---

## 8. Performance Characteristics

### 8.1 Memory Reduction

**HunyuanVideo benchmarks (720p, 40 steps, RTX 4090 24GB):**

| Method | Peak Reserved | Peak Allocated | Reduction |
|--------|--------------|----------------|-----------|
| Baseline | 66.2 GB | 38.4 GB | - |
| RabbitVideo (Phase 1-2) | 24.3 GB | 20.7 GB | 63% |
| RabbitVideo (Phase 1-2-3) | 23.0 GB | 19.0 GB | 65% |
| RabbitVideo + Stateless | 19.0 GB | 16.0 GB | 71% |

**Breakdown by phase:**
- Phase 1 (initialization): Saves ~21GB
- Phase 2 (block swapping): Maintains low memory
- Phase 3 (auxiliary offload): Saves additional ~12GB

### 8.2 Time Overhead

**Transfer analysis:**

```
Average block size: 648 MB
PCIe 3.0 x16 bandwidth: ~16 GB/s
Transfer time per block: ~40ms

Blocks per step: 60 blocks
Naive cost: 60 × 40ms = 2.4s per step
Actual cost: ~0.5s per step (LRU caching reduces transfers)

Total overhead: 50 steps × 0.5s = 25s (for 200s baseline = 1.125x)
```

**Measured performance:**

| Configuration | Time (s) | Slowdown | Use Case |
|--------------|----------|----------|----------|
| Baseline | 185 | 1.00x | ≥48GB GPU |
| CPU Offload | 487 | 2.63x | Not recommended |
| RabbitVideo | 210 | 1.14x | **Recommended for 24GB GPU** |
| RabbitVideo + Stateless | 240 | 1.30x | Extreme memory constraint |

### 8.3 Scaling Analysis

**Block count impact:**

```python
# Transfer time scales linearly with number of blocks
overhead_per_block = 40ms  # CPU↔GPU transfer
overhead_per_step = num_blocks × overhead_per_block × reuse_factor

# Reuse factor: How many blocks are reused (LRU caching)
# Typical: 0.1-0.2 (10-20% of blocks reused from previous step)
```

**Example scaling:**

| Blocks | Avg Transfer/Step | Overhead (50 steps) | Relative |
|--------|------------------|---------------------|----------|
| 30 | 200ms | 10s | 0.5x |
| 60 | 400ms | 20s | 1.0x |
| 120 | 800ms | 40s | 2.0x |

**Conclusion:** RabbitVideo scales well up to ~100 blocks. Beyond that, consider model quantization or other optimizations.

### 8.4 Cache Clearing Impact

**Why aggressive cache clearing matters:**

```
Without aggressive clearing:
  Reserved memory: 45GB (PyTorch hoards cache)

With synchronous clearing:
  Reserved memory: 24GB (cache actually freed)

Difference: 21GB (46% reduction in reserved memory)
```

**Protocol effectiveness:**

```python
# Ineffective (async, cache not freed)
block.to('cpu')
torch.cuda.empty_cache()

# Effective (sync, cache freed)
block.to('cpu')
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.synchronize()
torch.cuda.empty_cache()  # Double clear
```

### 8.5 KV Cache Performance (v1.1)

**Experimental results:**

| Configuration | Time Saving | Quality Impact | Recommendation |
|--------------|-------------|----------------|----------------|
| No KV cache | - | Baseline | Default |
| KV cache (threshold=0.05) | ~5% | Minimal | Conservative |
| KV cache (threshold=0.1) | ~8% | Slight | Moderate |
| KV cache (threshold=0.2) | ~12% | Noticeable | Aggressive |

**Note:** KV cache is experimental and may introduce quality artifacts. Use conservatively.

---

## 9. Limitations and Constraints

### 9.1 Known Limitations

#### 1. **Not Compatible with Distributed Inference**

```python
# This will FAIL:
python inference.py --rabbit-mode --ulysses-degree 2

# Error: RabbitVideo cannot be used with distributed inference
```

**Reason:** Sequence parallelism requires all blocks on GPU simultaneously.

**Workaround:** Use distributed inference OR RabbitVideo, not both.

#### 2. **Cannot Combine with Model-Level CPU Offload**

```python
# This will FAIL:
python inference.py --rabbit-mode --use-cpu-offload

# Error: Cannot use both --rabbit-mode and --use-cpu-offload
```

**Reason:** Both systems manage the same resources, conflict inevitable.

#### 3. **10-30% Time Overhead**

**Trade-off:** Memory efficiency comes at performance cost.

**Mitigation:**
- Use standard mode (1.15x) for balanced performance
- Reserve stateless mode for extreme memory constraints

#### 4. **Requires Sequential Block Execution**

**Incompatible architectures:**
- Parallel block execution (e.g., mixture-of-experts with parallel dispatch)
- Blocks with shared parameters (would need simultaneous GPU presence)
- Non-transformer architectures without clear block boundaries

### 9.2 Edge Cases and Handling

#### Case 1: Very Large Blocks (>2GB)

**Problem:** PCIe transfer becomes bottleneck.

**Solution:**
```python
# Option 1: Reduce model precision
model = model.half()  # FP16 halves block size

# Option 2: Increase blocks_to_keep
rabbit_offloader = RabbitVideoOffloader(..., blocks_to_keep=3)
```

#### Case 2: High-Frequency Block Reuse

**Problem:** Some architectures revisit blocks (e.g., recurrent patterns).

**Detection:**
```python
# Check reuse ratio
reuse_ratio = rabbit_offloader.tracker.block_access_count[0] / total_steps
if reuse_ratio > 2.0:
    print("WARNING: High block reuse detected, consider increasing blocks_to_keep")
```

**Solution:** Keep frequently-used blocks on GPU:
```python
# Identify hot blocks
hot_blocks = [i for i, count in enumerate(access_counts) if count > threshold]

# Modify LRU to never evict hot blocks
def select_blocks_to_offload(self, num_needed):
    cold_blocks = [b for b in self.gpu_blocks if b not in hot_blocks]
    return sorted(cold_blocks, key=...)[:num_needed]
```

#### Case 3: Extremely Limited VRAM (<16GB)

**Problem:** Even Phase 3 insufficient.

**Additional strategies:**
1. **Reduce batch size:** `--batch-size 1`
2. **Reduce resolution:** `--video-size 360 640`
3. **Gradient checkpointing:** (training only)
4. **Model quantization:** INT8 weights
5. **Tiled VAE:** Decode in spatial tiles

```python
# Example: Tiled VAE decoding
def decode_tiled(vae, latents, tile_size=64):
    B, C, T, H, W = latents.shape
    output = torch.zeros(B, 3, T, H*8, W*8)  # VAE upsamples 8x

    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            tile = latents[:, :, :, i:i+tile_size, j:j+tile_size]
            vae.to('cuda')
            decoded_tile = vae.decode(tile)
            vae.to('cpu')
            output[:, :, :, i*8:(i+tile_size)*8, j*8:(j+tile_size)*8] = decoded_tile
            torch.cuda.empty_cache()

    return output
```

### 9.3 Quality Considerations

#### 1. **Numerical Stability**

**Issue:** Frequent CPU↔GPU transfers can introduce floating-point errors.

**Mitigation:**
- Use FP32 for block parameters (default)
- Only use FP16/BF16 for activations
- Validate output quality against baseline

```python
# Quality check
baseline_output = run_inference(model, rabbit_mode=False)
rabbit_output = run_inference(model, rabbit_mode=True)

mse = torch.mean((baseline_output - rabbit_output)**2)
assert mse < 1e-4, f"Quality degradation: MSE={mse}"
```

#### 2. **KV Cache Artifacts**

**Issue:** Aggressive KV caching can cause temporal inconsistencies.

**Detection:**
- Visual inspection for "sticky" regions
- Frame-to-frame consistency metrics

**Prevention:**
- Start with conservative threshold (0.05)
- Disable for high-quality production renders
- Use only for draft/preview generation

### 9.4 Hardware Dependencies

**Optimal hardware:**
- **GPU:** RTX 4090 24GB (tested)
- **PCIe:** Gen 3.0 x16 or better (16 GB/s bandwidth)
- **RAM:** ≥64GB system memory (for offloaded blocks)
- **Storage:** Not critical (no disk swapping)

**Suboptimal scenarios:**
- PCIe Gen 2.0: 2x slower transfers (40ms → 80ms per block)
- <32GB RAM: Risk of system swap, catastrophic slowdown
- Shared PCIe lanes: Bandwidth contention

---

## 10. Code Reference

### 10.1 File Structure

```
hyvideo/
├── rabbit_video.py              # Core implementation (715 lines)
│   ├── MemoryMonitor            # Lines 18-108
│   ├── BlockTracker             # Lines 111-210
│   ├── BlockManager             # Lines 213-345
│   ├── RabbitVideoOffloader     # Lines 348-715
│   └── [v1.1] KV Cache classes  # Lines 932-1272
│
├── inference.py                 # Integration logic
│   ├── wrap_transformer_with_rabbit_video()  # Lines 45-95
│   ├── Inference.from_pretrained()           # Lines 380-450
│   └── HunyuanVideoSampler                   # Lines 520-600
│
├── config.py                    # Command-line args
│   └── RabbitVideo flags        # Lines 266-290
│
└── diffusion/pipelines/pipeline_hunyuan_video.py
    └── Phase 3 integration      # Lines 900-1150

visualize_rabbit_timeline.py    # Visualization tools (453 lines)
RABBIT_VIDEO.md                  # User documentation
VISUALIZATION_GUIDE.md           # Visualization guide
```

### 10.2 Key Function Reference

#### Core Algorithm Functions

```python
# Phase 1: Initialization
RabbitVideoOffloader.initialize_offloading()
  └─ BlockManager.move_block_to_cpu(block, idx)
      └─ Synchronous offload protocol

# Phase 2: Dynamic swapping
RabbitVideoOffloader.ensure_block_on_gpu(idx)
  ├─ BlockTracker.record_access(idx)
  ├─ BlockManager.free_gpu_memory(size_gb, blocks_dict)
  │   └─ BlockTracker.select_blocks_to_offload(n)
  │       └─ LRU policy
  ├─ BlockManager.move_block_to_gpu(block, idx)
  └─ RabbitVideoOffloader.try_prefetch_block(idx+1)

# Phase 3: Auxiliary management
offload_auxiliary_models_to_cpu(vae, text_encoder, ...)
temporarily_move_to_gpu(model, device, operation_name)

# Monitoring
MemoryMonitor.record(blocks_on_gpu, current_block)
MemoryMonitor.save_timeline(filepath)
```

#### Integration Points

```python
# Initialization (inference.py:380-450)
if args.rabbit_mode:
    rabbit_offloader = RabbitVideoOffloader(...)
    rabbit_offloader.initialize_offloading()
    wrap_transformer_with_rabbit_video(model, rabbit_offloader, logger)

# Block wrapping (inference.py:45-95)
def wrap_transformer_with_rabbit_video(model, offloader, logger):
    for block_idx, block in enumerate(model.double_blocks):
        original_forward = block.forward
        def wrapped_forward(*args, **kwargs):
            offloader.ensure_block_on_gpu(block_idx)
            result = original_forward(*args, **kwargs)
            offloader.clear_cache_after_block()
            return result
        block.forward = wrapped_forward

# Pipeline integration (pipeline_hunyuan_video.py:900-1150)
# - Text encoder offload after encoding
# - VAE offload before denoising
# - VAE load/offload around decoding
# - Memory timeline recording
```

### 10.3 Configuration Reference

```python
# Command-line flags
--rabbit-mode                    # Enable RabbitVideo
--rabbit-aggressive-offload      # Keep only 1 block on GPU
--rabbit-stateless              # Zero persistence mode
--rabbit-debug                  # Detailed logging
--save-memory-timeline PATH     # Save timeline JSON

# [v1.1] KV cache flags
--rabbit-kv-cache               # Enable KV cache
--rabbit-kv-threshold 0.05      # Static detection threshold
--rabbit-kv-max-gb 0.5         # Max cache memory (GB)

# Initialization parameters
RabbitVideoOffloader(
    model: nn.Module,            # Transformer model
    device: torch.device,        # CUDA device
    blocks_to_keep: int = 1,     # Blocks to keep on GPU
    debug: bool = False,         # Debug logging
    logger = None,               # Loguru logger
    stateless: bool = False,     # Stateless mode
    # [v1.1] KV cache
    enable_kv_cache: bool = False,
    kv_cache_threshold: float = 0.05,
    kv_cache_max_memory_gb: float = 0.5
)
```

---

## Conclusion

RabbitVideo demonstrates that careful system design can achieve dramatic memory reduction (65%) with acceptable performance cost (15% slowdown) by exploiting:

1. **Architectural insight:** Sequential block execution enables dynamic offloading
2. **Engineering rigor:** Synchronous transfer protocol ensures memory is actually freed
3. **Holistic optimization:** Three-phase approach addresses all memory consumers

**Key takeaways for adaptation:**

- **Block-level granularity is optimal** for transformer-based models
- **Proactive memory management** prevents OOM better than reactive strategies
- **Synchronous operations** are critical for actually freeing PyTorch cache
- **Phase 3 matters:** Don't forget to offload auxiliary models

**Future directions:**

1. **Model-agnostic wrapper:** Abstract pattern for any transformer
2. **Adaptive block retention:** ML-based prediction of reuse patterns
3. **Hybrid CPU/GPU cache:** Use both for larger effective cache
4. **Compression:** Compress offloaded blocks in CPU memory

This report provides sufficient detail to implement RabbitVideo on other large-scale diffusion models. The core principles (block-level offloading, synchronous transfers, LRU eviction, auxiliary management) are transferable to any sequential transformer architecture.

---

**Document Version:** 1.1
**Last Updated:** November 2025
**Implementation Version:** Commits e6de3f4 through 7e72742 on rabbit-video-final branch
**Contact:** ahengljh@gmail.com
