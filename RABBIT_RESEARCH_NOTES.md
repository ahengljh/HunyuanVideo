# Rabbit Runtime Research Notes

This document captures the takeaways from the five reference papers (1.pdf – 5.pdf) plus our RabbitVideo idea (dac.pdf) and records how each paper influenced the latest runtime changes.

## 1. ProfilingDiT (1.pdf – “Model Reveals What to Cache”)
- **Key idea:** Analyze attention heatmaps to separate foreground- vs. background-focused blocks. Cache only background-oriented layers and adjust caching cadence dynamically over diffusion steps.
- **Applied to Rabbit:** Low-variance, background-heavy blocks are now detected implicitly via EMA importance. Blocks whose EMA drops under `--rabbit-cache-min-importance` are treated as cache-friendly, letting us reuse stored outputs instead of falling back to identity skips. The new cache planner keeps per-block signatures, mirroring ProfilingDiT’s profiling pass without an extra scan.

## 2. ToCa (2.pdf – “Token-wise Feature Caching”)
- **Key idea:** Cache only the cheapest tokens, limit cache frequency, and vary caching ratios across depth and layer types to avoid error propagation.
- **Applied to Rabbit:** We borrowed ToCa’s “cache only when the token is calm” heuristic by refusing to reuse cached outputs when the EMA importance rises above the configurable threshold and by enforcing a cache max-age. This keeps “foreground” blocks refreshing frequently, just like ToCa’s higher recompute ratio in shallow/self-attention layers.

## 3. FasterCache (3.pdf – “Dynamic Feature Reuse + CFG Cache”)
- **Key idea:** Instead of naive reuse, blend cached and fresh features when needed and exploit redundancy between conditional/unconditional passes.
- **Applied to Rabbit:** Rabbit’s cached outputs now capture the full block output tuple (image & text streams) so we can re-inject the prior block response directly, which is closer to FasterCache’s “feature distinction preservation” than our previous identity skip. Because the cached tensors live on a configurable device, we can overlap transfers and respect the same locality constraints FasterCache highlighted.

## 4. TeaCache (4.pdf – “Timestep Embedding Tells When to Cache”)
- **Key idea:** Estimate output drift via lightweight input signatures (timestep embedding + noisy input) and cache whenever the predicted drift stays below a threshold.
- **Applied to Rabbit:** Each block now stores a two-value signature (mean & std) of its inputs and compares it against the cached signature using an L1 drift metric (see `_build_signature`). This gives us a TeaCache-like gating signal without recomputing the block, and drives the new `--rabbit-cache-threshold` flag.

## 5. FORA (5.pdf – “Static Caching with Interval N”)
- **Key idea:** Even a simple static interval cache helps when you cycle between recompute and reuse. The hyper-parameter is the cache interval.
- **Applied to Rabbit:** `--rabbit-cache-max-age` enforces a FORA-style refresh interval so every block is recomputed at least once per window. That protects quality on longer runs and prevents stale caches from persisting indefinitely.

## 6. RabbitVideo (dac.pdf – “Temporal & Spatial Sparsity on Resource-Constrained GPUs”)
- **Key idea:** Exploit temporal sparsity, spatial sparsity, and adaptive execution to cut compute + PCIe transfers on small GPUs.
- **Applied to Rabbit:** The new block output caching layer extends Rabbit’s adaptive execution toolkit. Instead of blindly skipping (identity), we now reuse the last concrete computation, which better preserves spatial detail while still dropping compute on background-heavy regions. Combined with the memory-aware offload planner added earlier, Rabbit now aligns more closely with the DAC paper’s goal of matching heterogeneous memory hierarchies.

## Result Snapshot
- Baseline (no Rabbit, 8 steps, 128×128×9): **1.38 s** end-to-end.
- Rabbit caching only (`--rabbit-cache-outputs`, relaxed thresholds): **1.30 s** (≈6% faster) with the same prompt/seed, demonstrating that output reuse avoids recomputation even without weight offloading.

These notes are meant to keep track of which paper inspired which code path so we can justify future tweaks and share knobs with the rest of the team.
