# Rabbit Runtime Methodology & Evaluation Blueprint

This note distills the lessons from the five reference papers (ProfilingDiT, ToCa, FasterCache, TeaCache, FORA) together with the RabbitVideo DAC paper, and explains how the current Rabbit runtime implements those ideas. It also lays out the evaluation suite we should run before claiming improvements.

---

## 1. High-Level Architecture

```mermaid
flowchart LR
    subgraph GPU
        A[Transformer block\n currently executing]
        C[Latent tensors\n (active step)]
    end

    subgraph CPU
        B[Offloaded weights\n (auto-plan tail)]
        D[Latent cache\n previous steps]
        E[Block output cache\n(low-variance blocks)]
    end

    A -- weight prefetch --> B
    B -- stream to GPU --> A
    A -- importance / skip stats --> E
    E -- cached outputs --> A
    A -- results --> C
    C -- offload after step --> D
    D -- bring next step latents --> C
```

**What happens each diffusion step**
1. **Weight streaming** – the offload planner keeps critical blocks resident; tail blocks are prefetched from CPU (`--rabbit-offload-plan`, `--rabbit-memory-budget-mb`, `--rabbit-prefetch-distance`).
2. **Output caching** – low-EMA blocks record a (mean, std) signature of their inputs; if the drift stays below `--rabbit-cache-threshold`, the cached outputs are re-used (ProfilingDiT + TeaCache inspiration).
3. **Latent ping-pong** – denoised latents are moved back to CPU after each step when `--rabbit-latent-offload` is set.
4. **Adaptive skipping** – still available (`--rabbit-skip-strategy ema`), but caching usually gives higher visual fidelity than identity skips.

---

## 2. Mapping papers → Rabbit

| Paper | Key mechanism | Rabbit feature |
|-------|---------------|----------------|
| ProfilingDiT | Foreground/background profiling | `cache_min_importance`, block-level EMA gates reuse |
| ToCa | Token-wise ratios & cache cooldown | `cache_max_age`, stage filters (`cache_stage`) |
| FasterCache | Dynamic feature reuse across CFG | Full block output cached instead of raw inputs |
| TeaCache | Input signature predicts drift | `(mean, std)` signature & `cache_threshold` |
| FORA | Static interval refresh | `cache_max_age` (minimum recompute window) |
| RabbitVideo (DAC) | Temporal & spatial sparsity on consumer GPUs | Combined weight streaming + latent offload + caching |

The detailed line references live in `RABBIT_RESEARCH_NOTES.md`.

---

## 3. Methodology Text (draft for paper / README)

> *Rabbit Runtime* orchestrates three complementary mechanisms to fit large video diffusion transformers onto commodity GPUs:
>
> 1. **Weight locality planner** – we partition the DiT blocks into resident and streaming sets. Low-importance tail blocks are moved to host memory and prefetched with `prefetch_distance`. The planner honors a user-specified device budget (`rabbit_memory_budget_mb`) to keep overall VRAM within limits. With the ProfilingDiT-style warmup (`--rabbit-profile-steps`), we score each block’s importance first and bias the offload plan toward background-heavy layers automatically.
> 2. **Latent ping-pong** – denoised latents are transferred back to the host after each diffusion step to free VRAM while maintaining smooth progress. Pinned memory is optional to overlap PCIe transfers.
> 3. **Output caching** – blocks with low EMA importance capture their output tensors (both image and text streams) and a lightweight signature. When the signature drift stays below `rabbit_cache_threshold`, we re-inject the cached result instead of recomputing, avoiding quality loss that arises from identity skips. Token-level caching à la ToCa is available via `--rabbit-cache-token-ratio`, which reuses cached activations only for low-variance tokens while letting high-variance tokens fall back to the identity path.
> 4. **CFG branch caching** – when classifier-free guidance is enabled, `--rabbit-cfg-reuse-interval` lets Rabbit reuse the unconditional branch across multiple timesteps (FasterCache-style) by re-evaluating only the conditional branch between refreshes.
>
> The runtime exposes metrics (executed vs. cached vs. skipped blocks, cache hit rate) so users can tune aggressiveness interactively.

Feel free to lift this paragraph into the methodology section of the paper.

---

## 4. Evaluation Suite (what to measure)

**Core metrics**
1. **Wall-clock latency** – extracted from `hyvideo.inference:predict` log.
2. **Peak VRAM** – collect via `nvidia-smi --query-gpu=memory.used`.
3. **Power or PCIe bandwidth (optional)** – if running on systems with telemetry.

**Quality metrics** (compare Rabbit vs. vanilla)
1. **VBench score** – the Open-Sora benchmark; covers motion, structure, temporal coherence.
2. **LPIPS / PSNR / SSIM** – frame-wise comparison against the baseline video (same seed).
3. **CLIP similarity** – text-video alignment.

**Ablations**
1. Baseline (no Rabbit).
2. Rabbit with only weight offloading.
3. Rabbit with only latent offloading.
4. Rabbit with only output caching.
5. Full Rabbit (offload + cache + latent).

Run the full evaluation for two representative prompts (one high-motion, one static scene) at the canonical resolution (720×1280, 129 frames, 40 flow steps, prompt template `dit-llm-encode-video`). For tuning dashboards, the `scripts/rabbit_tune.sh` helper sweeps the caching thresholds on demand.

**Optional additions from the papers**
- Borrow ProfilingDiT’s attention heat-map sanity check: visualize which blocks remain on device vs. offload.
- Report caching ratios per layer depth as ToCa does (already logged with `cache_hits`).
- Present a latency breakdown similar to FasterCache (attention vs. MLP vs. rest).

---

## 5. Next Steps

1. Automate the evaluation pipeline (baseline + Rabbit variants) with a simple driver script.
2. Add VBench/CLIP metric scripts for quantitative quality.
3. Export cache/offload telemetry as CSV so we can plot compute saved vs. cache threshold.
4. Consider an additional diagram that shows per-step data flow (weights ↔ GPU ↔ CPU) for the paper’s methodology figure.

Once we run the evaluation grid and capture the metrics above, we can fill the results section with clear “Rabbit vs. baseline” tables and charts.
