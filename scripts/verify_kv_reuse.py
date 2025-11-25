"""
KV-Cache Reuse Verification Script for HunyuanVideo

This script verifies the hypothesis that certain spatial regions in video generation
have stable KV representations across denoising steps, making them candidates for
KV-cache reuse optimization.

Key Metrics:
1. Per-token cosine similarity between consecutive steps
2. Spatial heatmaps showing which regions have stable KV
3. Per-block analysis to identify which layers show more stability

Usage:
    python scripts/verify_kv_reuse.py --model-base /path/to/model --prompt "a static scene"
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from collections import defaultdict
from loguru import logger
import json
import argparse

# Must import before monkey-patching
import hyvideo.modules.models
from hyvideo.modules.attenion import attention as original_attention
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


class KVTracker:
    """Tracks KV pairs across denoising steps for analysis."""

    def __init__(self, save_full_kv=False, max_tokens_to_track=None):
        """
        Args:
            save_full_kv: If True, save full KV tensors (memory intensive).
                         If False, only save statistics and samples.
            max_tokens_to_track: Limit tokens tracked to save memory. None = all.
        """
        self.save_full_kv = save_full_kv
        self.max_tokens_to_track = max_tokens_to_track

        self.current_step = 0
        self.block_counter = 0

        # Storage: step -> block -> {'k': tensor, 'v': tensor, 'img_len': int}
        self.kv_data = defaultdict(dict)

        # For analysis results
        self.similarity_results = []

        # Metadata
        self.img_seq_len = None
        self.txt_seq_len = None
        self.spatial_shape = None  # (T, H, W) after patchification

    def reset_for_step(self, step):
        """Reset counters for a new denoising step."""
        self.current_step = step
        self.block_counter = 0

    def capture_kv(self, k, v, img_len=None):
        """
        Capture K and V tensors for analysis.

        Args:
            k: Key tensor [B, S, H, D] where S = img_seq_len + txt_seq_len
            v: Value tensor [B, S, H, D]
            img_len: Length of image tokens (to separate from text)
        """
        block_idx = self.block_counter
        self.block_counter += 1

        # Detach and move to CPU to avoid GPU OOM
        k_cpu = k.detach().float().cpu()
        v_cpu = v.detach().float().cpu()

        # Store image tokens only (text tokens are less interesting for spatial analysis)
        if img_len is not None:
            k_img = k_cpu[:, :img_len]
            v_img = v_cpu[:, :img_len]
        else:
            k_img = k_cpu
            v_img = v_cpu

        # Optionally limit tokens to save memory
        if self.max_tokens_to_track is not None:
            k_img = k_img[:, :self.max_tokens_to_track]
            v_img = v_img[:, :self.max_tokens_to_track]

        if self.save_full_kv:
            self.kv_data[self.current_step][block_idx] = {
                'k': k_img,
                'v': v_img,
                'img_len': img_len
            }
        else:
            # Save only statistics to save memory
            # Per-token L2 norm (useful for identifying active regions)
            k_norm = torch.norm(k_img, dim=-1).mean(dim=(0, 2))  # [S]
            v_norm = torch.norm(v_img, dim=-1).mean(dim=(0, 2))  # [S]

            self.kv_data[self.current_step][block_idx] = {
                'k': k_img,  # Still save for similarity computation
                'v': v_img,
                'k_norm': k_norm,
                'v_norm': v_norm,
                'img_len': img_len
            }

    def compute_similarity_matrix(self):
        """
        Compute cosine similarity of KV between consecutive steps.

        Returns:
            dict: block_idx -> {
                'k_sim_per_token': [num_steps-1, num_tokens] array,
                'v_sim_per_token': [num_steps-1, num_tokens] array,
                'k_sim_mean': [num_steps-1] array,
                'v_sim_mean': [num_steps-1] array,
            }
        """
        results = {}
        steps = sorted(self.kv_data.keys())

        if len(steps) < 2:
            logger.warning("Need at least 2 steps to compute similarity")
            return results

        # Get all block indices
        block_indices = set()
        for step_data in self.kv_data.values():
            block_indices.update(step_data.keys())
        block_indices = sorted(block_indices)

        for block_idx in block_indices:
            k_sims_per_token = []
            v_sims_per_token = []
            k_sims_mean = []
            v_sims_mean = []

            for i in range(len(steps) - 1):
                step_a, step_b = steps[i], steps[i + 1]

                if block_idx not in self.kv_data[step_a] or block_idx not in self.kv_data[step_b]:
                    continue

                k_a = self.kv_data[step_a][block_idx]['k']  # [B, S, H, D]
                k_b = self.kv_data[step_b][block_idx]['k']
                v_a = self.kv_data[step_a][block_idx]['v']
                v_b = self.kv_data[step_b][block_idx]['v']

                # Compute per-token cosine similarity (average over batch and heads)
                # Flatten heads and head_dim: [B, S, H*D]
                k_a_flat = k_a.reshape(k_a.shape[0], k_a.shape[1], -1)
                k_b_flat = k_b.reshape(k_b.shape[0], k_b.shape[1], -1)
                v_a_flat = v_a.reshape(v_a.shape[0], v_a.shape[1], -1)
                v_b_flat = v_b.reshape(v_b.shape[0], v_b.shape[1], -1)

                # Cosine similarity per token
                k_sim = torch.nn.functional.cosine_similarity(k_a_flat, k_b_flat, dim=-1)  # [B, S]
                v_sim = torch.nn.functional.cosine_similarity(v_a_flat, v_b_flat, dim=-1)  # [B, S]

                # Average over batch
                k_sim_avg = k_sim.mean(dim=0).numpy()  # [S]
                v_sim_avg = v_sim.mean(dim=0).numpy()  # [S]

                k_sims_per_token.append(k_sim_avg)
                v_sims_per_token.append(v_sim_avg)
                k_sims_mean.append(k_sim_avg.mean())
                v_sims_mean.append(v_sim_avg.mean())

            if k_sims_per_token:
                results[block_idx] = {
                    'k_sim_per_token': np.stack(k_sims_per_token),  # [num_steps-1, S]
                    'v_sim_per_token': np.stack(v_sims_per_token),
                    'k_sim_mean': np.array(k_sims_mean),
                    'v_sim_mean': np.array(v_sims_mean),
                }

        return results

    def analyze_spatial_stability(self, spatial_shape):
        """
        Analyze which spatial positions show stable KV across steps.

        Args:
            spatial_shape: (T, H, W) - temporal frames, height patches, width patches

        Returns:
            dict with spatial analysis results
        """
        self.spatial_shape = spatial_shape
        T, H, W = spatial_shape
        expected_tokens = T * H * W

        similarity_results = self.compute_similarity_matrix()

        spatial_analysis = {}

        for block_idx, sim_data in similarity_results.items():
            k_sim = sim_data['k_sim_per_token']  # [num_steps-1, S]
            v_sim = sim_data['v_sim_per_token']

            num_tokens = k_sim.shape[1]

            if num_tokens != expected_tokens:
                logger.warning(f"Block {block_idx}: token count {num_tokens} != expected {expected_tokens}")
                # Try to use what we have
                usable_tokens = min(num_tokens, expected_tokens)
            else:
                usable_tokens = expected_tokens

            # Average similarity across all step transitions
            k_sim_avg = k_sim[:, :usable_tokens].mean(axis=0)  # [S]
            v_sim_avg = v_sim[:, :usable_tokens].mean(axis=0)

            # Reshape to spatial dimensions
            try:
                k_spatial = k_sim_avg[:T*H*W].reshape(T, H, W)
                v_spatial = v_sim_avg[:T*H*W].reshape(T, H, W)
            except ValueError:
                logger.warning(f"Block {block_idx}: Cannot reshape {usable_tokens} tokens to {spatial_shape}")
                continue

            # Identify high-stability regions (e.g., > 0.9 cosine similarity)
            k_stable_mask = k_spatial > 0.9
            v_stable_mask = v_spatial > 0.9

            k_stable_ratio = k_stable_mask.sum() / k_stable_mask.size
            v_stable_ratio = v_stable_mask.sum() / v_stable_mask.size

            spatial_analysis[block_idx] = {
                'k_spatial_sim': k_spatial,
                'v_spatial_sim': v_spatial,
                'k_stable_ratio': float(k_stable_ratio),
                'v_stable_ratio': float(v_stable_ratio),
                'k_mean_sim': float(k_sim_avg.mean()),
                'v_mean_sim': float(v_sim_avg.mean()),
            }

        return spatial_analysis

    def print_summary(self):
        """Print a summary of the KV stability analysis."""
        similarity_results = self.compute_similarity_matrix()

        print("\n" + "="*70)
        print("KV-CACHE REUSE ANALYSIS SUMMARY")
        print("="*70)

        if not similarity_results:
            print("No data captured. Check if the model ran correctly.")
            return

        print(f"\nCaptured {len(self.kv_data)} denoising steps")
        print(f"Analyzed {len(similarity_results)} attention blocks\n")

        # Group blocks by type (first 20 are double blocks, rest are single blocks)
        double_blocks = {k: v for k, v in similarity_results.items() if k < 20}
        single_blocks = {k: v for k, v in similarity_results.items() if k >= 20}

        def print_block_stats(blocks, name):
            if not blocks:
                return
            print(f"\n{name}:")
            print("-" * 50)

            all_k_sims = []
            all_v_sims = []

            for block_idx in sorted(blocks.keys()):
                data = blocks[block_idx]
                k_mean = data['k_sim_mean'].mean()
                v_mean = data['v_sim_mean'].mean()
                all_k_sims.append(k_mean)
                all_v_sims.append(v_mean)

                # Show per-step progression for first few blocks
                if block_idx < 5 or block_idx == 20:
                    k_per_step = ", ".join([f"{s:.3f}" for s in data['k_sim_mean']])
                    print(f"  Block {block_idx:2d}: K_sim={k_mean:.4f}, V_sim={v_mean:.4f}")
                    print(f"            K per step: [{k_per_step}]")

            avg_k = np.mean(all_k_sims)
            avg_v = np.mean(all_v_sims)
            print(f"\n  Average across all {name}: K_sim={avg_k:.4f}, V_sim={avg_v:.4f}")

        print_block_stats(double_blocks, "Double Stream Blocks (0-19)")
        print_block_stats(single_blocks, "Single Stream Blocks (20+)")

        # Overall assessment
        all_k_sims = [v['k_sim_mean'].mean() for v in similarity_results.values()]
        all_v_sims = [v['v_sim_mean'].mean() for v in similarity_results.values()]

        overall_k = np.mean(all_k_sims)
        overall_v = np.mean(all_v_sims)

        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        print(f"\nOverall average KV similarity: K={overall_k:.4f}, V={overall_v:.4f}")

        if overall_k > 0.8 or overall_v > 0.8:
            print("\n✓ HIGH POTENTIAL for KV-cache reuse!")
            print("  The KV representations show strong correlation across steps.")
            print("  Consider implementing selective KV caching for stable regions.")
        elif overall_k > 0.5 or overall_v > 0.5:
            print("\n◐ MODERATE POTENTIAL for KV-cache reuse.")
            print("  Some correlation exists. May benefit from adaptive caching")
            print("  that identifies and reuses only the most stable tokens.")
        else:
            print("\n✗ LOW POTENTIAL for naive KV-cache reuse.")
            print("  KV representations change significantly across steps.")
            print("  Consider alternative approaches like KV interpolation or")
            print("  caching only specific layers/tokens with high stability.")

        return similarity_results


# Global tracker instance
tracker = KVTracker(save_full_kv=False, max_tokens_to_track=2000)


def patched_attention(q, k, v, mode="flash", *args, **kwargs):
    """Monkey-patched attention function to capture KV pairs."""
    # Capture KV before attention computation
    # k, v shape depends on mode, but before layout transform: [B, S, H, D]
    tracker.capture_kv(k, v)

    # Call original attention
    return original_attention(q, k, v, mode=mode, *args, **kwargs)


def main():
    global tracker

    # Parse arguments
    args = parse_args()

    # Monkey-patch the attention function
    hyvideo.modules.models.attention = patched_attention

    # Load model
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`model_base` does not exist: {models_root_path}")

    logger.info("Loading HunyuanVideo model...")
    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

    # Wrap scheduler.step to track denoising steps
    original_scheduler_step = sampler.pipeline.scheduler.step

    def tracked_scheduler_step(model_output, timestep, sample, *args, **kwargs):
        result = original_scheduler_step(model_output, timestep, sample, *args, **kwargs)
        # After each step, increment the step counter
        tracker.current_step += 1
        tracker.block_counter = 0
        return result

    sampler.pipeline.scheduler.step = tracked_scheduler_step

    # Configure generation parameters
    # Use small size for faster verification
    height = getattr(args, 'height', 256) or 256
    width = getattr(args, 'width', 256) or 256
    video_length = getattr(args, 'video_length', 9) or 9
    infer_steps = getattr(args, 'infer_steps', 10) or 10
    prompt = getattr(args, 'prompt', None) or "a serene lake with mountains in the background, calm water, blue sky"

    logger.info(f"Generation config: {height}x{width}, {video_length} frames, {infer_steps} steps")
    logger.info(f"Prompt: {prompt}")

    # Calculate expected spatial shape
    # After patchification: T' = (T-1)//4 + 1, H' = H//16, W' = W//16 (assuming patch_size=2, vae_downsample=8)
    # For HunyuanVideo: temporal compression is 4x, spatial is 8x from VAE, then 2x from patch
    t_patches = (video_length - 1) // 4 + 1
    h_patches = height // 16
    w_patches = width // 16
    spatial_shape = (t_patches, h_patches, w_patches)
    logger.info(f"Expected spatial shape: {spatial_shape} = {t_patches * h_patches * w_patches} tokens")

    # Run generation
    logger.info("Starting generation...")
    tracker.current_step = 0
    tracker.block_counter = 0

    try:
        samples = sampler.predict(
            prompt=prompt,
            height=height,
            width=width,
            video_length=video_length,
            infer_steps=infer_steps,
            seed=42,
            negative_prompt="blurry, low quality",
        )
        logger.info("Generation complete!")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.info("Analyzing partial data...")

    # Analyze results
    similarity_results = tracker.print_summary()

    # Optionally save detailed results
    output_dir = Path("outputs/kv_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-token similarity data for further analysis
    if similarity_results:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for block_idx, data in similarity_results.items():
            json_results[str(block_idx)] = {
                'k_sim_mean': data['k_sim_mean'].tolist(),
                'v_sim_mean': data['v_sim_mean'].tolist(),
                'k_sim_per_token_mean': data['k_sim_per_token'].mean(axis=0).tolist()[:100],  # First 100 tokens
                'v_sim_per_token_mean': data['v_sim_per_token'].mean(axis=0).tolist()[:100],
            }

        output_file = output_dir / "kv_similarity_analysis.json"
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'height': height,
                    'width': width,
                    'video_length': video_length,
                    'infer_steps': infer_steps,
                    'prompt': prompt,
                    'spatial_shape': spatial_shape,
                },
                'results': json_results
            }, f, indent=2)
        logger.info(f"Saved analysis results to {output_file}")

        # Try to do spatial analysis
        try:
            spatial_results = tracker.analyze_spatial_stability(spatial_shape)
            if spatial_results:
                print("\n" + "="*70)
                print("SPATIAL STABILITY ANALYSIS")
                print("="*70)
                for block_idx in sorted(list(spatial_results.keys())[:5]):  # First 5 blocks
                    data = spatial_results[block_idx]
                    print(f"\nBlock {block_idx}:")
                    print(f"  K stable ratio (>0.9 sim): {data['k_stable_ratio']*100:.1f}%")
                    print(f"  V stable ratio (>0.9 sim): {data['v_stable_ratio']*100:.1f}%")
                    print(f"  K mean similarity: {data['k_mean_sim']:.4f}")
                    print(f"  V mean similarity: {data['v_mean_sim']:.4f}")
        except Exception as e:
            logger.warning(f"Spatial analysis failed: {e}")


if __name__ == "__main__":
    main()
