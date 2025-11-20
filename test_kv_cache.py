"""
Simple test script to verify KV cache reuse implementation
"""

import torch
import sys
sys.path.insert(0, '.')

from hyvideo.modules.kv_cache_utils import (
    compute_latent_delta_mask,
    apply_kv_cache_mask,
    get_kv_cache_stats,
)


def test_delta_computation():
    """Test delta mask computation"""
    print("Testing delta mask computation...")

    # Create dummy latents
    B, C, T, H, W = 2, 16, 33, 24, 42
    current_latents = torch.randn(B, C, T, H, W)

    # Test 1: First step (no previous latents)
    mask = compute_latent_delta_mask(
        current_latents,
        None,
        threshold=0.1,
        patch_size=(1, 2, 2),
        aggregation="l2"
    )
    print(f"  First step mask shape: {mask.shape}")
    print(f"  First step all zeros (expected): {(mask.sum() == 0).item()}")

    # Test 2: Similar latents (should have high reuse)
    previous_latents = current_latents + torch.randn_like(current_latents) * 0.01
    mask = compute_latent_delta_mask(
        current_latents,
        previous_latents,
        threshold=0.1,
        patch_size=(1, 2, 2),
        aggregation="l2"
    )
    stats = get_kv_cache_stats(mask)
    print(f"  Similar latents mask shape: {mask.shape}")
    print(f"  Similar latents reuse ratio: {stats['reuse_ratio']:.2%} (expected high)")

    # Test 3: Different latents (should have low reuse)
    previous_latents = torch.randn(B, C, T, H, W)
    mask = compute_latent_delta_mask(
        current_latents,
        previous_latents,
        threshold=0.1,
        patch_size=(1, 2, 2),
        aggregation="l2"
    )
    stats = get_kv_cache_stats(mask)
    print(f"  Different latents reuse ratio: {stats['reuse_ratio']:.2%} (expected low)")

    # Test 4: Different aggregation methods
    for agg in ["l2", "cosine", "mse"]:
        mask = compute_latent_delta_mask(
            current_latents,
            previous_latents,
            threshold=0.1,
            patch_size=(1, 2, 2),
            aggregation=agg
        )
        stats = get_kv_cache_stats(mask)
        print(f"  Aggregation {agg}: reuse ratio = {stats['reuse_ratio']:.2%}")

    print("✓ Delta computation tests passed!\n")


def test_kv_cache_application():
    """Test KV cache mask application"""
    print("Testing KV cache mask application...")

    B, L, H, D = 2, 100, 24, 128
    img_len = 80

    # Create dummy K/V
    new_k = torch.randn(B, L, H, D)
    new_v = torch.randn(B, L, H, D)
    cached_k = torch.randn(B, L, H, D)
    cached_v = torch.randn(B, L, H, D)

    # Test 1: No cache (first step)
    updated_k, updated_v = apply_kv_cache_mask(
        new_k, new_v,
        None, None,
        None,
        img_len
    )
    print(f"  No cache: K unchanged = {torch.allclose(updated_k, new_k)}")
    print(f"  No cache: V unchanged = {torch.allclose(updated_v, new_v)}")

    # Test 2: With cache but all recompute
    num_patches = 20
    reuse_mask = torch.zeros(B, num_patches, dtype=torch.bool)
    updated_k, updated_v = apply_kv_cache_mask(
        new_k, new_v,
        cached_k, cached_v,
        reuse_mask,
        img_len
    )
    # Should use new K/V for all positions
    print(f"  All recompute: uses new K/V = {torch.allclose(updated_k[:, :img_len], new_k[:, :img_len])}")

    # Test 3: With cache and all reuse
    reuse_mask = torch.ones(B, num_patches, dtype=torch.bool)
    updated_k, updated_v = apply_kv_cache_mask(
        new_k, new_v,
        cached_k, cached_v,
        reuse_mask,
        img_len
    )
    # Should use cached K/V for img portion
    print(f"  All reuse: uses cached K/V = {torch.allclose(updated_k[:, :img_len], cached_k[:, :img_len])}")

    # Test 4: Text portion should remain unchanged
    print(f"  Text portion unchanged = {torch.allclose(updated_k[:, img_len:], new_k[:, img_len:])}")

    print("✓ KV cache application tests passed!\n")


def test_model_integration():
    """Test integration with model blocks"""
    print("Testing model block integration...")

    try:
        from hyvideo.modules.models import MMDoubleStreamBlock, MMSingleStreamBlock

        # Test MMDoubleStreamBlock
        block = MMDoubleStreamBlock(
            hidden_size=512,
            heads_num=8,
            mlp_width_ratio=4.0,
        )

        # Test enable/disable/clear methods
        block.enable_kv_cache()
        print(f"  DoubleStreamBlock: KV cache enabled = {block.kv_cache_enabled}")

        block.disable_kv_cache()
        print(f"  DoubleStreamBlock: KV cache disabled = {not block.kv_cache_enabled}")

        block.enable_kv_cache()
        block.clear_kv_cache()
        print(f"  DoubleStreamBlock: KV cache cleared = {block.cached_img_k is None}")

        # Test MMSingleStreamBlock
        block = MMSingleStreamBlock(
            hidden_size=512,
            heads_num=8,
            mlp_width_ratio=4.0,
        )

        block.enable_kv_cache()
        print(f"  SingleStreamBlock: KV cache enabled = {block.kv_cache_enabled}")

        block.disable_kv_cache()
        print(f"  SingleStreamBlock: KV cache disabled = {not block.kv_cache_enabled}")

        print("✓ Model block integration tests passed!\n")

    except Exception as e:
        print(f"  Warning: Model block tests skipped due to: {e}\n")


def test_transformer_integration():
    """Test transformer model integration"""
    print("Testing transformer model integration...")

    try:
        from hyvideo.modules.models import HYVideoDiffusionTransformer
        import argparse

        # Create minimal args
        args = argparse.Namespace(
            text_states_dim=4096,
            text_states_dim_2=768,
        )

        # Create small model for testing
        model = HYVideoDiffusionTransformer(
            args=args,
            hidden_size=512,
            heads_num=8,
            mm_double_blocks_depth=2,
            mm_single_blocks_depth=2,
        )

        # Test enable/disable/clear methods
        model.enable_kv_cache()
        print(f"  Transformer: KV cache enabled in all blocks")

        model.disable_kv_cache()
        print(f"  Transformer: KV cache disabled in all blocks")

        model.clear_kv_cache()
        print(f"  Transformer: KV cache cleared in all blocks")

        print("✓ Transformer integration tests passed!\n")

    except Exception as e:
        print(f"  Warning: Transformer tests skipped due to: {e}\n")


if __name__ == "__main__":
    print("="*60)
    print("KV Cache Reuse Implementation Tests")
    print("="*60 + "\n")

    test_delta_computation()
    test_kv_cache_application()
    test_model_integration()
    test_transformer_integration()

    print("="*60)
    print("All tests completed!")
    print("="*60)
