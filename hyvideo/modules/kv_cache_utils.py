"""
KV Cache Reuse Utilities for HunyuanVideo DiT

This module implements region-based KV cache reusing mechanism for video generation.
The key idea is to reuse KV cache for regions where the change between consecutive
denoising steps is below a threshold.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_latent_delta_mask(
    current_latents: torch.Tensor,
    previous_latents: Optional[torch.Tensor],
    threshold: float = 0.1,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    aggregation: str = "l2",
) -> torch.Tensor:
    """
    Compute a binary mask indicating which spatial-temporal patches have changed
    significantly between consecutive denoising steps.

    Args:
        current_latents: Current step latents [B, C, T, H, W]
        previous_latents: Previous step latents [B, C, T, H, W], None for first step
        threshold: Change threshold. Patches with change < threshold will be marked for reuse
        patch_size: Patch size (T, H, W) for aggregation
        aggregation: Method to compute change ('l2', 'cosine', 'mse')

    Returns:
        Binary mask [B, num_patches] where 1 = reuse cache, 0 = recompute
        num_patches = (T // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])
    """
    if previous_latents is None:
        # First step, no cache to reuse
        B, C, T, H, W = current_latents.shape
        pt, ph, pw = patch_size
        num_patches = (T // pt) * (H // ph) * (W // pw)
        return torch.zeros(B, num_patches, dtype=torch.bool, device=current_latents.device)

    assert current_latents.shape == previous_latents.shape, \
        f"Shape mismatch: {current_latents.shape} vs {previous_latents.shape}"

    B, C, T, H, W = current_latents.shape
    pt, ph, pw = patch_size

    # Reshape to patches: [B, C, T//pt, pt, H//ph, ph, W//pw, pw]
    current_patches = current_latents.unfold(2, pt, pt).unfold(3, ph, ph).unfold(4, pw, pw)
    previous_patches = previous_latents.unfold(2, pt, pt).unfold(3, ph, ph).unfold(4, pw, pw)

    # Reshape to [B, num_patches, C * pt * ph * pw]
    num_patches_t = T // pt
    num_patches_h = H // ph
    num_patches_w = W // pw
    num_patches = num_patches_t * num_patches_h * num_patches_w

    current_patches = current_patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
    current_patches = current_patches.view(B, num_patches, -1)

    previous_patches = previous_patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
    previous_patches = previous_patches.view(B, num_patches, -1)

    # Compute change metric
    if aggregation == "l2":
        # L2 distance normalized by patch size
        delta = torch.norm(current_patches - previous_patches, p=2, dim=-1)
        delta = delta / torch.norm(previous_patches, p=2, dim=-1).clamp(min=1e-6)
    elif aggregation == "cosine":
        # Cosine similarity (1 - similarity to get distance)
        current_norm = F.normalize(current_patches, p=2, dim=-1)
        previous_norm = F.normalize(previous_patches, p=2, dim=-1)
        similarity = (current_norm * previous_norm).sum(dim=-1)
        delta = 1.0 - similarity
    elif aggregation == "mse":
        # Mean squared error
        delta = ((current_patches - previous_patches) ** 2).mean(dim=-1)
        delta = torch.sqrt(delta)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Create mask: 1 where change is small (can reuse), 0 where change is large (must recompute)
    reuse_mask = (delta < threshold).bool()

    return reuse_mask


def apply_kv_cache_mask(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cached_k: Optional[torch.Tensor],
    cached_v: Optional[torch.Tensor],
    reuse_mask: torch.Tensor,
    img_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selectively reuse cached K/V based on the reuse mask.

    Args:
        new_k: Newly computed K [B, L, H, D] where L = img_len + txt_len
        new_v: Newly computed V [B, L, H, D]
        cached_k: Cached K from previous step [B, img_len, H, D]
        cached_v: Cached V from previous step [B, img_len, H, D]
        reuse_mask: Binary mask [B, num_patches] indicating which patches to reuse
        img_len: Length of image sequence (before text)

    Returns:
        Updated K and V with selective reuse applied
    """
    if cached_k is None or cached_v is None:
        # No cache available, return new K/V as-is
        return new_k, new_v

    B, L, H, D = new_k.shape
    assert L >= img_len, f"Sequence length {L} must be >= img_len {img_len}"

    # Extract image portion
    new_k_img = new_k[:, :img_len]
    new_v_img = new_v[:, :img_len]

    # Expand reuse_mask to match K/V dimensions
    # reuse_mask: [B, num_patches] -> [B, img_len, H, D]
    num_patches = reuse_mask.shape[1]
    tokens_per_patch = img_len // num_patches

    # Create expanded mask
    mask_expanded = reuse_mask.unsqueeze(-1).repeat(1, 1, tokens_per_patch)  # [B, num_patches, tokens_per_patch]
    mask_expanded = mask_expanded.view(B, -1)[:, :img_len]  # [B, img_len]
    mask_expanded = mask_expanded.unsqueeze(-1).unsqueeze(-1).expand(B, img_len, H, D)  # [B, img_len, H, D]

    # Apply mask: use cached where mask=1, use new where mask=0
    updated_k_img = torch.where(mask_expanded, cached_k[:, :img_len], new_k_img)
    updated_v_img = torch.where(mask_expanded, cached_v[:, :img_len], new_v_img)

    # Concatenate back with text portion (if any)
    if L > img_len:
        updated_k = torch.cat([updated_k_img, new_k[:, img_len:]], dim=1)
        updated_v = torch.cat([updated_v_img, new_v[:, img_len:]], dim=1)
    else:
        updated_k = updated_k_img
        updated_v = updated_v_img

    return updated_k, updated_v


def get_kv_cache_stats(reuse_mask: torch.Tensor) -> dict:
    """
    Compute statistics about KV cache reuse.

    Args:
        reuse_mask: Binary mask [B, num_patches]

    Returns:
        Dictionary with statistics
    """
    total_patches = reuse_mask.numel()
    reused_patches = reuse_mask.sum().item()
    reuse_ratio = reused_patches / total_patches if total_patches > 0 else 0.0

    return {
        "total_patches": total_patches,
        "reused_patches": int(reused_patches),
        "recomputed_patches": total_patches - int(reused_patches),
        "reuse_ratio": reuse_ratio,
    }
