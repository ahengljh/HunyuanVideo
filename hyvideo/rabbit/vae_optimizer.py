"""
RabbitVideo VAE Optimizer
Memory-efficient VAE encoding/decoding with tiling support.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math


class VAEOptimizer:
    """Optimizes VAE memory usage through tiling and offloading."""

    def __init__(
        self,
        vae: nn.Module,
        tile_size: Tuple[int, int] = (256, 256),
        tile_overlap: int = 32,
        enable_tiling: bool = True,
        enable_slicing: bool = True,
        offload_to_cpu: bool = False,
    ):
        self.vae = vae
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.enable_tiling = enable_tiling
        self.enable_slicing = enable_slicing
        self.offload_to_cpu = offload_to_cpu

        # Enable VAE slicing if available (reduces memory for attention)
        if enable_slicing and hasattr(vae, 'enable_slicing'):
            vae.enable_slicing()

        # Enable VAE tiling if available
        if enable_tiling and hasattr(vae, 'enable_tiling'):
            vae.enable_tiling()

    def encode_tiled(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video frames with tiling to reduce memory."""
        if not self.enable_tiling:
            return self.vae.encode(x).latent_dist.sample()

        # x shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        tile_h, tile_w = self.tile_size
        overlap = self.tile_overlap

        # Calculate number of tiles
        num_tiles_h = math.ceil((H - overlap) / (tile_h - overlap))
        num_tiles_w = math.ceil((W - overlap) / (tile_w - overlap))

        # Process spatially per frame to save memory
        latents = []

        for t in range(T):
            frame = x[:, :, t:t+1, :, :]  # (B, C, 1, H, W)
            frame_latents = []

            for i in range(num_tiles_h):
                for j in range(num_tiles_w):
                    # Calculate tile coordinates
                    y1 = i * (tile_h - overlap)
                    x1 = j * (tile_w - overlap)
                    y2 = min(y1 + tile_h, H)
                    x2 = min(x1 + tile_w, W)

                    # Extract tile
                    tile = frame[:, :, :, y1:y2, x1:x2]

                    # Encode tile
                    with torch.cuda.amp.autocast(enabled=True):
                        if self.offload_to_cpu:
                            # Move VAE to GPU temporarily
                            self.vae.to('cuda')

                        tile_latent = self.vae.encode(tile).latent_dist.sample()

                        if self.offload_to_cpu:
                            # Move VAE back to CPU
                            self.vae.to('cpu')
                            torch.cuda.empty_cache()

                    frame_latents.append(tile_latent.cpu())

            # Blend overlapping regions
            frame_latent = self._blend_tiles(
                frame_latents, num_tiles_h, num_tiles_w,
                (B, self.vae.config.latent_channels, 1, H // 8, W // 8)
            )
            latents.append(frame_latent)

        # Stack temporal dimension
        result = torch.cat(latents, dim=2).to(x.device)
        return result

    def decode_tiled(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents with tiling to reduce memory."""
        if not self.enable_tiling:
            return self.vae.decode(latents).sample

        # latents shape: (B, C, T, H, W)
        B, C, T, H, W = latents.shape
        tile_h, tile_w = self.tile_size[0] // 8, self.tile_size[1] // 8  # Latent space is 8x downsampled
        overlap = self.tile_overlap // 8

        # Calculate number of tiles
        num_tiles_h = math.ceil((H - overlap) / (tile_h - overlap))
        num_tiles_w = math.ceil((W - overlap) / (tile_w - overlap))

        # Process spatially per frame
        frames = []

        for t in range(T):
            latent_frame = latents[:, :, t:t+1, :, :]  # (B, C, 1, H, W)
            frame_tiles = []

            for i in range(num_tiles_h):
                for j in range(num_tiles_w):
                    # Calculate tile coordinates
                    y1 = i * (tile_h - overlap)
                    x1 = j * (tile_w - overlap)
                    y2 = min(y1 + tile_h, H)
                    x2 = min(x1 + tile_w, W)

                    # Extract tile
                    tile = latent_frame[:, :, :, y1:y2, x1:x2]

                    # Decode tile
                    with torch.cuda.amp.autocast(enabled=True):
                        if self.offload_to_cpu:
                            self.vae.to('cuda')

                        decoded_tile = self.vae.decode(tile).sample

                        if self.offload_to_cpu:
                            self.vae.to('cpu')
                            torch.cuda.empty_cache()

                    frame_tiles.append(decoded_tile.cpu())

            # Blend overlapping regions
            out_h, out_w = H * 8, W * 8
            frame = self._blend_tiles(
                frame_tiles, num_tiles_h, num_tiles_w,
                (B, 3, 1, out_h, out_w)
            )
            frames.append(frame)

        # Stack temporal dimension
        result = torch.cat(frames, dim=2).to(latents.device)
        return result

    def _blend_tiles(
        self,
        tiles: List[torch.Tensor],
        num_h: int,
        num_w: int,
        output_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Blend overlapping tiles with feathering."""
        output = torch.zeros(output_shape)
        weight = torch.zeros(output_shape)

        B, C, T, H, W = output_shape
        tile_h = H // num_h
        tile_w = W // num_w
        overlap = self.tile_overlap if tiles[0].shape[-1] > 100 else self.tile_overlap // 8

        idx = 0
        for i in range(num_h):
            for j in range(num_w):
                y1 = i * (tile_h - overlap)
                x1 = j * (tile_w - overlap)

                tile = tiles[idx].to(output.device)
                th, tw = tile.shape[-2], tile.shape[-1]
                y2 = y1 + th
                x2 = x1 + tw

                # Create feathering mask
                mask = torch.ones_like(tile)
                if overlap > 0:
                    # Feather edges
                    fade = torch.linspace(0, 1, overlap)
                    if i > 0:  # Top edge
                        mask[:, :, :, :overlap, :] *= fade.view(1, 1, 1, -1, 1)
                    if j > 0:  # Left edge
                        mask[:, :, :, :, :overlap] *= fade.view(1, 1, 1, 1, -1)
                    if i < num_h - 1:  # Bottom edge
                        mask[:, :, :, -overlap:, :] *= fade.flip(0).view(1, 1, 1, -1, 1)
                    if j < num_w - 1:  # Right edge
                        mask[:, :, :, :, -overlap:] *= fade.flip(0).view(1, 1, 1, 1, -1)

                # Accumulate weighted tiles
                output[:, :, :, y1:y2, x1:x2] += tile * mask
                weight[:, :, :, y1:y2, x1:x2] += mask

                idx += 1

        # Normalize by weights
        output = output / (weight + 1e-8)
        return output

    def optimize_vae_memory(self):
        """Apply memory optimizations to VAE."""
        # Enable gradient checkpointing if available
        if hasattr(self.vae, 'enable_gradient_checkpointing'):
            self.vae.enable_gradient_checkpointing()

        # Offload to CPU if requested
        if self.offload_to_cpu:
            self.vae.to('cpu')
            print("[RabbitVideo] VAE offloaded to CPU")

        # Use memory-efficient attention
        if hasattr(self.vae, 'set_use_memory_efficient_attention_xformers'):
            try:
                self.vae.set_use_memory_efficient_attention_xformers(True)
            except:
                pass

        print(f"[RabbitVideo] VAE optimization enabled:")
        print(f"  Tiling: {self.enable_tiling} (tile_size={self.tile_size})")
        print(f"  Slicing: {self.enable_slicing}")
        print(f"  CPU offload: {self.offload_to_cpu}")


def optimize_vae_for_rabbit(
    vae: nn.Module,
    enable_tiling: bool = True,
    tile_size: Tuple[int, int] = (256, 256),
    offload_to_cpu: bool = False,
) -> VAEOptimizer:
    """Create VAE optimizer for RabbitVideo."""
    optimizer = VAEOptimizer(
        vae=vae,
        tile_size=tile_size,
        enable_tiling=enable_tiling,
        offload_to_cpu=offload_to_cpu,
    )
    optimizer.optimize_vae_memory()
    return optimizer
