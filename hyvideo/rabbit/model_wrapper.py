"""
RabbitVideo Model Wrapper
Wraps transformer blocks with memory optimization hooks.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from functools import wraps


class RabbitBlockWrapper(nn.Module):
    """Wrapper for transformer blocks with RabbitVideo optimizations."""

    def __init__(self, block: nn.Module, block_idx: int, rabbit_manager, block_type: str = 'double'):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.rabbit_manager = rabbit_manager
        self.block_type = block_type

    def forward(self, *args, **kwargs):
        """Forward pass with RabbitVideo memory management."""

        # Pre-block hook
        if self.rabbit_manager:
            context = self.rabbit_manager.before_block(self.block_idx, self.block_type)
        else:
            context = {}

        # Get the device of the block
        block_device = next(self.block.parameters()).device

        # Move inputs to the same device as the block
        args_on_device = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                args_on_device.append(arg.to(block_device))
            else:
                args_on_device.append(arg)

        kwargs_on_device = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs_on_device[key] = value.to(block_device)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                # Handle lists/tuples of tensors
                kwargs_on_device[key] = type(value)(v.to(block_device) if isinstance(v, torch.Tensor) else v for v in value)
            else:
                kwargs_on_device[key] = value

        # Execute block with inputs on the correct device
        if block_device.type == 'cuda':
            with torch.amp.autocast('cuda', enabled=False):  # Control precision manually
                output = self.block(*args_on_device, **kwargs_on_device)
        else:
            # CPU execution
            output = self.block(*args_on_device, **kwargs_on_device)

        # Move output back to original device (cuda) if needed
        if isinstance(output, torch.Tensor):
            output = output.to('cuda')
        elif isinstance(output, tuple):
            output = tuple(o.to('cuda') if isinstance(o, torch.Tensor) else o for o in output)
        elif isinstance(output, list):
            output = [o.to('cuda') if isinstance(o, torch.Tensor) else o for o in output]

        # Post-block hook
        if self.rabbit_manager:
            self.rabbit_manager.after_block(self.block_idx, context)

        return output


def wrap_transformer_with_rabbit(model: nn.Module, rabbit_manager) -> nn.Module:
    """Wrap all transformer blocks with RabbitVideo optimizations."""

    if rabbit_manager is None:
        return model

    block_idx = 0

    # Wrap double blocks
    if hasattr(model, 'double_blocks'):
        wrapped_double_blocks = []
        for block in model.double_blocks:
            wrapped_block = RabbitBlockWrapper(
                block, block_idx, rabbit_manager, 'double'
            )
            wrapped_double_blocks.append(wrapped_block)
            block_idx += 1
        model.double_blocks = nn.ModuleList(wrapped_double_blocks)

    # Wrap single blocks
    if hasattr(model, 'single_blocks'):
        wrapped_single_blocks = []
        for block in model.single_blocks:
            wrapped_block = RabbitBlockWrapper(
                block, block_idx, rabbit_manager, 'single'
            )
            wrapped_single_blocks.append(wrapped_block)
            block_idx += 1
        model.single_blocks = nn.ModuleList(wrapped_single_blocks)

    print(f"[RabbitVideo] Wrapped {block_idx} transformer blocks")
    return model


def add_attention_hooks(model: nn.Module, rabbit_manager):
    """Add hooks to attention layers for temporal caching."""

    if rabbit_manager is None or not rabbit_manager.config.enable_caching:
        return

    frame_idx = 0  # Track frame index

    def create_attention_hook(module, is_img_attn=True):
        """Create attention hook for a module."""
        original_forward = module.forward

        @wraps(module.forward)
        def hooked_forward(q, k, v, *args, **kwargs):
            nonlocal frame_idx

            # Check cache before computation
            cached_output, cache_hit = rabbit_manager.before_attention(
                q, k, v, frame_idx
            )

            if cache_hit:
                return cached_output

            # Compute attention
            output = original_forward(q, k, v, *args, **kwargs)

            # Update cache after computation
            rabbit_manager.after_attention(output, q, k, v, frame_idx)

            # Update frame index
            frame_idx += 1

            return output

        return hooked_forward

    # Hook attention layers in the model
    for name, module in model.named_modules():
        # Look for attention modules
        if 'attn' in name.lower() and hasattr(module, 'forward'):
            # Skip if it's a projection or norm layer
            if 'proj' in name or 'norm' in name or 'qkv' in name:
                continue

            # Add hook to attention computation
            if 'img_attn' in name:
                module.forward = create_attention_hook(module, is_img_attn=True)
            elif 'txt_attn' in name:
                module.forward = create_attention_hook(module, is_img_attn=False)

    print(f"[RabbitVideo] Added attention hooks for temporal caching")


def optimize_model_for_rabbit(model: nn.Module, rabbit_manager) -> nn.Module:
    """Apply all RabbitVideo optimizations to the model."""

    if rabbit_manager is None:
        return model

    # Wrap transformer blocks
    model = wrap_transformer_with_rabbit(model, rabbit_manager)

    # Add attention hooks for caching
    add_attention_hooks(model, rabbit_manager)

    # Enable gradient checkpointing if configured
    if rabbit_manager.config.enable_gradient_checkpointing:
        if hasattr(model, 'enable_gradient_checkpointing'):
            try:
                model.enable_gradient_checkpointing()
                print("[RabbitVideo] Gradient checkpointing enabled")
            except ValueError as e:
                # Model doesn't support built-in gradient checkpointing
                # We'll handle this through our block wrappers instead
                print("[RabbitVideo] Model doesn't support built-in gradient checkpointing, using manual checkpointing")
        else:
            print("[RabbitVideo] Gradient checkpointing will be handled per-block")

    return model