"""
RabbitVideo Model Wrapper (Improved)
Enhanced wrapper with gradient checkpointing and better device handling.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Any
from functools import wraps


class RabbitBlockWrapper(nn.Module):
    """Improved wrapper for transformer blocks with RabbitVideo optimizations."""

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
        try:
            block_device = next(self.block.parameters()).device
        except StopIteration:
            block_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move inputs to the same device as the block (non-blocking for performance)
        args_on_device = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device != block_device:
                args_on_device.append(arg.to(block_device, non_blocking=True))
            else:
                args_on_device.append(arg)

        kwargs_on_device = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.device != block_device:
                kwargs_on_device[key] = value.to(block_device, non_blocking=True)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # Handle lists/tuples of tensors
                if isinstance(value[0], torch.Tensor):
                    kwargs_on_device[key] = type(value)(
                        v.to(block_device, non_blocking=True) if isinstance(v, torch.Tensor) and v.device != block_device else v
                        for v in value
                    )
                else:
                    kwargs_on_device[key] = value
            else:
                kwargs_on_device[key] = value

        # Determine if we should use gradient checkpointing
        use_checkpoint = (
            self.rabbit_manager and
            self.rabbit_manager.config.enable_gradient_checkpointing and
            self.rabbit_manager.should_checkpoint_gradient(self.block_idx) and
            self.training  # Only in training mode
        )

        # Execute block
        if use_checkpoint and len(args_on_device) > 0:
            # Use gradient checkpointing to save memory during training
            def custom_forward(*inputs):
                # Reconstruct args from checkpoint inputs
                return self.block(*inputs, **kwargs_on_device)

            output = checkpoint(custom_forward, *args_on_device, use_reentrant=False)
        else:
            # Normal execution (inference or when checkpointing is disabled)
            output = self.block(*args_on_device, **kwargs_on_device)

        # Move output back to CUDA if the block was on CPU
        if block_device.type == 'cpu' and torch.cuda.is_available():
            if isinstance(output, torch.Tensor):
                output = output.to('cuda', non_blocking=True)
            elif isinstance(output, tuple):
                output = tuple(o.to('cuda', non_blocking=True) if isinstance(o, torch.Tensor) else o for o in output)
            elif isinstance(output, list):
                output = [o.to('cuda', non_blocking=True) if isinstance(o, torch.Tensor) else o for o in output]

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
    if hasattr(model, 'double_blocks') and isinstance(model.double_blocks, nn.ModuleList):
        wrapped_double_blocks = []
        for block in model.double_blocks:
            # Don't wrap if already wrapped
            if not isinstance(block, RabbitBlockWrapper):
                wrapped_block = RabbitBlockWrapper(
                    block, block_idx, rabbit_manager, 'double'
                )
                wrapped_double_blocks.append(wrapped_block)
            else:
                wrapped_double_blocks.append(block)
            block_idx += 1
        model.double_blocks = nn.ModuleList(wrapped_double_blocks)
        print(f"[RabbitVideo] Wrapped {len(wrapped_double_blocks)} double blocks")

    # Wrap single blocks
    if hasattr(model, 'single_blocks') and isinstance(model.single_blocks, nn.ModuleList):
        wrapped_single_blocks = []
        for block in model.single_blocks:
            # Don't wrap if already wrapped
            if not isinstance(block, RabbitBlockWrapper):
                wrapped_block = RabbitBlockWrapper(
                    block, block_idx, rabbit_manager, 'single'
                )
                wrapped_single_blocks.append(wrapped_block)
            else:
                wrapped_single_blocks.append(block)
            block_idx += 1
        model.single_blocks = nn.ModuleList(wrapped_single_blocks)
        print(f"[RabbitVideo] Wrapped {len(wrapped_single_blocks)} single blocks")

    print(f"[RabbitVideo] Total wrapped blocks: {block_idx}")
    return model


def optimize_model_for_rabbit(model: nn.Module, rabbit_manager) -> nn.Module:
    """Apply all RabbitVideo optimizations to the model."""

    if rabbit_manager is None:
        return model

    print("[RabbitVideo] Applying model optimizations...")

    # Wrap transformer blocks with offloading and checkpointing support
    model = wrap_transformer_with_rabbit(model, rabbit_manager)

    # Note: We don't use add_attention_hooks as it requires modifying the attention
    # function signature which may not be compatible with the actual implementation
    # Instead, caching will be handled at a higher level if needed

    # Enable gradient checkpointing message
    if rabbit_manager.config.enable_gradient_checkpointing:
        print("[RabbitVideo] Gradient checkpointing enabled (applied per-block)")

    # Set model to eval mode for inference (will be changed if training)
    if not model.training:
        model.eval()

    return model
