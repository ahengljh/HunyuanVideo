"""
Lazy Weight Loading Implementation for HunyuanVideo
Reduces memory consumption by avoiding loading all weights at once
"""

import torch
import os
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging
import gc

try:
    import safetensors
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    logging.warning("safetensors not installed. Using torch.load fallback.")

logger = logging.getLogger(__name__)


class LazyWeightLoader:
    """
    Loads model weights lazily to minimize memory consumption.

    Key features:
    1. Memory-mapped file access (no full load to RAM)
    2. Progressive layer-by-layer loading
    3. Direct placement to target device
    4. Automatic memory cleanup
    """

    def __init__(self, checkpoint_path: str, device_map: Optional[Dict] = None):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device_map: Dict mapping layer names to devices ('cpu', 'cuda:0', etc.)
        """
        self.checkpoint_path = checkpoint_path
        self.device_map = device_map or {}
        self.is_safetensors = checkpoint_path.endswith('.safetensors')

        # For non-safetensors checkpoints, we need a different strategy
        self._metadata = None
        self._file_handle = None

    def __enter__(self):
        """Context manager entry"""
        if HAS_SAFETENSORS and self.is_safetensors:
            self._file_handle = safe_open(self.checkpoint_path, framework="pt", device="cpu")
        else:
            # For .pt files, load metadata only
            self._metadata = self._load_checkpoint_metadata()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        if self._file_handle:
            self._file_handle.__exit__(exc_type, exc_val, exc_tb)
        self._file_handle = None
        self._metadata = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_checkpoint_metadata(self) -> Dict:
        """Load only the keys from a checkpoint without loading full tensors"""
        # For .pt files, we need to actually load to get the keys
        # But we'll load to CPU and delete immediately
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location='cpu',
            weights_only=False
        )

        # Extract just the keys
        metadata = {}
        if isinstance(checkpoint, dict):
            # Store the checkpoint structure temporarily
            self._checkpoint_structure = checkpoint

            # Find the actual state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'module' in checkpoint:
                state_dict = checkpoint['module']
            elif 'ema' in checkpoint:
                state_dict = checkpoint['ema']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            self._checkpoint_structure = None

        # Get the keys
        if isinstance(state_dict, dict):
            for key in state_dict.keys():
                metadata[key] = True  # Just mark that the key exists

        # Don't delete checkpoint yet, we'll need it for loading
        return metadata

    def get_weight_keys(self) -> List[str]:
        """Get list of all weight keys in checkpoint"""
        if HAS_SAFETENSORS and self._file_handle:
            return list(self._file_handle.keys())
        elif self._metadata:
            return list(self._metadata.keys())
        else:
            raise RuntimeError("Loader not initialized. Use within context manager.")

    def load_weight(self, key: str, device: Optional[str] = None) -> torch.Tensor:
        """
        Load a single weight tensor.

        Args:
            key: Weight key/name
            device: Target device. If None, uses device_map or 'cpu'

        Returns:
            Loaded tensor on target device
        """
        if device is None:
            device = self.device_map.get(key, 'cpu')

        if HAS_SAFETENSORS and self._file_handle:
            # Safetensors: directly load to target device
            tensor = self._file_handle.get_tensor(key)
            if device != 'cpu':
                tensor = tensor.to(device)
            return tensor
        else:
            # Fallback: load specific tensor from .pt file
            # This still loads the whole file but immediately extracts one tensor
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location='cpu',
                weights_only=False
            )

            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            tensor = state_dict[key]
            if device != 'cpu':
                tensor = tensor.to(device)

            # Cleanup
            del checkpoint
            gc.collect()

            return tensor

    def load_weights_for_layer(self, layer_prefix: str, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Load all weights for a specific layer.

        Args:
            layer_prefix: Prefix for layer weights (e.g., 'model.layers.0')
            device: Target device for all layer weights

        Returns:
            Dict of weight_name -> tensor
        """
        layer_weights = {}

        for key in self.get_weight_keys():
            if key.startswith(layer_prefix):
                layer_weights[key] = self.load_weight(key, device)

        return layer_weights

    def load_to_model_progressively(
        self,
        model: torch.nn.Module,
        layers_per_batch: int = 1,
        callback=None
    ):
        """
        Load weights to model progressively to minimize peak memory.

        Args:
            model: Target model
            layers_per_batch: Number of layers to load at once
            callback: Optional callback(layer_name, progress) for progress tracking
        """
        all_keys = self.get_weight_keys()
        total_keys = len(all_keys)
        loaded_keys = 0

        # Group keys by layer
        layer_groups = self._group_keys_by_layer(all_keys)

        # Process in batches
        batch = {}
        batch_count = 0

        for layer_name, layer_keys in layer_groups.items():
            # Determine device for this layer
            device = self.device_map.get(layer_name, 'cpu')

            # Load all weights for this layer
            for key in layer_keys:
                tensor = self.load_weight(key, device)
                batch[key] = tensor
                loaded_keys += 1

            batch_count += 1

            # Apply batch to model when reaching batch size
            if batch_count >= layers_per_batch:
                logger.info(f"Loading batch of {len(batch)} weights to model...")
                missing, unexpected = model.load_state_dict(batch, strict=False)

                if callback:
                    callback(layer_name, loaded_keys / total_keys)

                # Cleanup
                del batch
                batch = {}
                batch_count = 0
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Load remaining weights
        if batch:
            logger.info(f"Loading final batch of {len(batch)} weights to model...")
            missing, unexpected = model.load_state_dict(batch, strict=False)

            if callback:
                callback("final", 1.0)

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _group_keys_by_layer(self, keys: List[str]) -> Dict[str, List[str]]:
        """Group weight keys by their layer prefix"""
        groups = {}

        for key in keys:
            # Extract layer name (e.g., 'model.layers.0' from 'model.layers.0.weight')
            parts = key.split('.')
            if 'layers' in parts:
                idx = parts.index('layers')
                if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    layer_name = '.'.join(parts[:idx + 2])
                else:
                    layer_name = '.'.join(parts[:idx + 1])
            elif 'blocks' in parts:
                idx = parts.index('blocks')
                if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    layer_name = '.'.join(parts[:idx + 2])
                else:
                    layer_name = '.'.join(parts[:idx + 1])
            else:
                # Default: use first 2 components
                layer_name = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]

            if layer_name not in groups:
                groups[layer_name] = []
            groups[layer_name].append(key)

        return groups


def _load_from_safetensors_lazy(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    layers_per_batch: int = 2,
    progress_callback=None
) -> torch.nn.Module:
    """
    Load from safetensors with TRUE lazy loading - one tensor at a time.
    This never loads the full checkpoint into memory.
    """
    from safetensors import safe_open

    logger.info("Using TRUE lazy loading with safetensors (minimal memory usage)")

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        # Get all keys
        all_keys = list(f.keys())
        logger.info(f"Found {len(all_keys)} tensors in safetensors file")

        # Group keys by layer
        layer_groups = {}
        for key in all_keys:
            parts = key.split('.')
            if 'double_blocks' in parts:
                idx = parts.index('double_blocks')
                if idx + 1 < len(parts):
                    layer_id = '.'.join(parts[:idx + 2])
                else:
                    layer_id = 'double_blocks'
            elif 'single_blocks' in parts:
                idx = parts.index('single_blocks')
                if idx + 1 < len(parts):
                    layer_id = '.'.join(parts[:idx + 2])
                else:
                    layer_id = 'single_blocks'
            elif 'blocks' in parts:
                idx = parts.index('blocks')
                if idx + 1 < len(parts):
                    layer_id = '.'.join(parts[:idx + 2])
                else:
                    layer_id = 'blocks'
            else:
                layer_id = 'base'

            if layer_id not in layer_groups:
                layer_groups[layer_id] = []
            layer_groups[layer_id].append(key)

        logger.info(f"Organized into {len(layer_groups)} layer groups")

        # Load layer by layer
        total_groups = len(layer_groups)
        loaded_groups = 0
        current_batch = {}
        batch_count = 0

        for layer_id, keys in layer_groups.items():
            # Load all tensors for this layer (TRUE lazy loading - one at a time!)
            for key in keys:
                tensor = f.get_tensor(key)
                if device:
                    tensor = tensor.to(device)
                current_batch[key] = tensor

            batch_count += 1

            # Apply batch when we reach batch size
            if batch_count >= layers_per_batch:
                logger.info(f"Applying batch of {len(current_batch)} weights ({batch_count} layers) to model...")
                model.load_state_dict(current_batch, strict=False)

                loaded_groups += batch_count
                if progress_callback:
                    progress_callback(layer_id, loaded_groups / total_groups)

                # Clear batch and cleanup
                current_batch = {}
                batch_count = 0
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Load remaining weights
        if current_batch:
            logger.info(f"Applying final batch of {len(current_batch)} weights to model...")
            model.load_state_dict(current_batch, strict=False)
            loaded_groups += batch_count
            if progress_callback:
                progress_callback("final", 1.0)

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("TRUE lazy loading complete! (checkpoint was never fully loaded to memory)")
    return model


def _load_via_cpu_staged(
    checkpoint_path: str,
    model: torch.nn.Module,
    load_key: Optional[str] = None,
    device: Optional[torch.device] = None,
    layers_per_batch: int = 1,
    progress_callback=None
) -> torch.nn.Module:
    """
    Load checkpoint to CPU, then transfer to GPU layer by layer.

    Strategy: Load ONCE to CPU (uses CPU RAM), then stream layers to GPU.
    Good when you have abundant CPU RAM but limited GPU memory.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Target model to load weights into
        load_key: Key to extract from checkpoint (e.g., 'module', 'ema')
        device: Target device for model weights
        layers_per_batch: Number of layers to transfer to GPU at once
        progress_callback: Optional progress callback

    Returns:
        Model with loaded weights
    """
    logger.info("Step 1/2: Loading entire checkpoint to CPU RAM...")

    # Load to CPU
    if checkpoint_path.endswith('.safetensors') and HAS_SAFETENSORS:
        from safetensors import safe_open
        # For safetensors, load all tensors to CPU
        state_dict = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        # For .pt files
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and load_key and load_key in checkpoint:
            state_dict = checkpoint[load_key]
            del checkpoint
        elif isinstance(checkpoint, dict):
            if 'module' in checkpoint and isinstance(checkpoint['module'], dict):
                state_dict = checkpoint['module']
            elif 'ema' in checkpoint and isinstance(checkpoint['ema'], dict):
                state_dict = checkpoint['ema']
            else:
                state_dict = checkpoint
            del checkpoint
        else:
            state_dict = checkpoint

    gc.collect()

    logger.info(f"Step 2/2: Transferring {len(state_dict)} weights to GPU layer by layer...")

    # Group by layers
    layer_groups = {}
    for key in state_dict.keys():
        parts = key.split('.')
        if 'double_blocks' in parts:
            idx = parts.index('double_blocks')
            layer_id = '.'.join(parts[:idx + 2]) if idx + 1 < len(parts) else 'double_blocks'
        elif 'single_blocks' in parts:
            idx = parts.index('single_blocks')
            layer_id = '.'.join(parts[:idx + 2]) if idx + 1 < len(parts) else 'single_blocks'
        elif 'blocks' in parts:
            idx = parts.index('blocks')
            layer_id = '.'.join(parts[:idx + 2]) if idx + 1 < len(parts) else 'blocks'
        else:
            layer_id = 'base'

        if layer_id not in layer_groups:
            layer_groups[layer_id] = []
        layer_groups[layer_id].append(key)

    logger.info(f"Organized into {len(layer_groups)} layer groups")

    # Transfer layer by layer to GPU
    total_groups = len(layer_groups)
    loaded_groups = 0
    current_batch = {}
    batch_count = 0

    for layer_id, keys in layer_groups.items():
        # Transfer this layer's weights from CPU to GPU
        logger.info(f"Transferring layer {layer_id} ({len(keys)} tensors) from CPU to GPU...")

        for key in keys:
            tensor = state_dict[key]
            if device and str(device) != 'cpu':
                tensor = tensor.to(device)
            current_batch[key] = tensor

            # Immediately remove from CPU dict to free CPU memory
            del state_dict[key]

        batch_count += 1

        # Apply batch when we reach batch size
        if batch_count >= layers_per_batch:
            logger.info(f"Applying batch of {len(current_batch)} weights to model...")
            model.load_state_dict(current_batch, strict=False)

            loaded_groups += batch_count
            if progress_callback:
                progress_callback(layer_id, loaded_groups / total_groups)

            # Clear GPU batch
            current_batch = {}
            batch_count = 0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load remaining weights
    if current_batch:
        logger.info(f"Applying final batch of {len(current_batch)} weights to model...")
        model.load_state_dict(current_batch, strict=False)
        loaded_groups += batch_count
        if progress_callback:
            progress_callback("final", 1.0)

    # Cleanup
    del state_dict
    del current_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("CPU-staged loading complete!")
    return model


def load_state_dict_lazy(
    checkpoint_path: str,
    model: torch.nn.Module,
    load_key: Optional[str] = None,
    device: Optional[torch.device] = None,
    layers_per_batch: int = 2,
    progress_callback=None,
    via_cpu: bool = False
) -> torch.nn.Module:
    """
    Load state dict with lazy loading to minimize peak memory usage.

    For safetensors files, we use true lazy loading (one tensor at a time from disk).
    For .pt files, we load the checkpoint once but apply weights in batches,
    which reduces peak memory during the load_state_dict operation.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Target model to load weights into
        load_key: Key to extract state dict from checkpoint (e.g., 'module', 'ema')
        device: Target device for model weights
        layers_per_batch: Number of layers to load at once
        progress_callback: Optional callback(layer_name, progress) for progress tracking
        via_cpu: If True, load to CPU first then transfer layer-by-layer to GPU

    Returns:
        Model with loaded weights
    """
    is_safetensors = checkpoint_path.endswith('.safetensors')

    # For safetensors with via_cpu, use the CPU-based approach
    if via_cpu:
        logger.info(f"Loading model via CPU (load all to CPU, transfer layers to GPU one by one)")
        return _load_via_cpu_staged(
            checkpoint_path, model, load_key, device, layers_per_batch, progress_callback
        )

    if is_safetensors and HAS_SAFETENSORS:
        logger.info(f"Loading model with TRUE lazy loading from safetensors: {checkpoint_path}")
        return _load_from_safetensors_lazy(
            checkpoint_path, model, device, layers_per_batch, progress_callback
        )
    else:
        logger.info(f"Loading model with batched loading from {checkpoint_path}")
    logger.info(f"Loading checkpoint to CPU first...")

    # Load checkpoint to CPU
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict based on load_key
    if isinstance(checkpoint, dict) and load_key and load_key in checkpoint:
        logger.info(f"Extracting key '{load_key}' from checkpoint...")
        state_dict = checkpoint[load_key]
        del checkpoint
    elif isinstance(checkpoint, dict) and not load_key:
        # Try to find state dict automatically
        if 'module' in checkpoint and isinstance(checkpoint['module'], dict):
            state_dict = checkpoint['module']
            del checkpoint
        elif 'ema' in checkpoint and isinstance(checkpoint['ema'], dict):
            state_dict = checkpoint['ema']
            del checkpoint
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    gc.collect()

    # Group keys by layer to apply in batches
    logger.info(f"Organizing {len(state_dict)} weights into batches...")

    # Group keys by layer prefix
    layer_groups = {}
    for key in state_dict.keys():
        # Extract layer identifier
        parts = key.split('.')
        if 'double_blocks' in parts:
            idx = parts.index('double_blocks')
            if idx + 1 < len(parts):
                layer_id = '.'.join(parts[:idx + 2])
            else:
                layer_id = 'double_blocks'
        elif 'single_blocks' in parts:
            idx = parts.index('single_blocks')
            if idx + 1 < len(parts):
                layer_id = '.'.join(parts[:idx + 2])
            else:
                layer_id = 'single_blocks'
        elif 'blocks' in parts:
            idx = parts.index('blocks')
            if idx + 1 < len(parts):
                layer_id = '.'.join(parts[:idx + 2])
            else:
                layer_id = 'blocks'
        else:
            # Non-block weights (embeddings, norms, etc.)
            layer_id = 'base'

        if layer_id not in layer_groups:
            layer_groups[layer_id] = []
        layer_groups[layer_id].append(key)

    logger.info(f"Found {len(layer_groups)} layer groups")

    # Load in batches
    total_groups = len(layer_groups)
    loaded_groups = 0
    current_batch = {}
    batch_count = 0

    for layer_id, keys in layer_groups.items():
        # Add this layer's weights to current batch
        for key in keys:
            tensor = state_dict[key]
            if device:
                tensor = tensor.to(device)
            current_batch[key] = tensor

        batch_count += 1

        # Apply batch when we reach the batch size
        if batch_count >= layers_per_batch:
            logger.info(f"Applying batch of {len(current_batch)} weights to model...")
            model.load_state_dict(current_batch, strict=False)

            loaded_groups += batch_count
            if progress_callback:
                progress_callback(layer_id, loaded_groups / total_groups)

            # Clear batch
            current_batch = {}
            batch_count = 0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load remaining weights
    if current_batch:
        logger.info(f"Applying final batch of {len(current_batch)} weights to model...")
        model.load_state_dict(current_batch, strict=False)
        loaded_groups += batch_count
        if progress_callback:
            progress_callback("final", 1.0)

    # Cleanup
    del state_dict
    del current_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Model loading complete!")
    return model
