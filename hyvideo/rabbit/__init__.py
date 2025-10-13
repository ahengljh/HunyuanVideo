"""Runtime utilities inspired by the RabbitVideo paper.

This package exposes configuration helpers and runtime managers enabling
selective offloading and dynamic execution for HunyuanVideo's DiT backbone.
"""

from .config import RabbitRuntimeConfig, build_runtime_config
from .runtime import RabbitRuntimeManager

__all__ = [
    "RabbitRuntimeConfig",
    "RabbitRuntimeManager",
    "build_runtime_config",
]
