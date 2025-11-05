"""
RabbitVideo: Memory-Efficient Video Generation
Optimizations for running large DiT models on consumer GPUs.
"""

from .memory_manager import RabbitMemoryManager
from .offload_manager import OffloadManager
from .block_manager import BlockManager
from .cache_manager import TemporalCacheManager

__all__ = [
    'RabbitMemoryManager',
    'OffloadManager',
    'BlockManager',
    'TemporalCacheManager'
]

__version__ = '0.1.0'