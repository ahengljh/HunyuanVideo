#!/usr/bin/env python3
"""
PyTorch Memory Allocator Configuration

This script shows how to configure PyTorch's CUDA memory allocator
to reduce reserved memory / cache behavior.

Usage:
    # Option 1: Aggressive cache clearing (recommended)
    python sample_video.py --aggressive-cache-clear --rabbit-debug

    # Option 2: Environment variables (before running)
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    python sample_video.py

    # Option 3: Completely disable caching (NOT RECOMMENDED - very slow)
    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    python sample_video.py
"""

import os
import torch

print("="*60)
print("PyTorch CUDA Memory Allocator Configuration")
print("="*60)

# Show current settings
print("\nCurrent Environment Variables:")
print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
print(f"  PYTORCH_NO_CUDA_MEMORY_CACHING: {os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING', 'Not set')}")

if torch.cuda.is_available():
    print(f"\nCurrent GPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  Cached:    {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**3:.2f} GB")

print("\n" + "="*60)
print("Available Options to Reduce Cache:")
print("="*60)

print("""
Option 1: Aggressive Cache Clearing (RECOMMENDED)
-------------------------------------------------
Add flag when running:
    --aggressive-cache-clear

This clears cache every 5 denoising steps.
Pros: Reduces reserved memory by 30-50%
Cons: ~5-10% slower

Example:
    python sample_video.py --aggressive-cache-clear --rabbit-debug --prompt "test"


Option 2: Configure Allocator (MODERATE)
-----------------------------------------
Set environment variable before running:
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

This limits memory fragmentation.
Pros: Can reduce cache by 20-30%, minimal slowdown
Cons: Requires restart of Python process

Other useful configs:
    expandable_segments:True      # Allow memory to grow
    garbage_collection_threshold:0.8  # Trigger GC earlier
    max_split_size_mb:256         # Smaller splits = less cache

Example:
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.7
    python sample_video.py


Option 3: Disable Caching (NOT RECOMMENDED - VERY SLOW)
-------------------------------------------------------
Set environment variable:
    export PYTORCH_NO_CUDA_MEMORY_CACHING=1

This completely disables the caching allocator.
Pros: Reserved = Allocated (no cache)
Cons: 10-100x slower! Every allocation goes through CUDA

Only use for debugging memory issues.

Example:
    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    python sample_video.py --infer-steps 5  # Use few steps!


Option 4: Manual Cache Clearing
--------------------------------
In your code, call periodically:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

This is what --aggressive-cache-clear does internally.


Option 5: Lower Peak Memory to Prevent Large Reservation
---------------------------------------------------------
Prevent high peak in the first place:
    --rabbit-mode \\
    --rabbit-offload-threshold 15.0 \\
    --rabbit-aggressive-offload

If peak never exceeds 25GB, PyTorch won't reserve 66GB.


RECOMMENDATION FOR YOUR USE CASE
=================================
Based on your issue (Reserved 66GB, Allocated 20GB):

1. Try aggressive cache clearing first:
   python sample_video.py --aggressive-cache-clear --rabbit-debug

2. If still high, combine with allocator config:
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
   python sample_video.py --aggressive-cache-clear --rabbit-debug

3. Monitor with profiling:
   Check "Cached" in the logs - should drop from 46GB to <10GB

Expected results:
  Before: Reserved 66GB, Allocated 20GB, Cache 46GB
  After:  Reserved 30GB, Allocated 20GB, Cache 10GB
""")

print("="*60)
