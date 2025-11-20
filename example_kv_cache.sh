#!/bin/bash
#
# Example script for testing KV Cache Reuse feature
# This script demonstrates how to use the KV cache reuse mechanism
#

# Basic example with default settings
echo "=== Example 1: Basic KV Cache Usage ==="
python sample_video.py \
    --prompt "一只可爱的熊猫在竹林里玩耍" \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --seed 42 \
    --enable-kv-cache \
    --save-path ./results/kv_cache_basic

echo ""
echo "=== Example 2: High Quality Mode (lower threshold) ==="
python sample_video.py \
    --prompt "A serene lake with mountains in the background at sunset" \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --seed 42 \
    --enable-kv-cache \
    --kv-cache-threshold 0.05 \
    --kv-cache-patch-size 1 2 2 \
    --save-path ./results/kv_cache_high_quality

echo ""
echo "=== Example 3: Fast Mode (higher threshold, larger patches) ==="
python sample_video.py \
    --prompt "A dog running through a field of flowers" \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --seed 42 \
    --enable-kv-cache \
    --kv-cache-threshold 0.15 \
    --kv-cache-patch-size 1 4 4 \
    --save-path ./results/kv_cache_fast

echo ""
echo "=== Example 4: Comparison - Without KV Cache ==="
python sample_video.py \
    --prompt "一只可爱的熊猫在竹林里玩耍" \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --seed 42 \
    --save-path ./results/no_kv_cache

echo ""
echo "All examples completed! Check the results in ./results/"
echo "Compare the generation time and quality between different modes."
