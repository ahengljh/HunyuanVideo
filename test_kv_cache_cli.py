#!/usr/bin/env python3
"""
Quick test to verify --enable-kv-cache CLI argument works correctly
"""
import sys
sys.path.insert(0, '/home/user/HunyuanVideo')

from hyvideo.config import parse_args

# Test 1: Default (should be False)
print("Test 1: Default behavior")
args = parse_args(namespace=[])
print(f"  enable_kv_cache = {args.enable_kv_cache}")
assert args.enable_kv_cache == False, "Default should be False"
print("  ✓ Pass")

# Test 2: With --enable-kv-cache flag
print("\nTest 2: With --enable-kv-cache flag")
args = parse_args(namespace=["--enable-kv-cache"])
print(f"  enable_kv_cache = {args.enable_kv_cache}")
assert args.enable_kv_cache == True, "Should be True when flag provided"
print("  ✓ Pass")

# Test 3: Full command simulation
print("\nTest 3: Full command with various args")
args = parse_args(namespace=[
    "--model-base", "ckpts",
    "--prompt", "test prompt",
    "--infer-steps", "20",
    "--enable-kv-cache"
])
print(f"  enable_kv_cache = {args.enable_kv_cache}")
print(f"  infer_steps = {args.infer_steps}")
assert args.enable_kv_cache == True, "Should be True"
assert args.infer_steps == 20, "Should preserve other args"
print("  ✓ Pass")

print("\n✓ All tests passed!")
print("\nThe --enable-kv-cache CLI argument is working correctly.")
