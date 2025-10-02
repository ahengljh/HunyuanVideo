#!/usr/bin/env python3
"""
Benchmark script for comparing HunyuanVideo with and without CPU offloading.
"""
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

import torch
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.inference import HunyuanVideoSampler


def parse_benchmark_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo CPU Offloading Benchmark")

    parser.add_argument("--model-base", type=str, default="ckpts", help="Model base path")
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic style.",
                       help="Prompt for generation")
    parser.add_argument("--video-size", type=int, nargs=2, default=[544, 960],
                       help="Video size (height width)")
    parser.add_argument("--video-length", type=int, default=129, help="Video length in frames")
    parser.add_argument("--infer-steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=str, default="./benchmark_results",
                       help="Path to save results")
    parser.add_argument("--offload-blocks", type=str, default="0,1,2,3,4",
                       help="Comma-separated block indices to offload")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline (no offloading) run")
    parser.add_argument("--skip-offload", action="store_true",
                       help="Skip offload run")

    return parser.parse_args()


def run_inference(args_dict, run_name, output_dir):
    """Run single inference with given configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {run_name}")
    logger.info(f"{'='*60}\n")

    # Import here to get fresh config
    from hyvideo.config import parse_args

    # Convert dict to argument list
    arg_list = []
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                arg_list.append(f"--{key}")
        elif isinstance(value, list):
            arg_list.append(f"--{key}")
            arg_list.extend([str(v) for v in value])
        else:
            arg_list.append(f"--{key}")
            arg_list.append(str(value))

    args = parse_args(namespace=arg_list)

    # Initialize sampler
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"Model base path not found: {models_root_path}")

    start_time = time.time()
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    load_time = time.time() - start_time

    logger.info(f"Model loading time: {load_time:.2f}s")

    # Get updated args
    args = hunyuan_video_sampler.args

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Run inference
    inference_start = time.time()
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    inference_time = time.time() - inference_start

    # Collect metrics
    metrics = {
        'run_name': run_name,
        'load_time': load_time,
        'inference_time': inference_time,
        'total_time': load_time + inference_time,
        'prompt': args.prompt,
        'video_size': args.video_size,
        'video_length': args.video_length,
        'infer_steps': args.infer_steps,
        'seed': args.seed,
    }

    if torch.cuda.is_available():
        metrics['gpu_memory_peak_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        metrics['gpu_memory_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)

    # Save video
    samples = outputs['samples']
    if samples:
        video_path = output_dir / f"{run_name}.mp4"
        save_videos_grid(samples[0].unsqueeze(0), str(video_path), fps=24)
        metrics['output_video'] = str(video_path)
        logger.info(f"Video saved to: {video_path}")

    # Clean up
    del hunyuan_video_sampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    args = parse_benchmark_args()

    # Create output directory
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_metrics = []

    # Base configuration
    base_config = {
        'model-base': args.model_base,
        'prompt': args.prompt,
        'video-size': args.video_size,
        'video-length': args.video_length,
        'infer-steps': args.infer_steps,
        'seed': args.seed,
        'flow-reverse': True,
        'collect-metrics': True,
    }

    # Run 1: Baseline (no offloading)
    if not args.skip_baseline:
        baseline_config = base_config.copy()
        baseline_config['use-cpu-offload'] = False
        baseline_config['layer-offload'] = False

        try:
            baseline_metrics = run_inference(
                baseline_config,
                f"baseline_{timestamp}",
                output_dir
            )
            all_metrics.append(baseline_metrics)
        except Exception as e:
            logger.error(f"Baseline run failed: {e}")
            import traceback
            traceback.print_exc()

    # Run 2: With CPU offloading
    if not args.skip_offload:
        offload_config = base_config.copy()
        offload_config['use-cpu-offload'] = False  # Don't use sequential offload
        offload_config['layer-offload'] = True
        offload_config['offload-blocks'] = args.offload_blocks

        try:
            offload_metrics = run_inference(
                offload_config,
                f"offload_{timestamp}",
                output_dir
            )
            all_metrics.append(offload_metrics)
        except Exception as e:
            logger.error(f"Offload run failed: {e}")
            import traceback
            traceback.print_exc()

    # Save metrics to JSON
    metrics_file = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\nMetrics saved to: {metrics_file}")

    # Print comparison table
    if len(all_metrics) >= 2:
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON")
        print("="*80)
        print(f"{'Metric':<30} {'Baseline':<20} {'Offloaded':<20} {'Difference':<20}")
        print("-"*80)

        baseline = all_metrics[0]
        offloaded = all_metrics[1]

        print(f"{'Load Time (s)':<30} {baseline['load_time']:>18.2f}  {offloaded['load_time']:>18.2f}  {offloaded['load_time']-baseline['load_time']:>+18.2f}")
        print(f"{'Inference Time (s)':<30} {baseline['inference_time']:>18.2f}  {offloaded['inference_time']:>18.2f}  {offloaded['inference_time']-baseline['inference_time']:>+18.2f}")
        print(f"{'Total Time (s)':<30} {baseline['total_time']:>18.2f}  {offloaded['total_time']:>18.2f}  {offloaded['total_time']-baseline['total_time']:>+18.2f}")

        if 'gpu_memory_peak_gb' in baseline:
            print(f"{'GPU Memory Peak (GB)':<30} {baseline['gpu_memory_peak_gb']:>18.2f}  {offloaded['gpu_memory_peak_gb']:>18.2f}  {offloaded['gpu_memory_peak_gb']-baseline['gpu_memory_peak_gb']:>+18.2f}")
            savings_pct = (baseline['gpu_memory_peak_gb'] - offloaded['gpu_memory_peak_gb']) / baseline['gpu_memory_peak_gb'] * 100
            print(f"{'GPU Memory Savings (%)':<30} {'':<20} {'':<20} {savings_pct:>+18.2f}")

        slowdown_pct = (offloaded['inference_time'] - baseline['inference_time']) / baseline['inference_time'] * 100
        print(f"{'Inference Slowdown (%)':<30} {'':<20} {'':<20} {slowdown_pct:>+18.2f}")

        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Video Size: {args.video_size[0]}x{args.video_size[1]}")
        print(f"  Video Length: {args.video_length} frames")
        print(f"  Inference Steps: {args.infer_steps}")
        print(f"  Offloaded Blocks: {args.offload_blocks}")
        print("="*80)


if __name__ == "__main__":
    main()
