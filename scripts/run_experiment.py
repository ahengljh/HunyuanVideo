#!/usr/bin/env python3
"""Run sample_video.py, capture wall-clock and GPU VRAM metrics, store JSONL rows."""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

try:
    import pynvml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pynvml is required. Install via `pip install pynvml`." ) from exc


def _poll_vram(stop_event: threading.Event, device_index: int, interval: float, out_dict: dict):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    peak = 0
    try:
        while not stop_event.is_set():
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used // (1024 ** 2)
            peak = max(peak, mem)
            time.sleep(interval)
    finally:
        pynvml.nvmlShutdown()
    out_dict["peak_vram_mb"] = peak


def run_with_metrics(cmd, log_path: Path, device_index: int = 0, interval: float = 0.5):
    stop_event = threading.Event()
    shared = {"peak_vram_mb": 0}
    watcher = threading.Thread(
        target=_poll_vram, args=(stop_event, device_index, interval, shared)
    )
    watcher.start()

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    stop_event.set()
    watcher.join()

    log_path.write_text(result.stdout)
    log_path.with_suffix(".err").write_text(result.stderr)

    reported = None
    for line in result.stdout.splitlines():
        if "Success, time" in line:
            reported = line.split("time:")[-1].strip()
            break

    metrics = {
        "elapsed_sec": elapsed,
        "peak_vram_mb": shared["peak_vram_mb"],
        "reported_time": reported,
        "returncode": result.returncode,
    }
    if result.returncode != 0:
        metrics["error"] = "command exited with non-zero status"
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sample_video.py and record metrics.")
    parser.add_argument("--tag", required=True, help="Identifier for this run")
    parser.add_argument("--run-dir", default="runs", help="Output directory")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for NVML")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between NVML polls")
    parser.add_argument("sample_video_args", nargs=argparse.REMAINDER, help="Arguments for sample_video.py")
    args = parser.parse_args()

    # Remove the '--' separator if it exists at the beginning
    sample_video_args = args.sample_video_args
    if sample_video_args and sample_video_args[0] == '--':
        sample_video_args = sample_video_args[1:]

    if not sample_video_args:
        raise SystemExit("No arguments provided for sample_video.py")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / f"{args.tag}.log"

    metrics = run_with_metrics(
        sample_video_args,
        log_file,
        device_index=args.gpu_index,
        interval=args.poll_interval,
    )
    metrics.update({
        "tag": args.tag,
        "cmd": " ".join(sample_video_args),
    })

    metrics_file = run_dir / "metrics.jsonl"
    with metrics_file.open("a") as fh:
        fh.write(json.dumps(metrics) + "\n")

    print(json.dumps(metrics, indent=2))
    if metrics["returncode"] != 0:
        raise SystemExit(metrics["returncode"])