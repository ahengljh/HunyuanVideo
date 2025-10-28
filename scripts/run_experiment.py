#!/usr/bin/env python3
"""Run sample_video.py, capture wall-clock, VRAM traces, and per-step stats."""
import argparse
import json
import math
import re
import shlex
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import pynvml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pynvml is required. Install via `pip install pynvml`." ) from exc


def _poll_vram(stop_event: threading.Event, device_index: int, interval: float, out_dict: dict):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    peak = 0
    samples = []
    try:
        while not stop_event.is_set():
            timestamp = time.time()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used // (1024 ** 2)
            peak = max(peak, mem)
            samples.append((timestamp, mem))
            time.sleep(interval)
    finally:
        pynvml.nvmlShutdown()
    out_dict["peak_vram_mb"] = peak
    out_dict["memory_samples"] = samples


STEP_REGEX = re.compile(
    r"\[Rabbit\]\s*Step\s+(?P<current>\d+)/(?:\s*)(?P<total>\d+)\s*-\s*Progress:\s*(?P<progress>[0-9.]+)%\s*-\s*Timestep:\s*(?P<timestep>[-0-9.]+)"
)


def _stream_subprocess(cmd, log_path: Path, err_path: Path, step_events: List[dict]):
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    def _consume(stream, buffer, callback=None):
        for line in iter(stream.readline, ""):
            if not line:
                break
            timestamp = time.time()
            buffer.append(line)
            if callback is not None:
                callback(line, timestamp)
        stream.close()

    def _step_callback(line: str, timestamp: float):
        match = STEP_REGEX.search(line)
        if not match:
            return
        try:
            current = int(match.group("current"))
            total = int(match.group("total"))
            progress = float(match.group("progress"))
            timestep = float(match.group("timestep"))
        except (ValueError, TypeError):
            return
        step_events.append(
            {
                "step": current,
                "total": total,
                "progress_pct": progress,
                "timestep": timestep,
                "timestamp": timestamp,
            }
        )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_thread = threading.Thread(
        target=_consume, args=(process.stdout, stdout_lines, _step_callback)
    )
    stderr_thread = threading.Thread(
        target=_consume, args=(process.stderr, stderr_lines, None)
    )
    stdout_thread.start()
    stderr_thread.start()

    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    log_path.write_text("".join(stdout_lines))
    err_path.write_text("".join(stderr_lines))

    return returncode, stdout_lines


def _write_memory_trace(samples: Sequence[Tuple[float, int]], path: Path) -> Optional[Path]:
    if not samples:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("timestamp_iso,timestamp_unix_s,memory_used_mb\n")
        for ts, mem in samples:
            ts_iso = datetime.fromtimestamp(ts).isoformat()
            fh.write(f"{ts_iso},{ts:.6f},{mem}\n")
    return path


def _write_step_trace(events: Sequence[dict], path: Path):
    if not events:
        return None, {"step_samples": 0}
    events_sorted = sorted(events, key=lambda item: item["timestamp"])
    durations: List[float] = []
    prev_ts: Optional[float] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("step,total,progress_pct,timestep,time_iso,time_unix_s,delta_sec\n")
        for event in events_sorted:
            ts = event["timestamp"]
            delta = 0.0 if prev_ts is None else max(ts - prev_ts, 0.0)
            if prev_ts is not None:
                durations.append(delta)
            prev_ts = ts
            ts_iso = datetime.fromtimestamp(ts).isoformat()
            fh.write(
                f"{event['step']},{event['total']},{event['progress_pct']:.4f},{event['timestep']:.6f},"
                f"{ts_iso},{ts:.6f},{delta:.6f}\n"
            )

    summary = {}
    if durations:
        durations_sorted = sorted(durations)
        n = len(durations_sorted)
        mean_val = statistics.mean(durations_sorted)
        max_val = durations_sorted[-1]
        p95_index = max(0, min(n - 1, math.ceil(0.95 * n) - 1))
        p95_val = durations_sorted[p95_index]
        summary = {
            "step_mean_sec": mean_val,
            "step_max_sec": max_val,
            "step_p95_sec": p95_val,
            "step_samples": n,
        }
    else:
        summary = {"step_samples": 0}
    return path, summary


def run_with_metrics(cmd, log_path: Path, device_index: int = 0, interval: float = 0.5):
    stop_event = threading.Event()
    shared = {"peak_vram_mb": 0, "memory_samples": []}
    step_events: List[dict] = []
    watcher = threading.Thread(
        target=_poll_vram, args=(stop_event, device_index, interval, shared)
    )
    watcher.start()

    t0 = time.perf_counter()
    err_path = log_path.with_suffix(".err")
    returncode, stdout_lines = _stream_subprocess(cmd, log_path, err_path, step_events)
    elapsed = time.perf_counter() - t0

    stop_event.set()
    watcher.join()

    reported = None
    for raw_line in stdout_lines:
        for line in raw_line.splitlines():
            if "Success, time" in line:
                reported = line.split("time:")[-1].strip()
                break

    memory_trace_path = log_path.with_suffix(".memory.csv")
    step_trace_path = log_path.with_suffix(".steps.csv")

    metrics = {
        "elapsed_sec": elapsed,
        "peak_vram_mb": shared["peak_vram_mb"],
        "reported_time": reported,
        "returncode": returncode,
    }
    if returncode != 0:
        metrics["error"] = "command exited with non-zero status"

    memory_trace = _write_memory_trace(shared.get("memory_samples", []), memory_trace_path)
    if memory_trace is not None:
        metrics["memory_trace"] = str(memory_trace)

    step_trace, step_summary = _write_step_trace(step_events, step_trace_path)
    if step_trace is not None:
        metrics["step_trace"] = str(step_trace)
    metrics.update(step_summary)

    return metrics


def _normalise_command(cmd_spec) -> List[str]:
    if isinstance(cmd_spec, str):
        return shlex.split(cmd_spec)
    if isinstance(cmd_spec, Iterable):
        return [str(part) for part in cmd_spec]
    raise ValueError(f"Unsupported command format: {cmd_spec!r}")


def _prepare_command(entry: dict) -> List[str]:
    if "cmd" in entry:
        return _normalise_command(entry["cmd"])
    if "command" in entry:
        return _normalise_command(entry["command"])
    if "args" in entry:
        return _normalise_command(entry["args"])
    if "sample_args" in entry:
        base = entry.get("python_executable", "python")
        sample_args = _normalise_command(entry["sample_args"])
        script = entry.get("script", "sample_video.py")
        return [base, script] + sample_args
    raise ValueError("Batch entry must provide 'cmd', 'command', 'args', or 'sample_args'.")


def _load_batch(batch_path: Path) -> List[dict]:
    text = batch_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise SystemExit(f"Failed to parse batch file '{batch_path}': {err}") from err

    if isinstance(data, dict) and "runs" in data:
        runs = data["runs"]
    else:
        runs = data
    if not isinstance(runs, list):
        raise SystemExit("Batch file must contain a list of run definitions.")
    processed = []
    for entry in runs:
        if not isinstance(entry, dict):
            raise SystemExit(f"Each run entry must be an object, got: {entry!r}")
        if "tag" not in entry:
            raise SystemExit(f"Missing 'tag' in run entry: {entry}")
        processed.append(entry)
    return processed


def _record_metrics(metrics_file: Path, metrics: dict):
    with metrics_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(metrics) + "\n")


def _print_metrics(metrics: dict):
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sample_video.py and record metrics.")
    parser.add_argument("--tag", help="Identifier for this run")
    parser.add_argument("--run-dir", default="runs", help="Output directory (default: runs)")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for NVML (default: 0)")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between NVML polls (default: 0.5)")
    parser.add_argument(
        "--batch-file",
        type=Path,
        help="JSON file describing a list of runs to execute sequentially. "
        "Each entry needs 'tag' plus either 'cmd', 'command', 'args', or 'sample_args'.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort batch execution when a run exits with a non-zero status.",
    )
    parser.add_argument(
        "--default-script",
        default="sample_video.py",
        help="Script name to prepend when batch entries provide only 'sample_args'.",
    )
    parser.add_argument(
        "--max-memory-ratio",
        type=float,
        default=None,
        help="Force sample runs to pass --max-memory-fraction (0-1] if not already specified.",
    )
    parser.add_argument("sample_video_args", nargs=argparse.REMAINDER, help="Arguments for sample_video.py")
    args = parser.parse_args()

    if args.batch_file:
        batch_entries = _load_batch(args.batch_file)
        overall_exit = 0
        for entry in batch_entries:
            tag = entry["tag"]
            run_dir = Path(entry.get("run_dir", args.run_dir))
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = _prepare_command({**entry, "script": entry.get("script", args.default_script)})
            entry_fraction = entry.get("max_memory_ratio", None)
            effective_fraction = entry_fraction
            if effective_fraction is None:
                effective_fraction = args.max_memory_ratio
            if effective_fraction is not None:
                if not 0.0 < float(effective_fraction) <= 1.0:
                    raise SystemExit(f"Invalid max_memory_ratio {effective_fraction!r} for run '{tag}'")
                if not any(arg.startswith("--max-memory-fraction") for arg in cmd):
                    cmd = list(cmd) + ["--max-memory-fraction", str(effective_fraction)]

            gpu_index = entry.get("gpu_index", args.gpu_index)
            poll_interval = entry.get("poll_interval", args.poll_interval)

            log_file = run_dir / f"{tag}.log"
            metrics = run_with_metrics(
                cmd,
                log_file,
                device_index=gpu_index,
                interval=poll_interval,
            )
            metrics.update({
                "tag": tag,
                "cmd": " ".join(cmd),
                "run_dir": str(run_dir),
                "gpu_index": gpu_index,
                "poll_interval": poll_interval,
                "max_memory_fraction": float(effective_fraction) if effective_fraction is not None else None,
            })
            metrics_file = run_dir / "metrics.jsonl"
            _record_metrics(metrics_file, metrics)
            _print_metrics(metrics)

            if metrics.get("returncode", 0) != 0:
                overall_exit = metrics["returncode"]
                if args.stop_on_failure:
                    raise SystemExit(overall_exit)
        raise SystemExit(overall_exit)

    if not args.tag:
        raise SystemExit("--tag is required when not running in batch mode.")

    sample_video_args = args.sample_video_args
    if sample_video_args and sample_video_args[0] == '--':
        sample_video_args = sample_video_args[1:]

    if not sample_video_args:
        raise SystemExit("No arguments provided for sample_video.py")

    if args.max_memory_ratio is not None:
        if not 0.0 < float(args.max_memory_ratio) <= 1.0:
            raise SystemExit(f"Invalid --max-memory-ratio {args.max_memory_ratio}. Expected 0 < value <= 1.")
        if not any(arg.startswith("--max-memory-fraction") for arg in sample_video_args):
            sample_video_args = list(sample_video_args) + [
                "--max-memory-fraction",
                str(args.max_memory_ratio),
            ]

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
        "run_dir": str(run_dir),
        "gpu_index": args.gpu_index,
        "poll_interval": args.poll_interval,
        "max_memory_fraction": float(args.max_memory_ratio) if args.max_memory_ratio is not None else None,
    })

    metrics_file = run_dir / "metrics.jsonl"
    _record_metrics(metrics_file, metrics)
    _print_metrics(metrics)
    if metrics["returncode"] != 0:
        raise SystemExit(metrics["returncode"])
