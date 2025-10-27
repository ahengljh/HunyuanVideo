#!/usr/bin/env bash
set -euo pipefail
PROMPT="$1"
SAVE_DIR="$2"
SEED=${3:-42}
CACHE_THRESH=${4:-0.05}
CACHE_MIN_IMP=${5:-0.001}
CACHE_STAGE=${6:-single}
python sample_video.py \
  --prompt "$PROMPT" \
  --video-size 512 512 \
  --video-length 33 \
  --infer-steps 30 \
  --cfg-scale 4.0 \
  --seed "$SEED" \
  --save-path "$SAVE_DIR" \
  --model-base ckpts \
  --rabbit-enable \
  --rabbit-offload-mode weights \
  --rabbit-offload-plan auto:0.3 \
  --rabbit-memory-budget-mb 6000 \
  --rabbit-min-device-blocks 4 \
  --rabbit-prefetch-distance 2 \
  --rabbit-cache-outputs \
  --rabbit-cache-device cuda \
  --rabbit-cache-threshold "$CACHE_THRESH" \
  --rabbit-cache-max-age 4 \
  --rabbit-cache-min-importance "$CACHE_MIN_IMP" \
  --rabbit-cache-stage "$CACHE_STAGE" \
  --rabbit-latent-offload \
  --rabbit-latent-offload-device cpu \
  --rabbit-skip-strategy none \
  --rabbit-log-stats
