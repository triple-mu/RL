#!/usr/bin/env bash
# Smoke driver for diffusion-GRPO (Qwen-Image tiny) on a single GPU.
#
# Runs 5 training steps of `diffusion_grpo_train` against
# `tiny-random/Qwen-Image` with `DummyImageReward`. Designed for a 16 GB GPU.
#
# This script is intended to be invoked manually or by the `auto-research`
# campaign harness, not by CI's L0/L1 functional batch (we don't gate on
# diffusion in CI yet).
#
# Usage:
#   bash tests/functional/diffusion_grpo_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-results/diffusion_grpo_smoke/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "Logging to: $LOG_DIR"

PATH="$HOME/.local/bin:$PATH" uv run --frozen python \
  examples/run_diffusion_grpo.py \
  --config examples/configs/diffusion_grpo_qwen_image_tiny.yaml \
  2>&1 | tee "$LOG_DIR/run.log"

# Basic sanity check: 5 console-echo lines AND the step_5 checkpoint exist.
TRAIN_LOSS_COUNT=$(grep -c "\[diffusion_grpo\] step=" "$LOG_DIR/run.log" || true)
CHECKPOINT_DIR="results/diffusion_grpo_smoke/step_5"
if [[ "$TRAIN_LOSS_COUNT" -lt 5 ]]; then
  echo "FAILED: expected >= 5 progress lines, got $TRAIN_LOSS_COUNT" >&2
  exit 1
fi
if [[ ! -e "$CHECKPOINT_DIR/adapter_model.safetensors" ]] && \
   [[ ! -e "$CHECKPOINT_DIR/transformer.pt" ]]; then
  echo "FAILED: $CHECKPOINT_DIR did not produce a LoRA adapter or full-state checkpoint" >&2
  exit 1
fi

echo "Smoke OK: $TRAIN_LOSS_COUNT step lines, checkpoint at $CHECKPOINT_DIR."
