#!/usr/bin/env bash
# Smoke driver for diffusion-GRPO (Qwen-Image tiny) on a single GPU.
#
# Runs 5 training steps of `diffusion_grpo_train` against
# `tiny-random/Qwen-Image` with the `jpeg_compressibility` reward (a real
# optimizable signal), then asserts metric health (ratio window, grad norm)
# via check_metrics. Designed for a 16 GB GPU.
#
# This script is wired into `tests/functional/L1_Functional_Tests_Other_2.sh`;
# the CI-facing nightly recipes live at
# `tests/test_suites/diffusion/grpo-qwen-image*-1n8g-dp8-lora.sh`.
#
# Usage:
#   bash tests/functional/diffusion_grpo_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-results/diffusion_grpo_smoke/$(date +%Y%m%d_%H%M%S)}"
# Timestamped per run: a reused checkpoint dir would auto-resume at step 5
# and the run would train 0 steps.
CKPT_DIR="${CKPT_DIR:-$LOG_DIR/ckpts}"
JSON_METRICS="$LOG_DIR/metrics.json"
mkdir -p "$LOG_DIR"
echo "Logging to: $LOG_DIR"

# --extra diffusion: diffusers/peft live behind the optional extra and are
# not part of the base CI environment.
PATH="$HOME/.local/bin:$PATH" uv run --frozen --extra diffusion python \
  examples/run_diffusion_grpo.py \
  --config examples/configs/diffusion_grpo_qwen_image_tiny_jpeg.yaml \
  logger.log_dir="$LOG_DIR" \
  logger.tensorboard_enabled=True \
  checkpointing.checkpoint_dir="$CKPT_DIR" \
  2>&1 | tee "$LOG_DIR/run.log"

# The config trains 5 steps with save_period=5, so step_5 must exist.
CHECKPOINT_DIR="$CKPT_DIR/step_5"
if [[ ! -e "$CHECKPOINT_DIR/adapter_model.safetensors" ]] && \
   [[ ! -e "$CHECKPOINT_DIR/transformer.pt" ]]; then
  echo "FAILED: $CHECKPOINT_DIR did not produce a LoRA adapter or full-state checkpoint" >&2
  exit 1
fi

# Convert tensorboard logs to json and assert metric health.
# No --frozen here: check_metrics.py is a PEP 723 inline-metadata script and
# --frozen would demand a per-script lockfile.
uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    "median(data['train/mean_ratio']) > 0.5" \
    "median(data['train/mean_ratio']) < 1.5" \
    "max(data['train/grad_norm']) < 100"

echo "Smoke OK: checkpoint at $CHECKPOINT_DIR, metrics healthy."
