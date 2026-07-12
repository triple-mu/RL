#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=50
MAX_STEPS=50
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=180
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# The recipe inherits data.{train,val}.prompt_file from the exemplar config;
# generate the prompt files on first use.
if [[ ! -f examples/data/diffusion/train_prompts.jsonl ]]; then
  uv run python tools/export_diffusion_prompts.py \
    --dataset yuvalkirstain/pickapic_v2_no_images --split train --column caption \
    --train-size 4000 --val-size 64 --out-dir examples/data/diffusion
fi

uv run --extra diffusion examples/run_diffusion_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Diffusion-GRPO logs 0-based steps, so the last step key is MAX_STEPS - 1.
LAST_STEP=$((MAX_STEPS - 1))
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $LAST_STEP ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        "median(data['train/mean_ratio']) > 0.5" \
        "median(data['train/mean_ratio']) < 1.5" \
        "data['train/loss']['$LAST_STEP'] < 100" \
        "data['train/loss']['$LAST_STEP'] > -100"

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
