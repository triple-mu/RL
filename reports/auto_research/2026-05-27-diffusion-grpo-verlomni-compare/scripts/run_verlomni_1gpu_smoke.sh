#!/usr/bin/env bash
# verl-omni 1-GPU FlowGRPO smoke, parameters matched to NeMo-RL smoke
# (tiny-random/Qwen-Image, K=2, T=8, 128x128, LoRA rank=4, no KL).
#
# Runs inside the verlai/verl:vllm020.dev1 container which is then patched
# with vllm-omni + verl-omni installs.
set -xeuo pipefail
cd /workspace/verl-omni

NUM_GPUS=1
MODEL_PATH=/workspace/models/tiny-random/Qwen-Image
TOKENIZER_PATH=${MODEL_PATH}/tokenizer
DATA_DIR=/workspace/data/dummy_diffusion
TRAIN_FILES=${DATA_DIR}/train.parquet
VAL_FILES=${DATA_DIR}/test.parquet
TOTAL_TRAIN_STEPS=5

ENGINE=vllm_omni
max_prompt_length=64

n_resp_per_prompt=4
PPO_EPOCHS=${PPO_EPOCHS:-4}
micro_bsz_per_gpu=1
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=$((micro_bsz * n_resp_per_prompt))
train_batch_size=${mini_bsz}

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 4

LOG_DIR=/workspace/reports/verl_omni_1gpu_smoke
mkdir -p "$LOG_DIR"

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.lora_rank=4 \
    actor_rollout_ref.model.lora_alpha=8 \
    "actor_rollout_ref.model.target_modules=['to_q','to_k','to_v']" \
    actor_rollout_ref.actor.optim.lr=1e-2 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_OMNI_GPU_MEM:-0.2} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=8 \
    actor_rollout_ref.rollout.pipeline.height=128 \
    actor_rollout_ref.rollout.pipeline.width=128 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=8 \
    "actor_rollout_ref.rollout.algo.sde_window_range=[0,8]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=8 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowgrpo-1gpu-smoke \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    2>&1 | tee "$LOG_DIR/run.log"

echo "verl-omni smoke complete. Logs at $LOG_DIR/run.log"
