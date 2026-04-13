# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion GRPO: Group Relative Policy Optimization for diffusion image generation models.

This module implements GRPO for flow-matching diffusion models (e.g., Qwen-Image).
The training loop follows the same high-level structure as text GRPO but operates
on SDE denoising trajectories instead of token sequences:

1. GENERATE: Run T-step denoising to produce images + collect trajectory logprobs
2. REWARD:   Score generated images (PickScore, QwenVL, etc.)
3. ADVANTAGE: Per-prompt leave-one-out baseline normalization
4. LOGPROB:  Recompute trajectory logprobs under current policy
5. TRAIN:    Per-timestep clipped policy gradient optimization

Key differences from text GRPO (grpo.py):
- Actions are continuous latent denoising steps, not discrete tokens
- Log-probabilities are Gaussian SDE likelihoods, not categorical token logprobs
- Generation and training share the same FSDP workers (no separate vLLM engine)
- Rewards are per-image (sequence-level), advantages expand to [B, T] timesteps
"""

import time
from collections import defaultdict
from typing import Any, NotRequired, Optional, TypedDict

import ray
import torch
from torch.utils.data import DataLoader, Dataset

from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.algorithms.loss import (
    DiffusionClippedPGLossConfig,
    DiffusionClippedPGLossFn,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.image_reward_environment import (
    ImageRewardEnvConfig,
    ImageRewardEnvironment,
    ImageRewardMetadata,
)
from nemo_rl.models.diffusion import DiffusionPolicyConfig
from nemo_rl.models.diffusion.interfaces import DiffusionTrainDataSpec
from nemo_rl.models.diffusion.policy import DiffusionPolicy
from nemo_rl.utils.logger import Logger


# ---------------------------------------------------------------------------
# Configuration TypedDicts
# ---------------------------------------------------------------------------


class DiffusionGRPOConfig(TypedDict):
    """Configuration for the diffusion GRPO training algorithm."""

    num_prompts_per_step: int  # Number of unique prompts per training step
    num_images_per_prompt: int  # K images generated per prompt (for advantage)
    max_num_epochs: int
    max_num_steps: int
    normalize_rewards: bool  # Normalize advantages by per-prompt std
    use_leave_one_out_baseline: bool  # RLOO baseline for variance reduction
    val_period: int  # Validate every N steps
    seed: int


class DiffusionGRPOLoggerConfig(TypedDict):
    """Logger configuration."""

    log_dir: str
    wandb_enabled: NotRequired[bool]
    wandb_project: NotRequired[str]


class CheckpointingConfig(TypedDict):
    """Checkpointing configuration."""

    enabled: bool
    checkpoint_dir: str
    save_period: int  # Save every N steps


class DataConfig(TypedDict):
    """Data configuration."""

    train_data_path: str
    val_data_path: NotRequired[str]
    input_key: str  # Key in jsonl for the prompt text


class DiffusionMasterConfig(TypedDict):
    """Top-level configuration combining all sub-configs."""

    diffusion_policy: DiffusionPolicyConfig
    loss_fn: DiffusionClippedPGLossConfig
    env: ImageRewardEnvConfig
    data: DataConfig
    diffusion_grpo: DiffusionGRPOConfig
    logger: DiffusionGRPOLoggerConfig
    checkpointing: CheckpointingConfig


class DiffusionGRPOSaveState(TypedDict):
    """Checkpointable training state."""

    current_epoch: int
    total_steps: int
    total_samples: int


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TextPromptDataset(Dataset):
    """Simple dataset that loads text prompts from a file.

    Supports both plain text files (one prompt per line) and JSONL files.

    Args:
        data_path: Path to the prompt file.
        input_key: Key for extracting prompt text from JSONL entries.
    """

    def __init__(self, data_path: str, input_key: str = "prompt"):
        self.prompts = []
        if data_path.endswith(".jsonl"):
            import json

            with open(data_path) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    self.prompts.append(entry[input_key])
        else:
            with open(data_path) as f:
                self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _default_save_state() -> DiffusionGRPOSaveState:
    return DiffusionGRPOSaveState(
        current_epoch=0,
        total_steps=0,
        total_samples=0,
    )


def setup(
    config: DiffusionMasterConfig,
) -> tuple[
    DiffusionPolicy,
    DataLoader,
    DiffusionClippedPGLossFn,
    ImageRewardEnvironment,
    Logger,
    DiffusionGRPOSaveState,
]:
    """Initialize all components for diffusion GRPO training.

    Creates the distributed policy, dataloader, loss function, reward
    environment, and logger based on the master configuration.

    Args:
        config: Full DiffusionMasterConfig.

    Returns:
        Tuple of (policy, dataloader, loss_fn, reward_env, logger, save_state).
    """
    grpo_config = config["diffusion_grpo"]

    # Create Ray virtual cluster
    cluster = RayVirtualCluster(
        num_nodes=1,
        num_gpus_per_node=torch.cuda.device_count(),
    )

    # Create policy
    policy = DiffusionPolicy(cluster=cluster, config=config["diffusion_policy"])

    # Create dataset and dataloader
    dataset = TextPromptDataset(
        data_path=config["data"]["train_data_path"],
        input_key=config["data"].get("input_key", "prompt"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=True,
        drop_last=True,
    )

    # Create loss function
    loss_fn = DiffusionClippedPGLossFn(config["loss_fn"])

    # Create reward environment
    reward_env = ImageRewardEnvironment.remote(config["env"])

    # Create logger
    logger = Logger(config["logger"]["log_dir"])

    save_state = _default_save_state()

    return policy, dataloader, loss_fn, reward_env, logger, save_state


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------


def diffusion_grpo_train(
    policy: DiffusionPolicy,
    dataloader: DataLoader,
    loss_fn: DiffusionClippedPGLossFn,
    reward_env: ImageRewardEnvironment,
    logger: Logger,
    save_state: DiffusionGRPOSaveState,
    config: DiffusionMasterConfig,
) -> None:
    """Main diffusion GRPO training loop.

    For each batch of prompts:
    1. Generate K images per prompt with SDE trajectory tracking
    2. Compute image quality rewards
    3. Compute per-prompt advantages (leave-one-out baseline)
    4. Recompute trajectory logprobs under current policy
    5. Train policy with clipped PG loss
    6. Log metrics and save checkpoints

    Args:
        policy: Distributed diffusion policy.
        dataloader: Iterator over text prompts.
        loss_fn: DiffusionClippedPGLossFn instance.
        reward_env: Ray remote reward environment.
        logger: Metrics logger.
        save_state: Resumable training state.
        config: Full master configuration.
    """
    grpo_config = config["diffusion_grpo"]
    ckpt_config = config["checkpointing"]
    num_images_per_prompt = grpo_config["num_images_per_prompt"]
    max_steps = grpo_config["max_num_steps"]
    max_epochs = grpo_config["max_num_epochs"]

    # Advantage estimator (reuse existing NeMo-RL GRPO estimator)
    adv_estimator = GRPOAdvantageEstimator(
        estimator_config={
            "normalize_rewards": grpo_config["normalize_rewards"],
            "use_leave_one_out_baseline": grpo_config["use_leave_one_out_baseline"],
        },
        loss_config={},
    )

    total_steps = save_state["total_steps"]
    current_epoch = save_state["current_epoch"]

    print(f"Starting diffusion GRPO training from step {total_steps}")

    while current_epoch < max_epochs and total_steps < max_steps:
        for batch_prompts in dataloader:
            if total_steps >= max_steps:
                break

            step_start_time = time.time()
            step_metrics = {}

            # ---------------------------------------------------------------
            # 1. GENERATE: K images per prompt
            # ---------------------------------------------------------------
            # Repeat each prompt K times for multi-sample generation
            expanded_prompts = []
            for prompt in batch_prompts:
                expanded_prompts.extend([prompt] * num_images_per_prompt)

            gen_start = time.time()
            trajectory = policy.diffusion_generate(
                prompts=expanded_prompts,
            )
            step_metrics["gen_time"] = time.time() - gen_start

            # ---------------------------------------------------------------
            # 2. REWARD: Score generated images
            # ---------------------------------------------------------------
            reward_start = time.time()
            batch_size = trajectory["images"].shape[0]
            metadata = [
                ImageRewardMetadata(
                    images=trajectory["images"],
                    prompts=expanded_prompts,
                )
            ]
            env_return = ray.get(
                reward_env.step.remote(
                    message_log_batch=[{}] * batch_size,
                    metadata=metadata,
                )
            )
            rewards = env_return.rewards  # [B]
            step_metrics["reward_time"] = time.time() - reward_start
            step_metrics["reward_mean"] = float(rewards.mean().item())
            step_metrics["reward_std"] = float(rewards.std().item())

            # ---------------------------------------------------------------
            # 3. ADVANTAGE: Per-prompt baseline normalization
            # ---------------------------------------------------------------
            # Create prompt IDs for grouping (same prompt = same ID)
            prompt_ids = torch.arange(len(batch_prompts)).repeat_interleave(
                num_images_per_prompt
            )

            # Timestep mask: all ones for SDE window timesteps
            num_window_steps = trajectory["log_probs"].shape[1]
            timestep_mask = torch.ones(batch_size, num_window_steps)

            advantages = adv_estimator.compute_advantage(
                prompt_ids=prompt_ids,
                rewards=rewards,
                mask=timestep_mask,
            )  # [B, T]

            # ---------------------------------------------------------------
            # 4. LOGPROB INFERENCE: Recompute under current policy
            # ---------------------------------------------------------------
            lp_start = time.time()
            prev_logprobs, curr_means, curr_std_devs = policy.get_logprobs(
                latents=trajectory["latents"],
                timesteps=trajectory["timesteps"],
                prompt_embeds=trajectory["prompt_embeds"],
                prompt_embeds_mask=trajectory["prompt_embeds_mask"],
                negative_prompt_embeds=trajectory["negative_prompt_embeds"],
                negative_prompt_embeds_mask=trajectory["negative_prompt_embeds_mask"],
            )
            step_metrics["lp_inference_time"] = time.time() - lp_start

            # ---------------------------------------------------------------
            # 5. Assemble training data
            # ---------------------------------------------------------------
            train_data = DiffusionTrainDataSpec(
                latents=trajectory["latents"],
                generation_logprobs=trajectory["log_probs"],
                prev_logprobs=prev_logprobs,
                advantages=advantages,
                timestep_mask=timestep_mask.to(trajectory["latents"].device),
                sample_mask=torch.ones(batch_size, device=trajectory["latents"].device),
                timesteps=trajectory["timesteps"],
                prompt_embeds=trajectory["prompt_embeds"],
                prompt_embeds_mask=trajectory["prompt_embeds_mask"],
                negative_prompt_embeds=trajectory["negative_prompt_embeds"],
                negative_prompt_embeds_mask=trajectory["negative_prompt_embeds_mask"],
            )

            # Optional: reference model for KL regularization
            if config["loss_fn"]["kl_penalty"] > 0:
                ref_means, ref_std_devs = policy.get_reference_logprobs(
                    latents=trajectory["latents"],
                    timesteps=trajectory["timesteps"],
                    prompt_embeds=trajectory["prompt_embeds"],
                    prompt_embeds_mask=trajectory["prompt_embeds_mask"],
                    negative_prompt_embeds=trajectory["negative_prompt_embeds"],
                    negative_prompt_embeds_mask=trajectory[
                        "negative_prompt_embeds_mask"
                    ],
                )
                train_data["reference_policy_mean"] = ref_means
                train_data["current_policy_mean"] = curr_means
                train_data["std_dev"] = curr_std_devs

            # ---------------------------------------------------------------
            # 6. TRAIN: Policy gradient update
            # ---------------------------------------------------------------
            train_start = time.time()
            train_metrics = policy.train(data=train_data, loss_fn=loss_fn)
            step_metrics["train_time"] = time.time() - train_start
            step_metrics.update(train_metrics)

            # ---------------------------------------------------------------
            # 7. LOG & CHECKPOINT
            # ---------------------------------------------------------------
            step_metrics["step"] = total_steps
            step_metrics["epoch"] = current_epoch
            step_metrics["step_time"] = time.time() - step_start_time
            step_metrics["num_samples"] = batch_size

            logger.log_metrics(step_metrics, step=total_steps)

            if total_steps % 10 == 0:
                print(
                    f"Step {total_steps}: "
                    f"loss={step_metrics.get('loss', 'N/A'):.4f}, "
                    f"reward={step_metrics['reward_mean']:.4f}, "
                    f"time={step_metrics['step_time']:.1f}s"
                )

            # Checkpoint
            if (
                ckpt_config["enabled"]
                and (total_steps + 1) % ckpt_config["save_period"] == 0
            ):
                policy.save_checkpoint(
                    save_dir=ckpt_config["checkpoint_dir"],
                    step=total_steps,
                )

            total_steps += 1
            save_state["total_steps"] = total_steps
            save_state["total_samples"] += batch_size

        current_epoch += 1
        save_state["current_epoch"] = current_epoch

    print(f"Training complete. Total steps: {total_steps}, epochs: {current_epoch}")

    # Final checkpoint
    if ckpt_config["enabled"]:
        policy.save_checkpoint(
            save_dir=ckpt_config["checkpoint_dir"],
            step=total_steps,
        )

    policy.shutdown()
