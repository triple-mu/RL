# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Diffusion-GRPO training loop.

Mirrors :func:`nemo_rl.algorithms.grpo.grpo_train` in phase ordering
(cluster→policy→env→dataset→loop[rollout→reward→advantage→train→validate→checkpoint])
but drops token-only concepts (vLLM refit, message-log rollouts, token KL).

The advantage estimator reuses
:func:`nemo_rl.algorithms.utils.calculate_baseline_and_std_per_prompt` by
encoding each unique prompt as a ``(1,)`` integer tensor row, then
broadcasting the resulting per-sample advantage to ``[B*K, T]``.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

import torch

from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.image_reward_environment import ImageRewardEnvironment
from nemo_rl.models.diffusion.interfaces import (
    DiffusionGRPOAlgoConfig,
    DiffusionLossConfig,
    DiffusionPolicyConfig,
    DiffusionTrainDataSpec,
    DiffusionTrajectorySpec,
)
from nemo_rl.models.diffusion.policy import DiffusionPolicy
from nemo_rl.models.diffusion.sde import compute_window_mask
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import Timer


def _prompt_ids_for_baseline(rep_prompts: list[str]) -> torch.Tensor:
    """Encode prompts as a `(B*K, 1)` integer tensor; identical prompts share an id."""
    seen: dict[str, int] = {}
    ids: list[int] = []
    for p in rep_prompts:
        if p not in seen:
            seen[p] = len(seen)
        ids.append(seen[p])
    return torch.tensor(ids, dtype=torch.long).unsqueeze(-1)


def _compute_advantages(
    rep_prompts: list[str],
    rewards: torch.Tensor,
    *,
    use_leave_one_out_baseline: bool = True,
) -> torch.Tensor:
    """Group-relative advantage per the GRPO recipe."""
    prompt_ids = _prompt_ids_for_baseline(rep_prompts)
    valid_mask = torch.ones_like(rewards)
    baseline, std = calculate_baseline_and_std_per_prompt(
        prompts=prompt_ids,
        rewards=rewards,
        valid_mask=valid_mask,
        leave_one_out_baseline=use_leave_one_out_baseline,
    )
    return (rewards - baseline) / std.clamp_min(1e-6)


def _build_train_data(
    traj: DiffusionTrajectorySpec,
    advantages_per_sample: torch.Tensor,
    *,
    policy_cfg: DiffusionPolicyConfig,
    loss_multiplier: torch.Tensor,
) -> BatchedDataDict[DiffusionTrainDataSpec]:
    T = traj["timesteps"].shape[-1]
    gen_lp = traj["generation_logprobs"]
    B = gen_lp.shape[0]
    timestep_mask_1d = compute_window_mask(
        T,
        window_start=int((policy_cfg["algo"].get("sde_window_range") or [0, T])[0]),
        window_size=policy_cfg["algo"].get("sde_window_size") or T,
    )
    # Always keep timestep_mask at [B, T] — the loss broadcasts to extra
    # latent-token dims as needed (per-element mode produces [B, T, N, C]).
    timestep_mask = timestep_mask_1d.unsqueeze(0).expand(B, -1)
    advantages = advantages_per_sample.unsqueeze(-1).expand(-1, T)

    data: DiffusionTrainDataSpec = {
        "latents": traj["latents"],
        "timesteps": traj["timesteps"],
        "generation_logprobs": traj["generation_logprobs"],
        "advantages": advantages,
        "timestep_mask": timestep_mask,
        "sample_mask": loss_multiplier,
        "prompt_embeds": traj["prompt_embeds"],
        "prompt_embeds_mask": traj["prompt_embeds_mask"],
        "negative_prompt_embeds": traj["negative_prompt_embeds"],
        "negative_prompt_embeds_mask": traj["negative_prompt_embeds_mask"],
    }
    return BatchedDataDict(data)


def diffusion_grpo_train(
    policy: DiffusionPolicy,
    env: ImageRewardEnvironment,
    train_dataloader: Iterable[BatchedDataDict[Any]],
    val_dataloader: Iterable[BatchedDataDict[Any]] | None,
    *,
    algo_cfg: DiffusionGRPOAlgoConfig,
    loss_cfg: DiffusionLossConfig,
    policy_cfg: DiffusionPolicyConfig,
    logger: Logger,
    checkpoint_dir: str | None,
    save_period: int = 0,
) -> None:
    timer = Timer()
    K = algo_cfg["num_generations_per_prompt"]
    max_steps = algo_cfg["max_num_steps"]
    seed_base = int(algo_cfg.get("seed", 0))

    train_iter = iter(train_dataloader)
    for step in range(max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        with timer.time("rollout"):
            traj = policy.sample_trajectory(
                prompts=batch["prompts"],
                negative_prompts=batch["negative_prompts"],
                metadata=batch["metadata"],
                K=K,
                seed=seed_base + step,
            )

        with timer.time("reward"):
            # The reward pool is CPU by default in the smoke config; move
            # the GPU-side images to CPU before the Ray hop to keep CUDA
            # tensors local to the trainer worker.
            images_cpu = traj["images"].detach().to("cpu")
            rewards, reward_metrics = env.score_images(
                images_cpu, traj["prompts"], traj["metadata"]
            )

        with timer.time("advantage"):
            advantages_per_sample = _compute_advantages(
                traj["prompts"],
                rewards,
                use_leave_one_out_baseline=bool(
                    algo_cfg.get("use_leave_one_out_baseline", True)
                ),
            )

        loss_mult = torch.tensor(
            [m.get("loss_multiplier", 1.0) for m in traj["metadata"]],
            dtype=torch.float32,
        ) if traj["metadata"] else torch.ones(rewards.shape[0])

        train_data = _build_train_data(
            traj,
            advantages_per_sample,
            policy_cfg=policy_cfg,
            loss_multiplier=loss_mult,
        )

        ppo_epochs = int(algo_cfg.get("ppo_epochs", 1))
        with timer.time("train"):
            per_epoch_metrics = []
            for _epoch in range(ppo_epochs):
                per_epoch_metrics.append(policy.train(train_data, loss_cfg))
            # Aggregate: keep last-epoch values for ratio/loss (those are
            # the most informative since they reflect the largest policy drift),
            # but record per-epoch losses too.
            train_metrics = dict(per_epoch_metrics[-1])
            if ppo_epochs > 1:
                for i, m in enumerate(per_epoch_metrics):
                    train_metrics[f"epoch_{i}/loss"] = m.get("loss", 0.0)
                    train_metrics[f"epoch_{i}/mean_ratio"] = m.get(
                        "mean_ratio", 1.0
                    )

        metrics = {
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **reward_metrics,
            "train/advantage_mean": float(advantages_per_sample.mean().item()),
            "train/advantage_std": float(advantages_per_sample.std().item()),
            "train/reward_mean": float(rewards.mean().item()),
        }
        for tag, dur in timer.get_timing_metrics(reduction_op="mean").items():
            if isinstance(dur, float):
                metrics[f"timing/{tag}_s"] = dur
        timer.reset()
        logger.log_metrics(metrics, step=step)
        # Console echo so smoke + interactive runs always have observable
        # progress even when every logger backend is disabled. Report both
        # the final-epoch loss (signal under our reporting convention) and
        # the average-across-inner-epochs loss (matches verl-omni's console
        # aggregation, so the two stacks can be diffed directly).
        loss_val = metrics.get("train/loss")
        ratio_val = metrics.get("train/mean_ratio")
        rew_val = metrics.get("train/reward_mean")
        epoch_losses = [
            metrics.get(f"train/epoch_{i}/loss")
            for i in range(ppo_epochs)
            if f"train/epoch_{i}/loss" in metrics
        ]
        avg_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else loss_val
        )
        print(
            f"[diffusion_grpo] step={step} "
            f"train/loss_last={loss_val} train/loss_avg={avg_loss} "
            f"train/mean_ratio={ratio_val} train/reward_mean={rew_val}",
            flush=True,
        )

        if (
            checkpoint_dir is not None
            and save_period > 0
            and (step + 1) % save_period == 0
        ):
            policy.save_checkpoint(os.path.join(checkpoint_dir, f"step_{step + 1}"))

        if val_dataloader is not None:
            val_period = int(algo_cfg.get("val_period", 0))
            if val_period > 0 and (step + 1) % val_period == 0:
                _run_validation(policy, env, val_dataloader, step=step, logger=logger, K=K)


def _run_validation(
    policy: DiffusionPolicy,
    env: ImageRewardEnvironment,
    val_dataloader: Iterable[BatchedDataDict[Any]],
    *,
    step: int,
    logger: Logger,
    K: int,
) -> None:
    rewards_acc: list[float] = []
    for batch in val_dataloader:
        traj = policy.sample_trajectory(
            prompts=batch["prompts"],
            negative_prompts=batch["negative_prompts"],
            metadata=batch["metadata"],
            K=K,
            seed=None,
        )
        rewards, _ = env.score_images(
            traj["images"], traj["prompts"], traj["metadata"]
        )
        rewards_acc.append(float(rewards.mean().item()))
    if rewards_acc:
        logger.log_metrics(
            {"val/reward_mean": sum(rewards_acc) / len(rewards_acc)}, step=step
        )
