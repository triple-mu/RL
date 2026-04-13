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

"""Data type definitions for diffusion RL training.

These TypedDicts define the data formats flowing through the diffusion GRPO
pipeline: generation trajectories, training data, and intermediate results.
"""

from typing import NotRequired, TypedDict

import torch


class DiffusionTrajectorySpec(TypedDict):
    """Output of diffusion generation -- a denoising trajectory with log-probabilities.

    Produced by DiffusionGenerationPipeline.generate_trajectory().
    The latents tensor has T+1 entries (initial noise + T denoising steps),
    while log_probs has T entries (one per transition).
    Only timesteps within the SDE window have meaningful log_probs;
    others are filled with zeros.

    Shapes:
        latents:                       [B, T_window+1, C, H, W]
        log_probs:                     [B, T_window]
        timesteps:                     [T_window]
        images:                        [B, 3, H_img, W_img]
        prompt_embeds:                 [B, L, D]
        prompt_embeds_mask:            [B, L]
        negative_prompt_embeds:        [B, L, D]
        negative_prompt_embeds_mask:   [B, L]
    """

    latents: torch.Tensor
    log_probs: torch.Tensor
    timesteps: torch.Tensor
    images: torch.Tensor
    prompt_text: list[str]
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_embeds_mask: torch.Tensor


class DiffusionTrainDataSpec(TypedDict):
    """Data dict for a single diffusion GRPO training step.

    Assembled from DiffusionTrajectorySpec after reward and advantage computation.
    The loss function operates over the timestep dimension T (analogous to
    the sequence-length dimension in LLM GRPO).

    Shapes:
        latents:              [B, T_window+1, C, H, W]  (current + next latents)
        generation_logprobs:  [B, T_window]  logprobs from generation rollout
        prev_logprobs:        [B, T_window]  logprobs from current policy (pre-train inference)
        advantages:           [B, T_window]  expanded per-sample advantages
        timestep_mask:        [B, T_window]  1 for valid SDE window steps
        sample_mask:          [B]            1 for valid samples, 0 for padding
        timesteps:            [T_window]     timestep indices for SDE steps
        prompt_embeds:        [B, L, D]      text embeddings
        prompt_embeds_mask:   [B, L]
        negative_prompt_embeds:      [B, L, D]
        negative_prompt_embeds_mask: [B, L]
        reference_policy_mean: [B, T_window, C, H, W]  reference model mean predictions (optional)
        std_dev:               [B, T_window, C, H, W]  SDE std dev per step (optional, for KL)
    """

    latents: torch.Tensor
    generation_logprobs: torch.Tensor
    prev_logprobs: torch.Tensor
    advantages: torch.Tensor
    timestep_mask: torch.Tensor
    sample_mask: torch.Tensor
    timesteps: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_embeds_mask: torch.Tensor
    reference_policy_mean: NotRequired[torch.Tensor]
    std_dev: NotRequired[torch.Tensor]
