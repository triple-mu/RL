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
from typing import Any, NotRequired, Protocol, TypedDict

import torch


class DiffusionPipelineCfg(TypedDict):
    """Per-rollout pipeline knobs aligned with verl-omni `DiffusionPipelineConfig`."""

    height: int
    width: int
    num_inference_steps: int
    true_cfg_scale: float
    max_sequence_length: int
    guidance_scale: NotRequired[float | None]


class DiffusionAlgoCfg(TypedDict):
    """SDE rollout algorithm knobs aligned with verl-omni `DiffusionRolloutAlgoConfig`."""

    noise_level: float
    sde_type: str
    sde_window_size: NotRequired[int | None]
    sde_window_range: NotRequired[list[int] | None]


class DiffusionLoraCfg(TypedDict):
    """LoRA adapter configuration."""

    enabled: bool
    rank: int
    alpha: int
    target_modules: list[str]
    dropout: NotRequired[float]
    exclude_modules: NotRequired[list[str] | None]


class DiffusionPolicyConfig(TypedDict):
    """Top-level configuration for the diffusion policy/worker.

    Fields mirror `examples/configs/diffusion_grpo_qwen_image*.yaml`. Defaults live
    in YAML per `config-conventions`; this TypedDict only declares types.
    """

    model_name: str
    precision: str
    train_global_batch_size: int
    train_micro_batch_size: int
    enable_gradient_checkpointing: bool
    optimizer: dict[str, Any]
    pipeline: DiffusionPipelineCfg
    algo: DiffusionAlgoCfg
    lora_cfg: DiffusionLoraCfg
    reference_transformer_enabled: NotRequired[bool]
    seed: NotRequired[int]


class DiffusionGRPOAlgoConfig(TypedDict):
    """Top-level diffusion-GRPO training-loop config."""

    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_steps: int
    val_period: int
    seed: int
    ppo_epochs: NotRequired[int]
    val_at_start: NotRequired[bool]
    val_at_end: NotRequired[bool]
    max_val_samples: NotRequired[int]
    use_leave_one_out_baseline: NotRequired[bool]
    normalize_rewards: NotRequired[bool]


class DiffusionLossConfig(TypedDict):
    """Diffusion-GRPO loss knobs aligned with verl-omni `FlowGRPOLoss` config."""

    ratio_clip_min: float
    ratio_clip_max: float
    adv_clip_max: float
    beta: float
    # If True, sum logprobs over the T dimension before computing the ratio,
    # so the loss is `mean_B(-adv_B * ratio_B)`. This matches verl-omni's
    # 1-D-per-sample formulation. Default False uses per-(B, T) elements
    # (Flow-GRPO paper formulation).
    aggregate_logprobs_per_sample: NotRequired[bool]


class DiffusionDatumSpec(TypedDict):
    prompt: str
    negative_prompt: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]
    idx: int
    loss_multiplier: float
    task_name: NotRequired[str]


class DiffusionTrajectorySpec(TypedDict):
    prompts: list[str]
    negative_prompts: list[str]
    metadata: list[dict[str, Any]]
    images: torch.Tensor
    latents: torch.Tensor
    timesteps: torch.Tensor
    generation_logprobs: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_embeds_mask: torch.Tensor


class DiffusionTrainDataSpec(TypedDict):
    latents: torch.Tensor
    timesteps: torch.Tensor
    generation_logprobs: torch.Tensor
    advantages: torch.Tensor
    timestep_mask: torch.Tensor
    sample_mask: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_embeds_mask: torch.Tensor
    reference_policy_mean: NotRequired[torch.Tensor]
    current_policy_mean: NotRequired[torch.Tensor]
    std_dev: NotRequired[torch.Tensor]


class DiffusionPipelineAdapter(Protocol):
    def encode_condition(
        self, prompts: list[str], negative_prompts: list[str]
    ) -> dict[str, torch.Tensor]: ...

    def sample_trajectory(
        self,
        prompts: list[str],
        negative_prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> DiffusionTrajectorySpec: ...

    def compute_transition_logprob(
        self, data: DiffusionTrainDataSpec
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...
