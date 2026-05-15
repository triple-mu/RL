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
