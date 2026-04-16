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

"""DiffusionPolicy: local single-worker coordinator for diffusion policy.

Wraps a DiffusionPolicyWorkerImpl to provide the high-level API used
by the training loop (generate, get_logprobs, train, checkpoint).
Runs on a single GPU without Ray.
"""

from typing import Any, Optional

import torch

from nemo_rl.models.diffusion import DiffusionPolicyConfig
from nemo_rl.models.diffusion.interfaces import (
    DiffusionTrainDataSpec,
    DiffusionTrajectorySpec,
)


class DiffusionPolicy:
    """Single-GPU diffusion policy coordinator.

    Creates a DiffusionPolicyWorkerImpl on the local device and delegates
    all operations to it.
    """

    def __init__(self, config: DiffusionPolicyConfig):
        from nemo_rl.models.diffusion.workers.diffusion_worker import (
            DiffusionPolicyWorkerImpl,
        )

        self.config = config
        self.worker = DiffusionPolicyWorkerImpl(config)

    def diffusion_generate(
        self,
        prompts: list[str],
        negative_prompts: Optional[list[str]] = None,
    ) -> DiffusionTrajectorySpec:
        return self.worker.generate(
            prompts=prompts,
            negative_prompts=negative_prompts,
        )

    def get_logprobs(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.worker.get_logprobs(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        )

    def get_reference_logprobs(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.worker.get_reference_logprobs(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        )

    def train(
        self,
        data: DiffusionTrainDataSpec,
        loss_fn,
    ) -> dict[str, Any]:
        return self.worker.train(data=data, loss_fn=loss_fn)

    def save_checkpoint(self, save_dir: str, step: int):
        self.worker.save_checkpoint(save_dir=save_dir, step=step)

    def shutdown(self) -> bool:
        return self.worker.shutdown()
