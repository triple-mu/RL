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

"""SDE denoising step with log-probability computation for flow matching models.

Implements the reverse-time SDE step used in diffusion GRPO training.
The log-probability measures how likely a transition x_t -> x_{t+1} is under
the current policy (denoising model), enabling policy gradient optimization.

Adapted from flow_grpo (https://github.com/...) which itself adapts from
ddpo-pytorch (https://github.com/kvablack/ddpo-pytorch).
"""

import math
from typing import Optional, Union

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor


def sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform one reverse-time SDE step and compute the transition log-probability.

    Given the model's velocity prediction and the current latent state, computes:
    1. The mean of the reverse SDE transition distribution
    2. The next latent state (sampled or provided)
    3. The log-probability of the transition under a Gaussian distribution

    The SDE formulation adds controlled stochasticity to the ODE flow matching
    trajectory, enabling meaningful log-probability computation for policy gradients.

    When noise_level=0, this reduces to a deterministic ODE step with undefined
    log-probability (returns zeros).

    Args:
        scheduler: Flow matching Euler discrete scheduler with precomputed sigmas.
        model_output: Predicted velocity/noise from the denoising model. Shape [B, C, H, W].
        timestep: Current timestep values. Shape [B].
        sample: Current latent state x_t. Shape [B, C, H, W].
        noise_level: Controls SDE stochasticity. 0 = deterministic ODE, >0 = stochastic SDE.
        prev_sample: If provided, compute log-prob for this specific next state
            instead of sampling a new one. Used during training to evaluate
            log-prob of previously generated trajectories. Shape [B, C, H, W].
        generator: Random number generator for reproducibility.

    Returns:
        prev_sample: Next latent state x_{t+1}. Shape [B, C, H, W].
        log_prob: Log-probability of the transition, averaged over spatial dims. Shape [B].
        prev_sample_mean: Mean of the transition distribution. Shape [B, C, H, W].
        std_dev_t: Standard deviation of the SDE noise. Shape broadcastable to [B, C, H, W].
    """
    # bf16 can overflow when computing prev_sample_mean; use fp32
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    # Look up sigma values for current and next timestep
    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]
    sigma = scheduler.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = scheduler.sigmas[prev_step_index].view(
        -1, *([1] * (len(sample.shape) - 1))
    )
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma

    # Compute SDE diffusion coefficient
    # std_dev_t = sqrt(sigma / (1 - sigma)) * noise_level
    # Avoid division by zero when sigma=1 by substituting sigma_max
    std_dev_t = (
        torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
        * noise_level
    )

    # Compute the mean of the reverse SDE transition
    # This is the drift term: x_{t+1} = mean + noise
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    # Sample or use provided next state
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = (
            prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
        )

    # Compute Gaussian log-probability:
    # log p(x_{t+1} | x_t) = -||x_{t+1} - mu||^2 / (2 * sigma_noise^2)
    #                         - log(sigma_noise) - 0.5 * log(2*pi)
    # where sigma_noise = std_dev_t * sqrt(|dt|)
    sigma_noise = std_dev_t * torch.sqrt(-1 * dt)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * sigma_noise**2)
        - torch.log(sigma_noise)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # Average log-prob over all spatial/channel dimensions -> shape [B]
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
