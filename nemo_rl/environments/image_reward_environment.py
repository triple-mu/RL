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

"""Image reward scoring for diffusion GRPO training.

Provides pluggable scoring functions for generated images.
Currently ships a fake random scorer for pipeline validation;
real scorers (PickScore, etc.) can be added by extending
``_load_reward_fn``.
"""

import random
from typing import TypedDict

import torch
from torch import Tensor


class RewardFunctionConfig(TypedDict):
    """Configuration for a single reward function."""

    name: str  # "random", "pickscore", etc.
    weight: float
    kwargs: dict


class ImageRewardEnvConfig(TypedDict):
    """Configuration for the image reward environment."""

    reward_functions: list[RewardFunctionConfig]
    device: str


class ImageRewardMetadata(TypedDict):
    """Metadata containing generated images and prompts."""

    images: Tensor  # [B, 3, H, W]
    prompts: list[str]


def _load_reward_fn(config: RewardFunctionConfig, device: str):
    """Load a reward scoring function by name.

    Args:
        config: Reward function configuration.
        device: Device for the reward model.

    Returns:
        Callable: (images, prompts) -> list[float]
    """
    name = config["name"]

    if name == "random":

        def score_fn(images, prompts):
            return [random.random() for _ in range(len(prompts))]

        return score_fn

    elif name == "pickscore":
        from nemo_rl.environments.image_scorers import PickScoreScorer

        scorer = PickScoreScorer(dtype=torch.float32, device=device)

        def score_fn(images, prompts):
            scores = scorer(prompts, images)
            return scores.tolist()

        return score_fn

    else:
        raise ValueError(
            f"Unknown reward function: {name}. Supported: 'random', 'pickscore'."
        )


def compute_image_rewards(
    images: Tensor,
    prompts: list[str],
    config: ImageRewardEnvConfig,
) -> Tensor:
    """Score images using configured reward functions (no Ray).

    Args:
        images: Generated images [B, 3, H, W].
        prompts: Text prompts used for generation.
        config: Reward environment config with function list.

    Returns:
        Rewards tensor of shape [B].
    """
    batch_size = images.shape[0]
    device = config.get("device", "cpu")
    total_scores = [0.0] * batch_size

    for fn_config in config["reward_functions"]:
        fn = _load_reward_fn(fn_config, device)
        weight = fn_config["weight"]
        scores = fn(images, prompts)
        for i, score in enumerate(scores):
            total_scores[i] += weight * score

    return torch.tensor(total_scores, dtype=torch.float32)
