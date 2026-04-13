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

"""Image reward environment for diffusion GRPO training.

Computes rewards for generated images using pluggable scoring functions
(PickScore, QwenVL aesthetic scoring, OCR accuracy, ImageReward, etc.).
Multiple reward functions can be combined with configurable weights.

Implements EnvironmentInterface to integrate with NeMo-RL's reward pipeline.
Images are passed via the metadata dictionary rather than message_log,
since diffusion models don't use chat-format message logs.
"""

from typing import Any, Optional, TypedDict

import ray
import torch
from torch import Tensor

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class RewardFunctionConfig(TypedDict):
    """Configuration for a single reward function."""

    name: str  # "pickscore", "qwenvl", "ocr", "imagereward", etc.
    weight: float  # Weight for combining multiple rewards
    kwargs: dict  # Additional arguments for the reward function


class ImageRewardEnvConfig(TypedDict):
    """Configuration for the image reward environment."""

    num_workers: int  # Number of Ray workers for parallel reward computation
    reward_functions: list[RewardFunctionConfig]
    device: str  # Device for reward model inference


class ImageRewardMetadata(TypedDict):
    """Metadata passed to the environment containing generated images."""

    images: Tensor  # [B, 3, H, W] generated images as tensors
    prompts: list[str]  # Text prompts used for generation


def _load_reward_fn(config: RewardFunctionConfig, device: str):
    """Load a reward scoring function by name.

    Supported reward functions:
    - pickscore: CLIP-based image-text alignment scoring
    - qwenvl: Qwen2.5-VL aesthetic evaluation
    - imagereward: BLIP-based image-text alignment
    - ocr: OCR text recognition accuracy

    Args:
        config: Reward function configuration.
        device: Device for the reward model.

    Returns:
        Callable that takes (images, prompts) and returns list[float].
    """
    name = config["name"]
    kwargs = config.get("kwargs", {})

    if name == "pickscore":
        from flow_grpo.pickscore_scorer import PickScoreScorer

        scorer = PickScoreScorer(dtype=torch.float32, device=device)

        def score_fn(images, prompts):
            return scorer(prompts, images)

        return score_fn

    elif name == "qwenvl":
        from flow_grpo.qwenvl import QwenVLScorer

        scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

        def score_fn(images, prompts):
            return scorer(prompts, images)

        return score_fn

    elif name == "imagereward":
        import ImageReward as image_reward_lib

        model = image_reward_lib.load("ImageReward-v1.0", device=device)

        def score_fn(images, prompts):
            from torchvision.transforms.functional import to_pil_image

            pil_images = [to_pil_image(img.clamp(0, 1)) for img in images]
            scores = []
            for img, prompt in zip(pil_images, prompts):
                score = model.score(prompt, img)
                scores.append(float(score))
            return scores

        return score_fn

    elif name == "ocr":
        from flow_grpo.ocr_scorer import OcrScorer

        scorer = OcrScorer(device=device, **kwargs)

        def score_fn(images, prompts):
            return scorer(images, prompts)

        return score_fn

    else:
        raise ValueError(f"Unknown reward function: {name}")


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ImageRewardEnvironment(EnvironmentInterface[ImageRewardMetadata]):
    """Environment that scores generated images using configurable reward functions.

    Computes weighted multi-reward scores from generated images and their
    corresponding text prompts. Supports multiple reward functions that
    can be combined with configurable weights.

    Args:
        config: ImageRewardEnvConfig with reward function specifications.
    """

    def __init__(self, config: ImageRewardEnvConfig):
        self.config = config
        self.device = config.get("device", "cuda")
        self.reward_fns = []
        self.reward_weights = []

        for fn_config in config["reward_functions"]:
            fn = _load_reward_fn(fn_config, self.device)
            self.reward_fns.append((fn_config["name"], fn))
            self.reward_weights.append(fn_config["weight"])

    def step(
        self,
        message_log_batch: list,
        metadata: list[ImageRewardMetadata],
    ) -> EnvironmentReturn[ImageRewardMetadata]:
        """Compute rewards for generated images.

        Images and prompts are extracted from the metadata list.
        Each metadata item should contain 'images' (tensor) and 'prompts' (list[str]).

        Args:
            message_log_batch: Unused for diffusion (kept for interface compatibility).
            metadata: List of ImageRewardMetadata dicts with images and prompts.

        Returns:
            EnvironmentReturn with computed rewards.
        """
        # Extract images and prompts from metadata
        all_images = []
        all_prompts = []
        for meta in metadata:
            all_images.append(meta["images"])
            all_prompts.extend(meta["prompts"])

        images = torch.cat(all_images, dim=0) if len(all_images) > 1 else all_images[0]
        batch_size = images.shape[0]

        # Convert tensor images to PIL for reward functions that need it
        from torchvision.transforms.functional import to_pil_image

        pil_images = [to_pil_image(img.clamp(0, 1).cpu()) for img in images]

        # Compute weighted multi-reward
        total_scores = [0.0] * batch_size
        reward_details = {}

        for (name, fn), weight in zip(self.reward_fns, self.reward_weights):
            scores = fn(pil_images, all_prompts)
            reward_details[name] = scores
            for i, score in enumerate(scores):
                total_scores[i] += weight * score

        rewards = torch.tensor(total_scores, dtype=torch.float32)

        # Return standard EnvironmentReturn
        observations = [{"role": "system", "content": ""}] * batch_size
        terminateds = torch.ones(batch_size, dtype=torch.bool)
        updated_metadata = metadata

        return EnvironmentReturn(
            observations=observations,
            metadata=updated_metadata,
            next_stop_strings=[None] * batch_size,
            rewards=rewards,
            terminateds=terminateds,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Compute aggregate reward statistics.

        Args:
            batch: The full batch with rewards.

        Returns:
            Tuple of (batch, metrics_dict).
        """
        metrics = {}
        if "total_reward" in batch:
            rewards = batch["total_reward"]
            metrics["reward_mean"] = float(rewards.mean().item())
            metrics["reward_std"] = float(rewards.std().item())
            metrics["reward_min"] = float(rewards.min().item())
            metrics["reward_max"] = float(rewards.max().item())
        return batch, metrics
