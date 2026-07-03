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
"""Image-native reward environment for diffusion-GRPO.

Does NOT inherit :class:`nemo_rl.environments.interfaces.EnvironmentInterface`:
the latter's ``step(message_log_batch, metadata)`` is token-centric and the
image-reward signal does not need the LLM message-log path. The diffusion
trainer calls :meth:`ImageRewardEnvironment.score_images` directly.

Plugin contract (:class:`BaseImageReward`):

- ``name``: short identifier used in logged metric keys.
- ``weight``: scalar multiplier applied when aggregating into the total reward.
- ``score(images, prompts, metadata) -> dict[str, Tensor]``: each tensor must
  be 1-D with length equal to ``images.shape[0]``. Plugins may emit multiple
  component scores in the dict; aggregation sums all components.
"""

import hashlib
from typing import Any, Callable, Protocol

import ray
import torch


class BaseImageReward(Protocol):
    name: str
    weight: float

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]: ...


class DummyImageReward:
    """Deterministic reward derived from prompt hash and the per-image mean.

    The deterministic property is important for the smoke test: replaying the
    same prompts on the same generated tensors must give the same reward, so
    we can detect rollout-level non-determinism without entangling it with
    reward noise.
    """

    name: str = "dummy"
    weight: float = 1.0

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if images.shape[0] != len(prompts):
            raise ValueError(
                f"images batch={images.shape[0]} but prompts len={len(prompts)}"
            )
        prompt_scores = torch.tensor(
            [
                # Stable hash → [0, 1) via the first 8 hex chars of sha256.
                int(hashlib.sha256(p.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
                for p in prompts
            ],
            dtype=torch.float32,
        )
        image_means = images.float().mean(dim=tuple(range(1, images.ndim)))
        score = (prompt_scores.to(image_means.device) + image_means.clamp(-1, 1)) / 2.0
        return {"dummy": score}


@ray.remote
class _RewardWorker:  # pragma: no cover
    """Lightweight Ray actor that owns one reward plugin instance."""

    def __init__(self, plugin_factory: Callable[[], BaseImageReward]) -> None:
        self._plugin = plugin_factory()

    def name(self) -> str:
        return self._plugin.name

    def weight(self) -> float:
        return self._plugin.weight

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        return self._plugin.score(images, prompts, metadata)


class JpegCompressibilityReward:
    """Port of verl-omni `jpeg_compressibility` reward.

    Reward = `- jpeg_size_kb / 500`. More compressible (smaller JPEG) → higher
    reward. Used for cross-stack parity comparison against verl-omni since it
    is verl-omni's only zero-config rule-based image reward.
    """

    name: str = "jpeg_compressibility"
    weight: float = 1.0
    quality: int = 95

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        import io as _io

        from PIL import Image

        arr = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        arr = arr.transpose(0, 2, 3, 1)  # NCHW → NHWC
        sizes_kb = []
        for img_np in arr:
            buf = _io.BytesIO()
            Image.fromarray(img_np).save(buf, format="JPEG", quality=self.quality)
            sizes_kb.append(buf.tell() / 1000.0)
        rewards = -torch.tensor(sizes_kb, dtype=torch.float32) / 500.0
        return {"jpeg_compressibility": rewards}


class PickScoreReward:
    """Prompt-image preference score from PickScore-v1 (CLIP-H fine-tune).

    Scores each image against its own prompt with
    `logit_scale * cosine(text_emb, image_emb)` (raw scale ~16-26; the
    GRPO group normalization makes the absolute scale irrelevant).

    The model is ~4 GB; configure the reward pool with
    `num_gpus_per_worker: 1` to run it on GPU. Images arrive as CPU
    NCHW float [0,1] tensors and scores are returned on CPU, matching
    the trainer's device convention.
    """

    name: str = "pickscore"
    weight: float = 1.0

    model_name: str = "yuvalkirstain/PickScore_v1"
    processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    batch_size: int = 16

    def __init__(self) -> None:
        # Deferred heavy imports: the factory is pickled into the Ray actor.
        import torch as _torch
        from transformers import AutoModel, AutoProcessor

        self._device = "cuda" if _torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.processor_name)
        self._model = AutoModel.from_pretrained(self.model_name).eval().to(self._device)

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        from PIL import Image

        if images.shape[0] != len(prompts):
            raise ValueError(
                f"images batch={images.shape[0]} but prompts len={len(prompts)}"
            )
        arr = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        pil_images = [Image.fromarray(a.transpose(1, 2, 0)) for a in arr]

        scores: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(pil_images), self.batch_size):
                chunk_imgs = pil_images[start : start + self.batch_size]
                chunk_prompts = prompts[start : start + self.batch_size]
                image_inputs = self._processor(
                    images=chunk_imgs, return_tensors="pt"
                ).to(self._device)
                text_inputs = self._processor(
                    text=chunk_prompts,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self._device)
                image_embs = self._model.get_image_features(**image_inputs)
                image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
                text_embs = self._model.get_text_features(**text_inputs)
                text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
                chunk_scores = self._model.logit_scale.exp() * (
                    text_embs * image_embs
                ).sum(dim=-1)
                scores.append(chunk_scores.float().cpu())
        return {"pickscore": torch.cat(scores)}


_PLUGIN_REGISTRY: dict[str, Callable[[], BaseImageReward]] = {
    "dummy": lambda: DummyImageReward(),
    "jpeg_compressibility": lambda: JpegCompressibilityReward(),
    "pickscore": lambda: PickScoreReward(),
}


def register_image_reward(name: str, factory: Callable[[], BaseImageReward]) -> None:
    """Register a new reward plugin factory."""
    if name in _PLUGIN_REGISTRY:
        raise ValueError(f"Image reward plugin {name!r} is already registered")
    _PLUGIN_REGISTRY[name] = factory


class ImageRewardEnvironment:
    """A Ray-managed pool of one reward worker per plugin.

    Each plugin contributes a weighted component to the aggregated reward.
    Components are summed; per-component means are emitted as metrics.
    """

    def __init__(
        self,
        plugin_specs: list[dict[str, Any]],
        *,
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: float = 0.0,
    ) -> None:
        self._workers: list[ray.actor.ActorHandle] = []
        self._weights: list[float] = []
        self._names: list[str] = []
        for spec in plugin_specs:
            name = spec["name"]
            if name not in _PLUGIN_REGISTRY:
                raise KeyError(
                    f"Unknown image reward plugin {name!r}; "
                    f"registered={list(_PLUGIN_REGISTRY)}"
                )
            factory = _PLUGIN_REGISTRY[name]
            actor = _RewardWorker.options(
                num_cpus=num_cpus_per_worker, num_gpus=num_gpus_per_worker
            ).remote(factory)
            self._workers.append(actor)
            self._names.append(name)
            self._weights.append(float(spec.get("weight", 1.0)))

    def score_images(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        futures = [w.score.remote(images, prompts, metadata) for w in self._workers]
        per_worker_results: list[dict[str, torch.Tensor]] = ray.get(futures)

        total = torch.zeros(images.shape[0], dtype=torch.float32)
        components: dict[str, torch.Tensor] = {}
        for name, weight, result in zip(self._names, self._weights, per_worker_results):
            for comp_key, comp_value in result.items():
                full_key = f"{name}/{comp_key}"
                components[full_key] = comp_value
                total = total + weight * comp_value
        metrics: dict[str, Any] = {
            f"reward/{k}_mean": float(v.float().mean().item())
            for k, v in components.items()
        }
        metrics["reward/total_mean"] = float(total.mean().item())
        return total, metrics

    def shutdown(self) -> bool:
        for w in self._workers:
            ray.kill(w)
        self._workers = []
        return True
