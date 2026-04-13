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

"""DiffusionPolicy: Ray-based coordinator for distributed diffusion policy workers.

Manages a group of DiffusionPolicyWorker actors across GPUs, handling data
distribution, parallel execution, and result aggregation for:
  - Trajectory generation (diffusion denoising)
  - Log-probability inference
  - Policy gradient training
  - Checkpointing

Analogous to Policy (lm_policy.py) but specialized for diffusion models
where generation and training happen on the same workers (no separate
inference engine like vLLM/SGLang).
"""

from typing import Any, Optional

import ray
import torch

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.diffusion import DiffusionPolicyConfig
from nemo_rl.models.diffusion.interfaces import (
    DiffusionTrainDataSpec,
    DiffusionTrajectorySpec,
)


class DiffusionPolicy:
    """Distributed diffusion policy coordinator.

    Creates and manages a group of FSDP-sharded diffusion model workers
    across GPUs. Provides high-level methods for generation, inference,
    and training that handle data sharding and result gathering.

    Unlike the LLM Policy class, DiffusionPolicy does NOT use separate
    generation engines. The same workers handle both generation and training,
    since diffusion generation requires the full model (not a separate
    inference-optimized engine like vLLM).

    Args:
        cluster: Ray virtual cluster for resource allocation.
        config: Diffusion policy configuration.
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: DiffusionPolicyConfig,
    ):
        self.config = config
        self.cluster = cluster
        self.workers = []

        # Create one worker per GPU in the cluster
        num_workers = cluster.num_gpus
        for i in range(num_workers):
            worker = ray.remote(
                num_gpus=1,
            ).options(
                scheduling_strategy="SPREAD",
            )(
                _create_worker
            ).remote(config)
            self.workers.append(worker)

        self.num_workers = num_workers

    def diffusion_generate(
        self,
        prompts: list[str],
        negative_prompts: Optional[list[str]] = None,
    ) -> DiffusionTrajectorySpec:
        """Generate denoising trajectories distributed across workers.

        Shards the prompts across workers, runs generation in parallel,
        and gathers the results into a single trajectory batch.

        Args:
            prompts: Text prompts for image generation.
            negative_prompts: Negative prompts for CFG.

        Returns:
            Combined DiffusionTrajectorySpec from all workers.
        """
        # Shard prompts across workers
        prompts_per_worker = _shard_list(prompts, self.num_workers)
        neg_per_worker = (
            _shard_list(negative_prompts, self.num_workers)
            if negative_prompts
            else [None] * self.num_workers
        )

        # Launch generation on all workers in parallel
        futures = []
        for worker, worker_prompts, worker_neg in zip(
            self.workers, prompts_per_worker, neg_per_worker
        ):
            future = worker.generate.remote(
                prompts=worker_prompts,
                negative_prompts=worker_neg,
            )
            futures.append(future)

        # Gather results
        results = ray.get(futures)
        return _merge_trajectory_specs(results)

    def get_logprobs(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Distributed log-probability inference.

        Shards trajectories across workers for parallel logprob recomputation.

        Returns:
            Tuple of (log_probs [B, T], means [B, T, C, H, W], std_devs [B, T, ...]).
        """
        batch_size = latents.shape[0]
        shard_size = batch_size // self.num_workers

        futures = []
        for i, worker in enumerate(self.workers):
            start = i * shard_size
            end = start + shard_size if i < self.num_workers - 1 else batch_size
            future = worker.get_logprobs.remote(
                latents=latents[start:end],
                timesteps=timesteps,
                prompt_embeds=prompt_embeds[start:end],
                prompt_embeds_mask=prompt_embeds_mask[start:end],
                negative_prompt_embeds=negative_prompt_embeds[start:end],
                negative_prompt_embeds_mask=negative_prompt_embeds_mask[start:end],
            )
            futures.append(future)

        results = ray.get(futures)
        log_probs = torch.cat([r[0] for r in results], dim=0)
        means = torch.cat([r[1] for r in results], dim=0)
        std_devs = torch.cat([r[2] for r in results], dim=0)
        return log_probs, means, std_devs

    def get_reference_logprobs(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Distributed reference model mean prediction computation.

        Returns:
            Tuple of (means [B, T, C, H, W], std_devs [B, T, ...]).
        """
        batch_size = latents.shape[0]
        shard_size = batch_size // self.num_workers

        futures = []
        for i, worker in enumerate(self.workers):
            start = i * shard_size
            end = start + shard_size if i < self.num_workers - 1 else batch_size
            future = worker.get_reference_logprobs.remote(
                latents=latents[start:end],
                timesteps=timesteps,
                prompt_embeds=prompt_embeds[start:end],
                prompt_embeds_mask=prompt_embeds_mask[start:end],
                negative_prompt_embeds=negative_prompt_embeds[start:end],
                negative_prompt_embeds_mask=negative_prompt_embeds_mask[start:end],
            )
            futures.append(future)

        results = ray.get(futures)
        means = torch.cat([r[0] for r in results], dim=0)
        std_devs = torch.cat([r[1] for r in results], dim=0)
        return means, std_devs

    def train(
        self,
        data: DiffusionTrainDataSpec,
        loss_fn,
    ) -> dict[str, Any]:
        """Distributed training step.

        Shards the training data across workers, runs per-timestep gradient
        accumulation in parallel, and aggregates metrics.

        Args:
            data: Training data with trajectories, logprobs, and advantages.
            loss_fn: Loss function instance.

        Returns:
            Aggregated training metrics dictionary.
        """
        batch_size = data["latents"].shape[0]
        shard_size = batch_size // self.num_workers

        futures = []
        for i, worker in enumerate(self.workers):
            start = i * shard_size
            end = start + shard_size if i < self.num_workers - 1 else batch_size
            shard_data = _shard_train_data(data, start, end)
            future = worker.train.remote(data=shard_data, loss_fn=loss_fn)
            futures.append(future)

        results = ray.get(futures)
        return _aggregate_metrics(results)

    def prepare_for_training(self):
        """Prepare all workers for training mode."""
        futures = [w.prepare_for_training.remote() for w in self.workers]
        ray.get(futures)

    def finish_training(self):
        """Switch all workers to eval mode."""
        futures = [w.finish_training.remote() for w in self.workers]
        ray.get(futures)

    def save_checkpoint(self, save_dir: str, step: int):
        """Save model checkpoint across all workers."""
        futures = [
            w.save_checkpoint.remote(save_dir=save_dir, step=step)
            for w in self.workers
        ]
        ray.get(futures)

    def shutdown(self) -> bool:
        """Shutdown all workers."""
        futures = [w.shutdown.remote() for w in self.workers]
        ray.get(futures)
        return True


def _create_worker(config: DiffusionPolicyConfig):
    """Factory function for creating DiffusionPolicyWorkerImpl instances."""
    from nemo_rl.models.diffusion.workers.diffusion_worker import (
        DiffusionPolicyWorkerImpl,
    )

    return DiffusionPolicyWorkerImpl(config)


def _shard_list(items: list, num_shards: int) -> list[list]:
    """Split a list into approximately equal shards."""
    shard_size = len(items) // num_shards
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else len(items)
        shards.append(items[start:end])
    return shards


def _shard_train_data(
    data: DiffusionTrainDataSpec, start: int, end: int
) -> DiffusionTrainDataSpec:
    """Extract a batch slice from training data."""
    sliced = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if value.shape[0] > 1:  # Batch dimension exists
                sliced[key] = value[start:end]
            else:
                sliced[key] = value  # Shared tensor (e.g., timesteps)
        else:
            sliced[key] = value
    return sliced


def _merge_trajectory_specs(
    specs: list[DiffusionTrajectorySpec],
) -> DiffusionTrajectorySpec:
    """Merge trajectory specs from multiple workers into a single batch."""
    merged = {}
    for key in specs[0]:
        values = [s[key] for s in specs]
        if isinstance(values[0], torch.Tensor):
            if values[0].ndim > 0 and values[0].shape[0] > 0:
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = values[0]  # Shared tensor
        elif isinstance(values[0], list):
            merged[key] = sum(values, [])
        else:
            merged[key] = values[0]
    return merged


def _aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Average metrics across workers."""
    if not metrics_list:
        return {}
    aggregated = {}
    for key in metrics_list[0]:
        values = [m[key] for m in metrics_list if key in m]
        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = values[0]
    return aggregated
