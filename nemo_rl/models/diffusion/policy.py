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
"""Controller-side ``DiffusionPolicy`` facade.

Owns one :class:`nemo_rl.distributed.worker_groups.RayWorkerGroup` of
:class:`nemo_rl.models.diffusion.workers.diffusion_worker.DiffusionPolicyWorker`
actors. For v1 single-GPU smoke runs the group contains exactly one worker;
multi-GPU sharding (FSDP2 + DP) will be added in a follow-up that introduces
the diffusion-side sharding annotations.

Method names mirror the API the trainer calls: ``sample_trajectory``,
``compute_transition_logprob``, ``train``, ``save_checkpoint``, ``shutdown``.
"""
from __future__ import annotations

from typing import Any

import ray

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.diffusion.interfaces import (
    DiffusionLossConfig,
    DiffusionPolicyConfig,
    DiffusionTrainDataSpec,
    DiffusionTrajectorySpec,
)


class DiffusionPolicy:
    """Controller-side facade around a Ray pool of diffusion workers."""

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: DiffusionPolicyConfig,
        *,
        name_prefix: str = "diffusion_policy",
    ) -> None:
        self.cluster = cluster
        self.config = config
        builder = RayWorkerBuilder(
            "nemo_rl.models.diffusion.workers.diffusion_worker.DiffusionPolicyWorker",
            config,
        )
        self.worker_group = RayWorkerGroup(
            cluster=cluster,
            remote_worker_builder=builder,
            name_prefix=name_prefix,
        )

    # ------------------------------------------------------------------
    # Forward to worker(s) — single-worker smoke path
    # ------------------------------------------------------------------
    def _call_all(self, method_name: str, **kwargs) -> list:
        """Invoke a method on every worker; return list of results (ray.get).

        Bypasses ``run_all_workers_single_data`` because that path requires
        ``sharding_annotations`` which we don't define for the diffusion
        single-worker v1.
        """
        futures = [
            getattr(w, method_name).remote(**kwargs)
            for w in self.worker_group.workers
        ]
        return ray.get(futures)

    def sample_trajectory(
        self,
        prompts: list[str],
        negative_prompts: list[str],
        metadata: list[dict[str, Any]],
        *,
        K: int,
        seed: int | None = None,
    ) -> DiffusionTrajectorySpec:
        results = self._call_all(
            "sample_trajectory",
            prompts=prompts,
            negative_prompts=negative_prompts,
            metadata=metadata,
            K=K,
            seed=seed,
        )
        return results[0]

    def compute_transition_logprob(
        self,
        data: DiffusionTrainDataSpec,
        *,
        use_reference: bool = False,
    ) -> dict[str, Any]:
        return self._call_all(
            "compute_transition_logprob", data=data, use_reference=use_reference
        )[0]

    def train(
        self,
        data: DiffusionTrainDataSpec,
        loss_cfg: DiffusionLossConfig,
    ) -> dict[str, float]:
        per_worker = self._call_all("train_step", data=data, loss_cfg=loss_cfg)
        keys = set().union(*(d.keys() for d in per_worker))
        return {
            k: sum(d.get(k, 0.0) for d in per_worker) / len(per_worker)
            for k in keys
        }

    def prepare_for_generation(self) -> None:
        self._call_all("prepare_for_generation")

    def prepare_for_training(self) -> None:
        self._call_all("prepare_for_training")

    def save_checkpoint(self, path: str) -> None:
        self._call_all("save_checkpoint", path=path)

    def shutdown(self) -> bool:
        try:
            self._call_all("shutdown")
        finally:
            self.worker_group.shutdown()
        return True
