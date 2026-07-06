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
import pytest
import ray
import torch

from nemo_rl.environments.image_reward_environment import (
    DummyImageReward,
    ImageRewardEnvironment,
    register_image_reward,
)


def test_dummy_image_reward_is_deterministic():
    a = DummyImageReward()
    b = DummyImageReward()
    images = torch.zeros(2, 3, 4, 4)
    prompts = ["cat", "dog"]
    sa = a.score(images, prompts, [{}, {}])
    sb = b.score(images, prompts, [{}, {}])
    assert torch.equal(sa["dummy"], sb["dummy"])


def test_dummy_image_reward_rejects_size_mismatch():
    images = torch.zeros(2, 3, 4, 4)
    with pytest.raises(ValueError, match="prompts len"):
        DummyImageReward().score(images, ["only one"], [{}])


# pragma: no cover # multi-actor Ray test
def test_image_reward_environment_aggregates_weighted_components(ray_init_and_shutdown):  # noqa: F811
    env = ImageRewardEnvironment(
        plugin_specs=[
            {"name": "dummy", "weight": 0.25},
        ],
        num_cpus_per_worker=1,
        num_gpus_per_worker=0.0,
    )
    images = torch.zeros(3, 3, 4, 4)
    prompts = ["a", "b", "c"]
    total, metrics = env.score_images(images, prompts, [{}, {}, {}])
    direct = DummyImageReward().score(images, prompts, [{}, {}, {}])["dummy"]
    assert torch.allclose(total, 0.25 * direct)
    assert "reward/dummy/dummy_mean" in metrics
    assert "reward/total_mean" in metrics
    env.shutdown()


def test_image_reward_environment_two_plugins_sum(ray_init_and_shutdown):  # noqa: F811
    def _half():
        class _Half:
            name = "half"
            weight = 1.0

            def score(self, images, prompts, metadata):
                return {"const": torch.full((images.shape[0],), 0.5)}

        return _Half()

    register_image_reward("half", _half)
    try:
        env = ImageRewardEnvironment(
            plugin_specs=[
                {"name": "dummy", "weight": 1.0},
                {"name": "half", "weight": 2.0},
            ]
        )
        total, metrics = env.score_images(
            torch.zeros(2, 3, 2, 2),
            ["a", "b"],
            [{}, {}],
        )
        # Each "half/const" contributes 2.0 * 0.5 = 1.0 per sample
        # plus the dummy component.
        direct_dummy = DummyImageReward().score(
            torch.zeros(2, 3, 2, 2), ["a", "b"], [{}, {}]
        )["dummy"]
        assert torch.allclose(total, direct_dummy + 1.0)
        assert "reward/half/const_mean" in metrics
        env.shutdown()
    finally:
        # Reset module-global registry to avoid test pollution.
        from nemo_rl.environments.image_reward_environment import _PLUGIN_REGISTRY

        _PLUGIN_REGISTRY.pop("half", None)


@pytest.fixture
def ray_init_and_shutdown():
    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            num_cpus=2,
            local_mode=False,
            ignore_reinit_error=True,
        )
    yield
    if ray.is_initialized():
        ray.shutdown()
