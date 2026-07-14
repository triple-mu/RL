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
from pydantic import ValidationError

from nemo_rl.environments.image_reward_environment import (
    _PLUGIN_REGISTRY,
    DummyImageReward,
    ImageRewardEnvConfig,
    ImageRewardEnvironment,
    PickScoreReward,
    register_image_reward,
)


def test_env_config_rejects_zero_workers_per_plugin():
    with pytest.raises(ValidationError):
        ImageRewardEnvConfig(plugins=[{"name": "dummy"}], num_workers_per_plugin=0)


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
        {
            "plugins": [{"name": "dummy", "weight": 0.25}],
            "num_cpus_per_worker": 1,
            "num_gpus_per_worker": 0.0,
        }
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
            {
                "plugins": [
                    {"name": "dummy", "weight": 1.0},
                    {"name": "half", "weight": 2.0},
                ],
                "num_cpus_per_worker": 1,
            }
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


def test_sharded_reward_pool_matches_single_worker(ray_init_and_shutdown):  # noqa: F811
    import torch

    from nemo_rl.environments.image_reward_environment import ImageRewardEnvironment

    images = torch.rand(6, 3, 8, 8)
    prompts = [f"p{i}" for i in range(6)]
    meta = [{} for _ in range(6)]
    env1 = ImageRewardEnvironment(
        {"plugins": [{"name": "dummy"}], "num_cpus_per_worker": 1}
    )
    env2 = ImageRewardEnvironment(
        {
            "plugins": [{"name": "dummy"}],
            "num_cpus_per_worker": 1,
            "num_workers_per_plugin": 4,  # 6 images over 4 replicas: uneven sharding
        }
    )
    # extra="allow" swallows unimplemented fields, so comparing scores alone
    # could pass spuriously; assert the replica count first.
    assert env2._replicas_per_plugin == 4
    r1, _ = env1.score_images(images, prompts, meta)
    r2, _ = env2.score_images(images, prompts, meta)
    assert torch.allclose(r1, r2)
    env1.shutdown()
    env2.shutdown()


def test_pickscore_is_registered():
    assert "pickscore" in _PLUGIN_REGISTRY


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeCLIPProcessor:
    """Mimics the AutoProcessor call signatures used by PickScoreReward."""

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        if images is not None:
            return _FakeBatch(pixel_values=torch.zeros(len(images), 1))
        return _FakeBatch(
            input_ids=torch.tensor([1 if t == "match" else 0 for t in text]),
            attention_mask=torch.ones(len(text)),
        )


class _FakeCLIPModel:
    """Image embeddings are always [1, 0]; text embeddings are [1, 0] for
    the prompt "match" and [0, 1] otherwise, so the paired (diagonal)
    logits_per_image score is 1 or 0."""

    def __call__(self, input_ids=None, attention_mask=None, pixel_values=None):
        image_embs = torch.tensor([[1.0, 0.0]]).expand(pixel_values.shape[0], -1)
        text_embs = torch.zeros(input_ids.shape[0], 2)
        text_embs[input_ids == 1, 0] = 1.0
        text_embs[input_ids == 0, 1] = 1.0

        class _Out:
            logits_per_image = image_embs @ text_embs.T

        return _Out()


def _make_fake_pickscore() -> PickScoreReward:
    plugin = PickScoreReward.__new__(PickScoreReward)
    plugin._device = "cpu"
    plugin._processor = _FakeCLIPProcessor()
    plugin._model = _FakeCLIPModel()
    return plugin


def test_pickscore_pairs_each_image_with_its_own_prompt():
    plugin = _make_fake_pickscore()
    plugin.batch_size = 2  # 3 images → exercises the chunked loop
    images = torch.rand(3, 3, 8, 8)
    scores = plugin.score(images, ["match", "other", "match"], [{}, {}, {}])
    result = scores["pickscore"]
    assert result.shape == (3,)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.allclose(result, torch.tensor([1.0, 0.0, 1.0]))


def test_pickscore_rejects_size_mismatch():
    plugin = _make_fake_pickscore()
    with pytest.raises(ValueError, match="prompts len"):
        plugin.score(torch.rand(2, 3, 8, 8), ["only one"], [{}])


def test_ocr_edit_distance_score_semantics():
    from nemo_rl.environments.image_reward_environment import ocr_edit_distance_score

    assert (
        ocr_edit_distance_score("Hello World", "hello world") == 1.0
    )  # exact after normalization
    assert (
        ocr_edit_distance_score("xx helloworld yy", "Hello World") == 1.0
    )  # substring hit
    assert ocr_edit_distance_score("helxo", "hello") == 1.0 - 1 / 5  # one edit
    assert ocr_edit_distance_score("", "hello") == 0.0  # distance capped at len(gt)
    assert ocr_edit_distance_score("anything", "") == 0.0  # empty gt scores 0


def test_ocr_reward_plugin_with_injected_engine():
    import torch

    from nemo_rl.environments.image_reward_environment import OcrEditDistanceReward

    plugin = OcrEditDistanceReward(ocr_fn=lambda img_np: "hello")
    images = torch.rand(2, 3, 8, 8)
    out = plugin.score(
        images,
        ["p1", "p2"],
        [{"ground_truth": "hello"}, {"ground_truth": "help"}],
    )
    assert out["ocr"].tolist() == [1.0, 1.0 - 2 / 4]  # hello -> help is distance 2


def test_ocr_default_engine_constructed_under_init_lock(monkeypatch, tmp_path):
    """Concurrent replicas race on the ~/.paddleocr model download; the default
    engine must be constructed while holding the exclusive init file lock."""
    import fcntl
    import sys
    import types

    from nemo_rl.environments.image_reward_environment import OcrEditDistanceReward

    monkeypatch.setenv("HOME", str(tmp_path))
    lock_path = tmp_path / ".paddleocr" / ".nemo-rl-init.lock"
    constructed = []

    class _StubPaddleOCR:
        def __init__(self, **kwargs):
            # The lock must already be held here: a second non-blocking
            # exclusive flock on the same path has to fail.
            with open(lock_path) as probe:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            constructed.append(kwargs)

        def ocr(self, img_np, cls=False):
            return [[([[0, 0]], ("stub", 1.0))]]

    stub_module = types.ModuleType("paddleocr")
    stub_module.PaddleOCR = _StubPaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", stub_module)

    plugin = OcrEditDistanceReward()
    assert constructed == [{"use_angle_cls": False, "lang": "en", "show_log": False}]
    # After construction the lock is released and the engine is usable.
    assert plugin._ocr_fn(object()) == "stub"


@pytest.fixture
def ray_init_and_shutdown():
    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            # The sharded-pool test schedules 5 concurrent 1-CPU actors.
            num_cpus=8,
            local_mode=False,
            ignore_reinit_error=True,
        )
    yield
    if ray.is_initialized():
        ray.shutdown()
