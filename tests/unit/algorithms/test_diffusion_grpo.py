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
"""Tests for diffusion-GRPO resume helpers."""

import os

from nemo_rl.algorithms.diffusion_grpo import _latest_checkpoint


def _make_ckpt(root, name: str, complete: bool = True) -> str:
    path = os.path.join(root, name)
    os.makedirs(path)
    if complete:
        with open(os.path.join(path, "optimizer.pt"), "wb"):
            pass
    return path


def test_latest_checkpoint_missing_dir_returns_none(tmp_path):
    assert _latest_checkpoint(str(tmp_path / "does_not_exist")) is None


def test_latest_checkpoint_picks_highest_complete_step(tmp_path):
    _make_ckpt(tmp_path, "step_1")
    expected = _make_ckpt(tmp_path, "step_10")
    # Incomplete checkpoint (no optimizer.pt yet) must be skipped even
    # though its step number is the highest.
    _make_ckpt(tmp_path, "step_30", complete=False)
    _make_ckpt(tmp_path, "not_a_checkpoint")
    assert _latest_checkpoint(str(tmp_path)) == (expected, 10)


def test_latest_checkpoint_all_incomplete_returns_none(tmp_path):
    _make_ckpt(tmp_path, "step_5", complete=False)
    assert _latest_checkpoint(str(tmp_path)) is None


def test_master_config_rejects_kl_with_full_param():
    import pytest
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.diffusion_grpo import DiffusionMasterConfig
    from nemo_rl.utils.config import load_config

    cfg = OmegaConf.to_container(
        load_config("examples/configs/diffusion_grpo_qwen_image_tiny.yaml"),
        resolve=True,
    )
    cfg["loss_fn"]["beta"] = 0.04
    cfg["policy"]["lora_cfg"]["enabled"] = False
    with pytest.raises(Exception, match="beta"):
        DiffusionMasterConfig.model_validate(cfg)


def test_run_validation_passes_generation_overrides():
    import torch

    from nemo_rl.algorithms.diffusion_grpo import _run_validation

    seen = []

    class FakePolicy:
        def sample_trajectory(
            self,
            prompts,
            negative_prompts,
            metadata,
            *,
            K,
            seed,
            generation_overrides=None,
        ):
            seen.append(generation_overrides)
            B = len(prompts) * K
            return {
                "prompts": prompts * K,
                "negative_prompts": negative_prompts * K,
                "metadata": metadata * K,
                "images": torch.zeros(B, 3, 4, 4),
                "latents": torch.zeros(B, 2, 4),
                "timesteps": torch.zeros(B, 1),
                "generation_logprobs": torch.zeros(B, 1),
                "timestep_mask": torch.zeros(B, 1),
                "prompt_embeds": torch.zeros(B, 1, 1),
                "prompt_embeds_mask": torch.ones(B, 1),
                "negative_prompt_embeds": torch.zeros(B, 1, 1),
                "negative_prompt_embeds_mask": torch.ones(B, 1),
            }

    class FakeEnv:
        def score_images(self, images, prompts, metadata):
            return torch.zeros(images.shape[0]), {}

    class FakeLogger:
        def log_metrics(self, *a, **k):
            pass

    _run_validation(
        FakePolicy(),
        FakeEnv(),
        [{"prompts": ["p"], "negative_prompts": [" "], "metadata": [{}]}],
        step=0,
        logger=FakeLogger(),
        seed=42,
        generation_overrides={"num_inference_steps": 40},
    )
    assert seen == [{"num_inference_steps": 40}]


def test_build_train_data_slices_to_window_columns():
    import torch

    from nemo_rl.algorithms.diffusion_grpo import _build_train_data

    B, T, w = 2, 8, 3
    mask = torch.zeros(B, T)
    mask[0, 1 : 1 + w] = 1.0  # sample 0 window [1, 4)
    mask[1, 4 : 4 + w] = 1.0  # sample 1 window [4, 7)
    traj = {
        "latents": torch.arange(B * (T + 1) * 4, dtype=torch.float32).reshape(
            B, T + 1, 4
        ),
        "timesteps": torch.arange(T, dtype=torch.float32).repeat(B, 1),
        "generation_logprobs": torch.randn(B, T) * mask,
        "timestep_mask": mask,
        "prompt_embeds": torch.zeros(B, 4, 8),
        "prompt_embeds_mask": torch.ones(B, 4),
        "negative_prompt_embeds": torch.zeros(B, 4, 8),
        "negative_prompt_embeds_mask": torch.ones(B, 4),
        "prompts": ["a", "b"],
        "negative_prompts": [" ", " "],
        "metadata": [{}, {}],
        "images": torch.zeros(B, 3, 4, 4),
    }
    out = _build_train_data(
        traj, torch.tensor([1.0, -1.0]), loss_multiplier=torch.ones(B)
    )
    assert out["timesteps"].shape == (B, w)
    assert out["latents"].shape == (B, w + 1, 4)
    assert torch.all(out["timestep_mask"] == 1)
    # Sample 1's window starts at 4, so the sliced timesteps must be [4, 5, 6].
    assert out["timesteps"][1].tolist() == [4.0, 5.0, 6.0]
    # Latents keep one extra column (w + 1): sample 1 must equal the original latents[1, 4:8].
    assert torch.equal(out["latents"][1], traj["latents"][1, 4:8])


def test_global_std_tames_constant_group_amplification():
    import torch

    from nemo_rl.algorithms.diffusion_grpo import _compute_advantages

    # 组 a 全常数（OCR 常见），组 b 有微小信号
    prompts = ["a"] * 4 + ["b"] * 4
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.02, 0.0, 0.0])
    adv_group = _compute_advantages(prompts, rewards, use_global_std=False)
    adv_global = _compute_advantages(prompts, rewards, use_global_std=True)
    # 组内 std 归一会把组 b 的微小差异放大到远超全局归一的幅度
    assert adv_group.abs().max() > 10 * adv_global.abs().max()
    # 全局归一下常数组 advantage 为 0，且整体幅度有界、无爆炸
    assert torch.all(adv_global[:4].abs() < 1e-3)
    assert adv_global.abs().max() < 5.0
