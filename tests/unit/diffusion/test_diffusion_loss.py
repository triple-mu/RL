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

"""Unit tests for DiffusionClippedPGLossFn."""

import torch

from nemo_rl.algorithms.loss.loss_functions import (
    DiffusionClippedPGLossConfig,
    DiffusionClippedPGLossFn,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _make_loss_fn(
    ratio_clip_min=0.2,
    ratio_clip_max=0.2,
    adv_clip_max=5.0,
    kl_penalty=0.0,
):
    config = DiffusionClippedPGLossConfig(
        ratio_clip_min=ratio_clip_min,
        ratio_clip_max=ratio_clip_max,
        adv_clip_max=adv_clip_max,
        kl_penalty=kl_penalty,
    )
    return DiffusionClippedPGLossFn(config)


def _make_data(B=4, T=2, **overrides):
    """Create minimal training data for loss computation."""
    data = BatchedDataDict(
        {
            "prev_logprobs": torch.zeros(B, T),
            "generation_logprobs": torch.zeros(B, T),
            "advantages": torch.ones(B, T),
            "timestep_mask": torch.ones(B, T),
            "sample_mask": torch.ones(B),
        }
    )
    data.update(overrides)
    return data


class TestDiffusionClippedPGLossFn:
    """Tests for the diffusion clipped PG loss function."""

    def test_basic_loss_computation(self):
        """Test that loss is computed without errors."""
        loss_fn = _make_loss_fn()
        data = _make_data()
        curr_logprobs = torch.zeros(4, 2)

        global_valid_seqs = torch.tensor(4.0)
        global_valid_toks = torch.tensor(8.0)

        loss, metrics = loss_fn(
            data,
            global_valid_seqs,
            global_valid_toks,
            next_token_logprobs=curr_logprobs,
        )

        assert loss.ndim == 0, "Loss should be scalar"
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert "policy_loss" in metrics
        assert "approx_kl" in metrics
        assert "clip_fraction" in metrics

    def test_zero_advantage_gives_zero_loss(self):
        """With zero advantages, the policy gradient loss should be zero."""
        loss_fn = _make_loss_fn()
        data = _make_data(advantages=torch.zeros(4, 2))
        curr_logprobs = torch.randn(4, 2) * 0.1

        loss, metrics = loss_fn(
            data,
            torch.tensor(4.0),
            torch.tensor(8.0),
            next_token_logprobs=curr_logprobs,
        )

        assert abs(metrics["policy_loss"]) < 1e-6

    def test_on_policy_ratio_is_one(self):
        """When curr == prev logprobs, ratio should be 1 and no clipping occurs."""
        loss_fn = _make_loss_fn()
        logprobs = torch.randn(4, 2)
        data = _make_data(prev_logprobs=logprobs)

        loss, metrics = loss_fn(
            data,
            torch.tensor(4.0),
            torch.tensor(8.0),
            next_token_logprobs=logprobs,
        )

        assert abs(metrics["mean_ratio"] - 1.0) < 1e-4
        assert metrics["clip_fraction"] < 1e-4

    def test_clipping_behavior(self):
        """Large logprob differences should trigger clipping."""
        loss_fn = _make_loss_fn(ratio_clip_min=0.1, ratio_clip_max=0.1)
        data = _make_data(
            prev_logprobs=torch.zeros(4, 2),
            advantages=torch.ones(4, 2),
        )
        # Large positive log-ratio -> ratio >> 1, should be clipped
        curr_logprobs = torch.ones(4, 2) * 5.0

        loss, metrics = loss_fn(
            data,
            torch.tensor(4.0),
            torch.tensor(8.0),
            next_token_logprobs=curr_logprobs,
        )

        assert metrics["clip_fraction"] > 0.5, "Should have significant clipping"

    def test_masking(self):
        """Masked timesteps and samples should not contribute to loss."""
        loss_fn = _make_loss_fn()
        data = _make_data(
            B=4,
            T=3,
            advantages=torch.ones(4, 3) * 10.0,
            timestep_mask=torch.tensor(
                [
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                ],
                dtype=torch.float,
            ),
            sample_mask=torch.tensor([1, 1, 0, 0], dtype=torch.float),
        )
        curr_logprobs = torch.zeros(4, 3)

        loss, metrics = loss_fn(
            data,
            torch.tensor(2.0),
            torch.tensor(2.0),
            next_token_logprobs=curr_logprobs,
        )

        # Only 2 samples x 1 timestep = 2 valid entries
        assert metrics["num_valid_samples"] == 2

    def test_advantage_clipping(self):
        """Advantages exceeding adv_clip_max should be clamped."""
        loss_fn = _make_loss_fn(adv_clip_max=2.0)
        data = _make_data(advantages=torch.ones(4, 2) * 100.0)
        curr_logprobs = torch.zeros(4, 2)

        loss_big_adv, _ = loss_fn(
            data,
            torch.tensor(4.0),
            torch.tensor(8.0),
            next_token_logprobs=curr_logprobs,
        )

        # Compare with directly using clipped advantages
        data_clipped = _make_data(advantages=torch.ones(4, 2) * 2.0)
        loss_clipped, _ = loss_fn(
            data_clipped,
            torch.tensor(4.0),
            torch.tensor(8.0),
            next_token_logprobs=curr_logprobs,
        )

        torch.testing.assert_close(loss_big_adv, loss_clipped, atol=1e-5, rtol=1e-5)

    def test_kl_penalty(self):
        """KL penalty should increase loss when current differs from reference."""
        loss_fn_no_kl = _make_loss_fn(kl_penalty=0.0)
        loss_fn_with_kl = _make_loss_fn(kl_penalty=1.0)

        B, T, C, H, W = 2, 2, 4, 4, 4
        data = _make_data(B=B, T=T)
        data["reference_policy_mean"] = torch.zeros(B, T, C, H, W)
        data["current_policy_mean"] = torch.ones(B, T, C, H, W)  # Different from ref
        data["std_dev"] = torch.ones(B, T, C, H, W)
        curr_logprobs = torch.zeros(B, T)

        loss_no_kl, metrics_no_kl = loss_fn_no_kl(
            data,
            torch.tensor(float(B)),
            torch.tensor(float(B * T)),
            next_token_logprobs=curr_logprobs,
        )
        loss_with_kl, metrics_with_kl = loss_fn_with_kl(
            data,
            torch.tensor(float(B)),
            torch.tensor(float(B * T)),
            next_token_logprobs=curr_logprobs,
        )

        assert loss_with_kl > loss_no_kl, "KL penalty should increase loss"
        assert metrics_with_kl["kl_loss"] > 0

    def test_loss_type_and_input_type(self):
        """Verify loss_type and input_type are correctly set."""
        from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType

        loss_fn = _make_loss_fn()
        assert loss_fn.loss_type == LossType.SEQUENCE_LEVEL
        assert loss_fn.input_type == LossInputType.LOGPROB
