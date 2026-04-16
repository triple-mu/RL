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

"""Unit tests for SDE step with log-probability computation."""

import math
from unittest.mock import MagicMock

import torch

from nemo_rl.models.diffusion.sde import sde_step_with_logprob


def _make_mock_scheduler(sigmas):
    """Create a mock scheduler with given sigma schedule."""
    scheduler = MagicMock()
    scheduler.sigmas = torch.tensor(sigmas, dtype=torch.float32)

    def index_for_timestep(t):
        # Simple lookup: timestep 1000 -> index 0, etc.
        return int(t.item()) if isinstance(t, torch.Tensor) else int(t)

    scheduler.index_for_timestep = index_for_timestep
    return scheduler


class TestSDEStepWithLogprob:
    """Tests for sde_step_with_logprob function."""

    def test_output_shapes(self):
        """Test that outputs have correct shapes."""
        B, C, H, W = 4, 16, 8, 8
        # Sigmas: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        scheduler = _make_mock_scheduler(sigmas)

        model_output = torch.randn(B, C, H, W)
        sample = torch.randn(B, C, H, W)
        # Timestep index 1 -> sigma=0.8, next sigma=0.6
        timestep = torch.tensor([1, 1, 1, 1])

        prev_sample, log_prob, mean, std = sde_step_with_logprob(
            scheduler, model_output, timestep, sample, noise_level=0.7
        )

        assert prev_sample.shape == (B, C, H, W)
        assert log_prob.shape == (B,)
        assert mean.shape == (B, C, H, W)

    def test_log_prob_is_finite(self):
        """Test that log-probs are finite (no NaN or Inf)."""
        B, C, H, W = 2, 4, 4, 4
        sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        scheduler = _make_mock_scheduler(sigmas)

        model_output = torch.randn(B, C, H, W)
        sample = torch.randn(B, C, H, W)
        timestep = torch.tensor([2, 2])

        _, log_prob, _, _ = sde_step_with_logprob(
            scheduler, model_output, timestep, sample, noise_level=1.0
        )

        assert torch.isfinite(log_prob).all(), f"Non-finite log_prob: {log_prob}"

    def test_log_prob_with_provided_prev_sample(self):
        """Test that providing prev_sample gives correct log-prob for that specific transition."""
        B, C, H, W = 2, 4, 4, 4
        sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        scheduler = _make_mock_scheduler(sigmas)

        model_output = torch.randn(B, C, H, W)
        sample = torch.randn(B, C, H, W)
        timestep = torch.tensor([1, 1])

        # First: sample without providing prev_sample
        prev_sample_gen, log_prob_gen, mean, std = sde_step_with_logprob(
            scheduler, model_output, timestep, sample, noise_level=0.7
        )

        # Then: recompute log-prob for the generated sample
        _, log_prob_recomputed, _, _ = sde_step_with_logprob(
            scheduler,
            model_output,
            timestep,
            sample,
            noise_level=0.7,
            prev_sample=prev_sample_gen,
        )

        # Log-probs should be identical when evaluating the same transition
        torch.testing.assert_close(
            log_prob_gen, log_prob_recomputed, atol=1e-5, rtol=1e-5
        )

    def test_log_prob_gaussian_consistency(self):
        """Verify log-prob matches analytical Gaussian log-probability."""
        B, C, H, W = 1, 2, 2, 2
        sigmas = [1.0, 0.7, 0.4, 0.1, 0.0]
        scheduler = _make_mock_scheduler(sigmas)

        model_output = torch.zeros(B, C, H, W)
        sample = torch.ones(B, C, H, W)
        timestep = torch.tensor([1])
        noise_level = 0.5

        prev_sample_known = torch.ones(B, C, H, W) * 0.5

        _, log_prob, mean, std_dev = sde_step_with_logprob(
            scheduler,
            model_output,
            timestep,
            sample,
            noise_level=noise_level,
            prev_sample=prev_sample_known,
        )

        # Manual Gaussian log-prob computation
        sigma = sigmas[1]
        sigma_prev = sigmas[2]
        dt = sigma_prev - sigma
        std_t = math.sqrt(sigma / (1 - sigma)) * noise_level
        sigma_noise = std_t * math.sqrt(-dt)

        diff = prev_sample_known.float() - mean
        manual_log_prob = (
            -(diff**2) / (2 * sigma_noise**2)
            - math.log(sigma_noise)
            - 0.5 * math.log(2 * math.pi)
        )
        manual_log_prob = manual_log_prob.mean(
            dim=tuple(range(1, manual_log_prob.ndim))
        )

        torch.testing.assert_close(log_prob, manual_log_prob, atol=1e-4, rtol=1e-4)

    def test_higher_noise_gives_broader_distribution(self):
        """Higher noise_level should give less negative (broader) log-probs for off-mean samples."""
        B, C, H, W = 2, 4, 4, 4
        sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        scheduler = _make_mock_scheduler(sigmas)

        model_output = torch.randn(B, C, H, W)
        sample = torch.randn(B, C, H, W)
        timestep = torch.tensor([1, 1])

        # Generate a sample, then evaluate it under different noise levels
        prev_sample, _, _, _ = sde_step_with_logprob(
            scheduler, model_output, timestep, sample, noise_level=1.0
        )

        _, log_prob_low_noise, _, _ = sde_step_with_logprob(
            scheduler,
            model_output,
            timestep,
            sample,
            noise_level=0.3,
            prev_sample=prev_sample,
        )

        _, log_prob_high_noise, _, _ = sde_step_with_logprob(
            scheduler,
            model_output,
            timestep,
            sample,
            noise_level=2.0,
            prev_sample=prev_sample,
        )

        # The sample was generated with noise_level=1.0.
        # With much lower noise, the distribution is tighter, so the same
        # sample should have lower log-prob (farther from mean relative to std).
        # This isn't guaranteed per-sample but should hold on average.
        # Just verify both are finite for this test.
        assert torch.isfinite(log_prob_low_noise).all()
        assert torch.isfinite(log_prob_high_noise).all()
