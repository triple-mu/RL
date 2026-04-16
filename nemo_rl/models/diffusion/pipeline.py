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

"""Diffusion generation pipeline with log-probability collection.

Wraps a HuggingFace DiffusionPipeline to run the full T-step denoising loop,
collecting latent states and SDE log-probabilities within a configurable
training window. Used for both:
  1. Rollout generation (collect trajectories + images for reward)
  2. Log-probability recomputation (evaluate existing trajectories under current policy)
"""

import random
from typing import Optional

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from nemo_rl.models.diffusion import DiffusionGenerationConfig
from nemo_rl.models.diffusion.interfaces import DiffusionTrajectorySpec
from nemo_rl.models.diffusion.sde import sde_step_with_logprob


class DiffusionGenerationPipeline:
    """Manages diffusion trajectory generation with SDE log-probability tracking.

    This pipeline runs the denoising loop of a flow-matching diffusion model,
    optionally injecting SDE noise within a configurable window of timesteps
    to enable log-probability computation for policy gradient training.

    The SDE window mechanism allows training on only a subset of denoising steps,
    reducing memory and compute while still providing meaningful gradient signal.
    Steps outside the window use deterministic ODE integration (noise_level=0).

    Args:
        pipeline: A HuggingFace DiffusionPipeline instance (e.g., QwenImagePipeline).
            Must have: transformer, vae, scheduler, text_encoder, image_processor,
            encode_prompt(), prepare_latents(), _unpack_latents(), check_inputs().
        config: Diffusion generation configuration.
    """

    def __init__(self, pipeline, config: DiffusionGenerationConfig):
        self.pipeline = pipeline
        self.config = config

    @property
    def device(self):
        return self.pipeline.device

    @property
    def transformer(self):
        return self.pipeline.transformer

    @torch.no_grad()
    def generate_trajectory(
        self,
        prompts: list[str],
        negative_prompts: Optional[list[str]] = None,
        generator: Optional[torch.Generator] = None,
        process_index: int = 0,
    ) -> DiffusionTrajectorySpec:
        """Run the full denoising loop, collecting trajectory data for GRPO training.

        For each timestep, runs the transformer with classifier-free guidance,
        then performs an SDE step. Within the SDE window, stochastic noise is
        injected and log-probabilities are recorded. Outside the window,
        deterministic ODE steps are used.

        Args:
            prompts: Text prompts for image generation.
            negative_prompts: Negative prompts for CFG. Defaults to empty strings.
            generator: Random generator for reproducible latent initialization.
            process_index: Used to seed the SDE window random selection per worker.

        Returns:
            DiffusionTrajectorySpec containing the denoising trajectory,
            log-probabilities, decoded images, and text embeddings.
        """
        pipeline = self.pipeline
        config = self.config
        batch_size = len(prompts)
        device = self.device
        height = config["height"]
        width = config["width"]

        if negative_prompts is None:
            negative_prompts = [" "] * batch_size

        # Encode prompts (positive + negative concatenated, then split)
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompt=prompts + negative_prompts,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        prompt_embeds, negative_prompt_embeds = prompt_embeds.chunk(2, dim=0)
        prompt_embeds_mask, negative_prompt_embeds_mask = prompt_embeds_mask.chunk(
            2, dim=0
        )

        # Prepare initial noise latents
        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents = pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        img_shapes = [
            [
                (
                    1,
                    height // pipeline.vae_scale_factor // 2,
                    width // pipeline.vae_scale_factor // 2,
                )
            ]
        ] * batch_size

        # Prepare timesteps with flow matching schedule
        num_steps = config["num_inference_steps"]
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        image_seq_len = latents.shape[1]

        # Calculate timestep shift for flow matching
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

        mu = calculate_shift(
            image_seq_len,
            pipeline.scheduler.config.get("base_image_seq_len", 256),
            pipeline.scheduler.config.get("max_image_seq_len", 4096),
            pipeline.scheduler.config.get("base_shift", 0.5),
            pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_steps = retrieve_timesteps(
            pipeline.scheduler, num_steps, device, sigmas=sigmas, mu=mu
        )

        # Handle guidance
        guidance = None
        if pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1],
                config["guidance_scale"],
                device=device,
                dtype=torch.float32,
            )
            guidance = guidance.expand(latents.shape[0])

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()

        # Determine SDE window (random subset of timesteps for training)
        random.seed(process_index)
        sde_window_size = config["sde_window_size"]
        if sde_window_size > 0:
            start = random.randint(
                config["sde_window_range_start"],
                config["sde_window_range_end"] - sde_window_size,
            )
            end = start + sde_window_size
            sde_window = (start, end)
        else:
            # Full trajectory except last step (near-image step has very peaked distribution)
            sde_window = (0, len(timesteps) - 1)

        all_latents = []
        all_log_probs = []
        all_timesteps = []

        # Denoising loop
        pipeline.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            # Determine noise level: stochastic within SDE window, deterministic outside
            if i < sde_window[0]:
                cur_noise_level = 0
            elif i == sde_window[0]:
                cur_noise_level = config["noise_level"]
                all_latents.append(latents)
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = config["noise_level"]
            else:
                cur_noise_level = 0

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # Classifier-free guidance: forward pass with both positive and negative prompts
            noise_pred = pipeline.transformer(
                hidden_states=torch.cat([latents, latents], dim=0),
                timestep=torch.cat([timestep, timestep], dim=0) / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=torch.cat(
                    [prompt_embeds_mask, negative_prompt_embeds_mask], dim=0
                ),
                encoder_hidden_states=torch.cat(
                    [prompt_embeds, negative_prompt_embeds], dim=0
                ),
                img_shapes=img_shapes * 2,
                txt_seq_lens=txt_seq_lens + negative_txt_seq_lens,
            )[0]

            noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)
            comb_pred = neg_noise_pred + config["guidance_scale"] * (
                noise_pred - neg_noise_pred
            )

            # Conditional norm scaling (preserves prediction magnitude)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

            # SDE step with log-probability
            latents_dtype = latents.dtype
            latents, log_prob, _, _ = sde_step_with_logprob(
                pipeline.scheduler,
                noise_pred.float(),
                t.unsqueeze(0).repeat(latents.shape[0]),
                latents.float(),
                noise_level=cur_noise_level,
            )
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            # Collect data within SDE window
            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(t)

        # Decode final latents to images via VAE
        decoded_latents = pipeline._unpack_latents(
            latents, height, width, pipeline.vae_scale_factor
        )
        decoded_latents = decoded_latents.to(pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(decoded_latents.device, decoded_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(
            1, pipeline.vae.config.z_dim, 1, 1, 1
        ).to(decoded_latents.device, decoded_latents.dtype)
        decoded_latents = decoded_latents / latents_std + latents_mean
        images = pipeline.vae.decode(decoded_latents, return_dict=False)[0][:, :, 0]
        images = pipeline.image_processor.postprocess(images, output_type="pt")

        return DiffusionTrajectorySpec(
            latents=torch.stack(all_latents, dim=1),
            log_probs=torch.stack(all_log_probs, dim=1),
            timesteps=torch.stack(all_timesteps),
            images=images,
            prompt_text=prompts,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        )

    def compute_logprobs_for_trajectory(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log-probabilities for an existing trajectory under the current policy.

        Used during the logprob inference phase of GRPO training to evaluate
        how likely the previously generated trajectory is under the (possibly updated)
        policy parameters.

        Args:
            latents: Trajectory latent states. Shape [B, T_window+1, C, H, W].
                Index [:, t] is x_t, [:, t+1] is x_{t+1}.
            timesteps: Timestep values for each SDE step. Shape [T_window].
            prompt_embeds: Text embeddings. Shape [B, L, D].
            prompt_embeds_mask: Mask for text embeddings. Shape [B, L].
            negative_prompt_embeds: Negative text embeddings. Shape [B, L, D].
            negative_prompt_embeds_mask: Mask for negative embeddings. Shape [B, L].

        Returns:
            log_probs: Log-probabilities per timestep. Shape [B, T_window].
            means: Mean predictions per timestep. Shape [B, T_window, C, H, W].
            std_devs: SDE std deviations per timestep. Shape [B, T_window, 1, 1, 1] (broadcastable).
        """
        pipeline = self.pipeline
        config = self.config
        batch_size = latents.shape[0]
        num_window_steps = (
            latents.shape[1] - 1
        )  # T_window transitions from T_window+1 states

        height = config["height"]
        width = config["width"]
        img_shapes = [
            [
                (
                    1,
                    height // pipeline.vae_scale_factor // 2,
                    width // pipeline.vae_scale_factor // 2,
                )
            ]
        ] * batch_size

        # Trim embeddings to max actual length for efficiency
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()
        max_len = max(txt_seq_lens + negative_txt_seq_lens)
        prompt_embeds = prompt_embeds[:, :max_len]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_len]
        negative_prompt_embeds = negative_prompt_embeds[:, :max_len]
        negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :max_len]

        # Handle guidance
        guidance = None
        if pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1],
                config["guidance_scale"],
                device=latents.device,
                dtype=torch.float32,
            )
            guidance = guidance.expand(batch_size)

        all_log_probs = []
        all_means = []
        all_std_devs = []

        for t_idx in range(num_window_steps):
            x_t = latents[:, t_idx]
            x_next = latents[:, t_idx + 1]
            t = timesteps[t_idx]
            timestep_expanded = t.expand(batch_size).to(x_t.dtype)

            # Classifier-free guidance forward pass
            noise_pred = pipeline.transformer(
                hidden_states=torch.cat([x_t, x_t], dim=0),
                timestep=torch.cat([timestep_expanded, timestep_expanded], dim=0)
                / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=torch.cat(
                    [prompt_embeds_mask, negative_prompt_embeds_mask], dim=0
                ),
                encoder_hidden_states=torch.cat(
                    [prompt_embeds, negative_prompt_embeds], dim=0
                ),
                img_shapes=img_shapes * 2,
                txt_seq_lens=txt_seq_lens + negative_txt_seq_lens,
            )[0]

            noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)
            comb_pred = neg_noise_pred + config["guidance_scale"] * (
                noise_pred - neg_noise_pred
            )

            # Conditional norm scaling
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

            # Compute log-prob for the recorded next latent
            _, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
                pipeline.scheduler,
                noise_pred.float(),
                t.unsqueeze(0).repeat(batch_size),
                x_t.float(),
                prev_sample=x_next.float(),
                noise_level=config["noise_level"],
            )

            all_log_probs.append(log_prob)
            all_means.append(prev_sample_mean)
            all_std_devs.append(std_dev_t)

        return (
            torch.stack(all_log_probs, dim=1),
            torch.stack(all_means, dim=1),
            torch.stack(all_std_devs, dim=1),
        )
