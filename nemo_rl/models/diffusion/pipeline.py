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
collecting latent states and SDE log-probabilities for every timestep
(except the last, whose distribution is too peaked for stable training).

Used for:
  1. Rollout generation (collect trajectories + images for reward)
  2. Log-probability recomputation (evaluate existing trajectories under current policy)
"""

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
    """Runs the full denoising loop with SDE log-probability tracking.

    All timesteps except the last use stochastic SDE noise (noise_level > 0).
    The last step uses deterministic ODE (noise_level=0) because its
    distribution is too peaked and causes numerical issues.

    Args:
        pipeline: A HuggingFace DiffusionPipeline (e.g., QwenImagePipeline).
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
    ) -> DiffusionTrajectorySpec:
        """Run the full denoising loop, collecting trajectory data for GRPO.

        All timesteps except the last collect SDE log-probabilities.

        Args:
            prompts: Text prompts for image generation.
            negative_prompts: Negative prompts for CFG. Defaults to empty strings.
            generator: Random generator for reproducible latent initialization.

        Returns:
            DiffusionTrajectorySpec with trajectory data and decoded images.
        """
        pipeline = self.pipeline
        config = self.config
        batch_size = len(prompts)
        device = self.device
        height = config["height"]
        width = config["width"]

        if negative_prompts is None:
            negative_prompts = [" "] * batch_size

        # Encode prompts
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
            ).expand(latents.shape[0])

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()

        # Train on all timesteps except the last (whose distribution is too peaked)
        num_train_steps = len(timesteps) - 1

        all_latents = [latents]  # initial noise is the first latent
        all_log_probs = []
        all_timesteps = []

        # Denoising loop
        pipeline.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            # SDE noise for all steps except the last
            cur_noise_level = config["noise_level"] if i < num_train_steps else 0

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # CFG forward pass
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

            # Conditional norm scaling
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

            # SDE step
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

            # Collect for all training steps
            if i < num_train_steps:
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

        Args:
            latents: Trajectory latent states [B, T+1, C, H, W].
            timesteps: Timestep values [T].
            prompt_embeds: Text embeddings [B, L, D].
            prompt_embeds_mask: [B, L].
            negative_prompt_embeds: [B, L, D].
            negative_prompt_embeds_mask: [B, L].

        Returns:
            (log_probs [B, T], means [B, T, C, H, W], std_devs [B, T, ...])
        """
        pipeline = self.pipeline
        config = self.config
        batch_size = latents.shape[0]
        num_steps = latents.shape[1] - 1

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

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()
        max_len = max(txt_seq_lens + negative_txt_seq_lens)
        prompt_embeds = prompt_embeds[:, :max_len]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_len]
        negative_prompt_embeds = negative_prompt_embeds[:, :max_len]
        negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :max_len]

        guidance = None
        if pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1],
                config["guidance_scale"],
                device=latents.device,
                dtype=torch.float32,
            ).expand(batch_size)

        all_log_probs = []
        all_means = []
        all_std_devs = []

        for t_idx in range(num_steps):
            x_t = latents[:, t_idx]
            x_next = latents[:, t_idx + 1]
            t = timesteps[t_idx]
            timestep_expanded = t.expand(batch_size).to(x_t.dtype)

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

            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

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
