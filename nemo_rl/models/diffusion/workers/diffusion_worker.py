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

"""FSDP-based worker for diffusion model training and generation.

Each worker holds a shard of the diffusion transformer model (via FSDP),
along with frozen text encoder and VAE components. Workers handle:
  - Model loading and FSDP wrapping
  - LoRA application
  - Trajectory generation (denoising with log-prob collection)
  - Log-probability recomputation for existing trajectories
  - Per-timestep gradient accumulation training
"""

import functools
import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from nemo_rl.models.diffusion import DiffusionPolicyConfig
from nemo_rl.models.diffusion.interfaces import (
    DiffusionTrainDataSpec,
    DiffusionTrajectorySpec,
)
from nemo_rl.models.diffusion.pipeline import DiffusionGenerationPipeline
from nemo_rl.models.diffusion.sde import sde_step_with_logprob


def _ensure_dist_initialized():
    """Initialize torch.distributed for single-GPU if not already initialized."""
    if dist.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)


def _get_qwenimage_transformer_layer_cls():
    """Get the transformer layer classes for FSDP auto-wrapping."""
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformerBlock,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLDecoderLayer,
        Qwen2_5_VLVisionBlock,
    )

    return {QwenImageTransformerBlock, Qwen2_5_VLVisionBlock, Qwen2_5_VLDecoderLayer}


def _apply_fsdp(model, config: DiffusionPolicyConfig, get_layer_cls):
    """Wrap a model with FSDP and optional activation checkpointing."""
    precision_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    mixed_dtype = precision_map.get(config["precision"], torch.bfloat16)

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=get_layer_cls(),
        ),
        mixed_precision=MixedPrecision(
            param_dtype=mixed_dtype,
            reduce_dtype=mixed_dtype,
            buffer_dtype=mixed_dtype,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=config["fsdp"]["cpu_offload"]),
        use_orig_params=True,
    )

    if config["fsdp"]["activation_checkpointing"]:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda m: isinstance(m, tuple(get_layer_cls())),
        )

    return fsdp_model


class DiffusionPolicyWorkerImpl:
    """FSDP-sharded diffusion policy worker for a single GPU."""

    def __init__(self, config: DiffusionPolicyConfig):
        _ensure_dist_initialized()

        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        precision_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.inference_dtype = precision_map.get(config["precision"], torch.bfloat16)

        self._load_model(config)
        self._setup_optimizer(config)
        self._setup_pipeline(config)

    def _load_model(self, config: DiffusionPolicyConfig):
        """Load diffusion pipeline, apply LoRA, wrap with FSDP."""
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            config["model_name"],
            torch_dtype=self.inference_dtype,
        )
        pipeline.safety_checker = None

        # Freeze VAE and text encoder
        pipeline.vae.to(torch.float32)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.to(self.inference_dtype)
        pipeline.text_encoder.requires_grad_(False)

        transformer = pipeline.transformer
        transformer.requires_grad_(False)

        # Apply LoRA if configured
        if config["lora"]["enabled"]:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=config["lora"]["rank"],
                lora_alpha=config["lora"]["alpha"],
                init_lora_weights=config["lora"].get("init_weights", "gaussian"),
                target_modules=config["lora"]["target_modules"],
            )
            transformer = get_peft_model(transformer, lora_config)
            transformer.print_trainable_parameters()

        # Ensure uniform dtype for FSDP (LoRA params may init as float32)
        transformer = transformer.to(self.inference_dtype)

        # Wrap transformer with FSDP
        self.transformer = _apply_fsdp(
            transformer, config, _get_qwenimage_transformer_layer_cls
        )

        # Store frozen components
        self.vae = pipeline.vae.to(self.device)
        self.text_encoder = pipeline.text_encoder.to(self.device)
        self.scheduler = pipeline.scheduler
        self.pipeline_instance = pipeline
        self.pipeline_instance.transformer = self.transformer

        # Reference model for KL regularization (optional)
        self.transformer_ref = None
        if config.get("reference_model", False):
            ref_pipeline = DiffusionPipeline.from_pretrained(
                config["model_name"],
                torch_dtype=self.inference_dtype,
            )
            ref_transformer = ref_pipeline.transformer.to(self.device)
            ref_transformer.requires_grad_(False)
            ref_transformer.eval()
            self.transformer_ref = _apply_fsdp(
                ref_transformer, config, _get_qwenimage_transformer_layer_cls
            )
            del ref_pipeline

    def _setup_optimizer(self, config: DiffusionPolicyConfig):
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.trainable_params = trainable_params

        opt_config = config["optimizer"]
        optimizer_cls = _resolve_class(opt_config["name"])
        self.optimizer = optimizer_cls(trainable_params, **opt_config["kwargs"])

    def _setup_pipeline(self, config: DiffusionPolicyConfig):
        self.gen_pipeline = DiffusionGenerationPipeline(
            self.pipeline_instance, config["generation"]
        )

    def generate(
        self,
        prompts: list[str],
        negative_prompts: Optional[list[str]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> DiffusionTrajectorySpec:
        self.transformer.eval()
        with torch.no_grad():
            return self.gen_pipeline.generate_trajectory(
                prompts=prompts,
                negative_prompts=negative_prompts,
                generator=generator,
                process_index=self.rank,
            )

    def get_logprobs(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.transformer.eval()
        with torch.no_grad():
            return self.gen_pipeline.compute_logprobs_for_trajectory(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            )

    def get_reference_logprobs(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transformer_ref is None:
            raise RuntimeError("Reference model not loaded. Set reference_model=True.")

        orig_transformer = self.pipeline_instance.transformer
        self.pipeline_instance.transformer = self.transformer_ref
        ref_pipeline = DiffusionGenerationPipeline(
            self.pipeline_instance, self.config["generation"]
        )

        self.transformer_ref.eval()
        with torch.no_grad():
            _, means, std_devs = ref_pipeline.compute_logprobs_for_trajectory(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            )

        self.pipeline_instance.transformer = orig_transformer
        return means, std_devs

    def train(
        self,
        data: DiffusionTrainDataSpec,
        loss_fn,
    ) -> dict[str, Any]:
        """Per-timestep gradient accumulation training step."""
        self.transformer.train()
        self.optimizer.zero_grad()

        latents = data["latents"]  # [B, T+1, C, H, W]
        num_timesteps = latents.shape[1] - 1
        config = self.config
        height = config["generation"]["height"]
        width = config["generation"]["width"]
        batch_size = latents.shape[0]

        img_shapes = [
            [
                (
                    1,
                    height // self.pipeline_instance.vae_scale_factor // 2,
                    width // self.pipeline_instance.vae_scale_factor // 2,
                )
            ]
        ] * batch_size

        prompt_embeds = data["prompt_embeds"]
        prompt_embeds_mask = data["prompt_embeds_mask"]
        negative_prompt_embeds = data["negative_prompt_embeds"]
        negative_prompt_embeds_mask = data["negative_prompt_embeds_mask"]

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()
        max_len = max(txt_seq_lens + negative_txt_seq_lens)
        prompt_embeds = prompt_embeds[:, :max_len]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_len]
        negative_prompt_embeds = negative_prompt_embeds[:, :max_len]
        negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :max_len]

        guidance = None
        if self.pipeline_instance.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1],
                config["generation"]["guidance_scale"],
                device=self.device,
                dtype=torch.float32,
            ).expand(batch_size)

        all_metrics = defaultdict(list)

        for t_idx in range(num_timesteps):
            x_t = latents[:, t_idx]
            x_next = latents[:, t_idx + 1]
            t = data["timesteps"][t_idx]
            timestep_expanded = t.expand(batch_size).to(x_t.dtype)

            # Forward pass with CFG
            with torch.amp.autocast("cuda", dtype=self.inference_dtype):
                noise_pred = self.transformer(
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
            comb_pred = neg_noise_pred + config["generation"]["guidance_scale"] * (
                noise_pred - neg_noise_pred
            )

            # Conditional norm scaling
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred_scaled = comb_pred * (cond_norm / noise_norm)

            # Compute log-prob for recorded next latent
            _, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
                self.pipeline_instance.scheduler,
                noise_pred_scaled.float(),
                t.unsqueeze(0).repeat(batch_size),
                x_t.float(),
                prev_sample=x_next.float(),
                noise_level=config["generation"]["noise_level"],
            )

            # Per-timestep loss data
            step_data = {
                "prev_logprobs": data["prev_logprobs"][:, t_idx : t_idx + 1],
                "generation_logprobs": data["generation_logprobs"][
                    :, t_idx : t_idx + 1
                ],
                "advantages": data["advantages"][:, t_idx : t_idx + 1],
                "timestep_mask": data["timestep_mask"][:, t_idx : t_idx + 1],
                "sample_mask": data["sample_mask"],
            }

            if "reference_policy_mean" in data:
                step_data["reference_policy_mean"] = data["reference_policy_mean"][
                    :, t_idx : t_idx + 1
                ]
                step_data["current_policy_mean"] = prev_sample_mean.unsqueeze(1)
                step_data["std_dev"] = std_dev_t.unsqueeze(1)

            global_valid_seqs = data["sample_mask"].sum()
            global_valid_toks = (
                data["timestep_mask"] * data["sample_mask"].unsqueeze(-1)
            ).sum()

            step_loss, step_metrics = loss_fn(
                step_data,
                global_valid_seqs,
                global_valid_toks,
                next_token_logprobs=log_prob.unsqueeze(1),
            )

            scaled_loss = step_loss / num_timesteps
            scaled_loss.backward()

            for k, v in step_metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k].append(v)

        # Gradient clipping and optimizer step
        if config.get("max_grad_norm"):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_params, config["max_grad_norm"]
            )
        else:
            grad_norm = torch.tensor(0.0)

        self.optimizer.step()
        self.optimizer.zero_grad()

        metrics = {}
        for k, v_list in all_metrics.items():
            metrics[k] = sum(v_list) / len(v_list) if v_list else 0.0
        metrics["grad_norm"] = float(grad_norm.item()) if grad_norm.numel() > 0 else 0.0

        return metrics

    def save_checkpoint(self, save_dir: str, step: int):
        from safetensors.torch import save_file

        save_path = os.path.join(save_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)

        with FSDP.state_dict_type(
            self.transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            state_dict = self.transformer.state_dict()
            if self.rank == 0:
                save_file(state_dict, os.path.join(save_path, "model.safetensors"))
            del state_dict

        dist.barrier()

    def shutdown(self) -> bool:
        return True


def _resolve_class(fqn: str):
    """Resolve a fully qualified class name to the class object."""
    module_path, class_name = fqn.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
