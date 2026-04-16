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

"""Diffusion model support for RL training (e.g., Qwen-Image with GRPO)."""

from typing import NotRequired, TypedDict


class LoRAConfig(TypedDict):
    """LoRA configuration for diffusion model training."""

    enabled: bool
    rank: int
    alpha: int
    target_modules: list[str]
    init_weights: str  # "gaussian" or "default"


class LoRAConfigDisabled(TypedDict):
    enabled: bool


class FSDPConfig(TypedDict):
    """FSDP configuration for distributed diffusion training."""

    enabled: bool
    cpu_offload: bool
    activation_checkpointing: bool
    mixed_precision_dtype: str  # "bfloat16", "float16", "float32"


class TextEncoderConfig(TypedDict):
    """Configuration for the frozen text encoder."""

    freeze: bool


class VAEConfig(TypedDict):
    """Configuration for the frozen VAE."""

    freeze: bool
    dtype: NotRequired[str]


class OptimizerConfig(TypedDict):
    """Optimizer configuration."""

    name: str  # e.g. "torch.optim.AdamW"
    kwargs: dict


class DiffusionGenerationConfig(TypedDict):
    """Diffusion-specific generation parameters."""

    num_inference_steps: int  # Total denoising steps (e.g., 28)
    eval_num_inference_steps: int  # Steps for evaluation (e.g., 50)
    guidance_scale: float  # Classifier-free guidance scale
    height: int  # Image height in pixels
    width: int  # Image width in pixels
    noise_level: float  # SDE stochasticity control (e.g., 0.7)


class DiffusionPolicyConfig(TypedDict):
    """Full configuration for a diffusion policy."""

    model_name: str  # HuggingFace model identifier
    precision: str  # "bfloat16", "float16", "float32"
    train_global_batch_size: int
    train_micro_batch_size: int
    logprob_batch_size: int
    max_grad_norm: float
    lora: LoRAConfig | LoRAConfigDisabled
    fsdp: FSDPConfig
    text_encoder: TextEncoderConfig
    vae: VAEConfig
    optimizer: OptimizerConfig
    generation: DiffusionGenerationConfig
    reference_model: NotRequired[bool]  # Whether to load reference model for KL
