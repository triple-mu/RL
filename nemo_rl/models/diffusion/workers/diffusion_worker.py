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
"""Diffusion policy worker (Ray actor).

Mirrors the *shape* of
:class:`nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorkerImpl`
but does not inherit from it — the parent's API targets a HF causal-LM
``forward``. Methods exposed to ``DiffusionPolicy``:

``sample_trajectory``, ``compute_transition_logprob``, ``train_step``,
``save_checkpoint``, ``shutdown``, ``report_device_id``.

All ``torch``/``diffusers``/``peft`` imports are deferred into method bodies
because Ray pickles the actor class at submission time, and ``torch`` at
import time drags ``torch._dynamo.config`` (a ``ConfigModuleInstance``) which
is not picklable.

LoRA reference: when enabled, the reference-policy mean is computed by
disabling the PEFT adapter on the *same* transformer instance, avoiding a
duplicate model copy.
"""

from __future__ import annotations

import os
from typing import Any

import ray


def build_no_adapter_forward(transformer: Any, forward_fn: Any) -> Any:
    """Reference-policy forward: run `forward_fn` with the PEFT adapter disabled.

    Returns a closure matching the `forward_override(**kwargs)` convention of
    `QwenImagePipelineAdapter._denoise_step`; `forward_fn` is the worker's
    `_forward_transformer_with_img_shapes`, so the reference forward shares the
    exact input preparation (dtype cast, img_shapes, guidance) of the policy.
    """

    def forward(**kwargs: Any) -> Any:
        with transformer.disable_adapter():
            return forward_fn(transformer, **kwargs)

    return forward


def accumulate_metrics(
    acc: dict[str, float], metrics: dict[str, Any], weight: float
) -> None:
    """Fold one micro-batch's loss metrics into `acc`.

    Extremum metrics take min/max across chunks; everything else is a
    sample-count-weighted mean (weights over all chunks sum to 1).
    """
    import torch

    for k, v in metrics.items():
        v = float(v.item()) if torch.is_tensor(v) else float(v)
        if k == "ratio_min":
            acc[k] = min(acc.get(k, float("inf")), v)
        elif k == "ratio_max":
            acc[k] = max(acc.get(k, float("-inf")), v)
        else:
            acc[k] = acc.get(k, 0.0) + v * weight


@ray.remote
class DiffusionPolicyWorker:  # pragma: no cover
    """Ray actor owning one Qwen-Image pipeline + LoRA optimizer."""

    # Populated by the _load_pipeline/_maybe_apply_lora/_build_optimizer/
    # _build_adapter helpers invoked from __init__ (torch/diffusers/peft types
    # stay `Any` because those imports are deferred into method bodies).
    device: Any
    _pipe: Any
    transformer: Any
    text_encoder: Any
    tokenizer: Any
    vae: Any
    scheduler: Any
    _vae_scale_factor: int
    _num_channels_latents: int
    _lora_enabled: bool
    optimizer: Any
    _img_shapes: list[tuple[int, int, int]]
    adapter: Any
    _loss_fn: Any

    @staticmethod
    def configure_worker(
        num_gpus: int | float | None = None,
        bundle_indices: tuple[int, list[int]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
        """Returns (resources, env_vars, init_kwargs, runtime_env_overrides) for ``RayWorkerGroup``."""
        resources = {"num_gpus": 1, "num_cpus": 4}
        env_vars: dict[str, str] = {}
        init_kwargs: dict[str, Any] = {}
        runtime_env_overrides: dict[str, Any] = {}
        return resources, env_vars, init_kwargs, runtime_env_overrides

    def __init__(
        self,
        # model_dump() dict view of DiffusionPolicyConfig.
        config: dict[str, Any],
        *,
        rank: int = 0,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ) -> None:
        import torch

        from nemo_rl.algorithms.utils import set_seed

        self.config = config
        # RayWorkerGroup injects RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT env
        # vars per worker; the ctor args only serve as fallbacks outside it.
        self.rank = int(os.environ.get("RANK", rank))
        self.world_size = int(os.environ.get("WORLD_SIZE", world_size))

        if config.get("seed") is not None:
            set_seed(int(config["seed"]))
        elif self.world_size > 1:
            # DP correctness relies on every rank materializing bit-identical
            # LoRA init; without a shared seed the ranks silently diverge.
            raise ValueError(
                "policy.seed is required when running data-parallel "
                f"(world_size={self.world_size})"
            )

        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", str(master_port))
        os.environ.setdefault("RANK", str(self.rank))
        os.environ.setdefault("WORLD_SIZE", str(self.world_size))
        os.environ.setdefault("LOCAL_RANK", "0")

        if torch.cuda.is_available() and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", rank=self.rank, world_size=self.world_size
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self._parse_precision(config["precision"])

        self._load_pipeline()
        self._maybe_apply_lora()
        self._build_optimizer()
        self._loss_fn = None
        self._build_adapter()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_precision(precision: str):
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }[precision]

    def _load_pipeline(self) -> None:
        from diffusers import QwenImagePipeline

        model_name = self.config["model_name"]
        pipe = QwenImagePipeline.from_pretrained(model_name, torch_dtype=self.dtype)
        pipe.to(self.device)
        self._pipe = pipe
        self.transformer = pipe.transformer
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self._vae_scale_factor = int(pipe.vae_scale_factor)
        self._num_channels_latents = int(
            getattr(self.transformer.config, "in_channels", 64) // 4
        )

        self._set_scheduler_timesteps(self.config["pipeline"]["num_inference_steps"])

        for module in (self.text_encoder, self.vae):
            for p in module.parameters():
                p.requires_grad_(False)
            module.eval()

        # QwenImageTransformer2DModel gates on `torch.is_grad_enabled() and
        # self.gradient_checkpointing`, so this is a no-op during no_grad
        # rollout and only recomputes activations in the training recompute.
        if self.config["enable_gradient_checkpointing"]:
            self.transformer.enable_gradient_checkpointing()

    def _set_scheduler_timesteps(self, num_inference_steps: int) -> None:
        import numpy as np
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        height = self.config["pipeline"]["height"]
        width = self.config["pipeline"]["width"]
        image_seq_len = (height // self._vae_scale_factor // 2) * (
            width // self._vae_scale_factor // 2
        )
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.set_timesteps(
            num_inference_steps, device=self.device, sigmas=sigmas.tolist(), mu=mu
        )

    def _maybe_apply_lora(self) -> None:
        lora_cfg = self.config["lora_cfg"]
        if not lora_cfg["enabled"]:
            self._lora_enabled = False
            return
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            exclude_modules=lora_cfg["exclude_modules"],
        )
        self.transformer = get_peft_model(self.transformer, peft_config)
        self._lora_enabled = True

    def _build_optimizer(self) -> None:
        import torch

        opt_cfg = self.config["optimizer"]
        trainable = [p for p in self.transformer.parameters() if p.requires_grad]
        assert len(trainable) > 0, "no trainable params; check LoRA targets"
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
        )

    def _build_adapter(self) -> None:
        from nemo_rl.models.diffusion.pipeline import QwenImagePipelineAdapter

        height = self.config["pipeline"]["height"]
        width = self.config["pipeline"]["width"]
        h_pack = height // self._vae_scale_factor // 2
        w_pack = width // self._vae_scale_factor // 2
        self._img_shapes = [(1, h_pack, w_pack)]

        self.adapter = QwenImagePipelineAdapter(
            transformer=self.transformer,
            scheduler=self.scheduler,
            pipeline_cfg=self.config["pipeline"],
            algo_cfg=self.config["algo"],
            encode_condition_fn=self._encode_condition,
            decode_fn=self._decode,
            prepare_initial_latents_fn=self._prepare_initial_latents,
            latent_channels=self._num_channels_latents,
            vae_scale_factor=self._vae_scale_factor,
            device=self.device,
            dtype=self.dtype,
            forward_transformer_fn=self._forward_transformer_with_img_shapes,
            per_element_logprob=bool(self.config["per_element_logprob"]),
        )

    def _forward_transformer_with_img_shapes(self, transformer, **kwargs):
        import torch

        timestep = kwargs.pop("timestep")
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        hs = kwargs["hidden_states"].to(dtype=self.dtype)
        eh = kwargs["encoder_hidden_states"].to(dtype=self.dtype)
        timestep = timestep.to(dtype=hs.dtype).expand(hs.shape[0])
        guidance = None
        if getattr(transformer.config, "guidance_embeds", False):
            gscale = self.config["pipeline"]["guidance_scale"]
            if gscale is None:
                raise ValueError(
                    "transformer.config.guidance_embeds is True but pipeline.guidance_scale is None"
                )
            guidance = torch.full(
                [hs.shape[0]],
                float(gscale),
                device=hs.device,
                dtype=torch.float32,
            )
        img_shapes = [self._img_shapes] * hs.shape[0]
        out = transformer(
            hidden_states=hs,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=eh,
            encoder_hidden_states_mask=kwargs.get("encoder_hidden_states_mask"),
            img_shapes=img_shapes,
            return_dict=False,
        )
        return out[0] if isinstance(out, tuple) else out

    # ------------------------------------------------------------------
    # Pipeline callbacks
    # ------------------------------------------------------------------
    def _encode_condition(self, prompts: list[str], negative_prompts: list[str]):
        import torch

        with torch.no_grad():
            max_len = int(self.config["pipeline"]["max_sequence_length"])
            prompt_embeds, prompt_embeds_mask = self._pipe.encode_prompt(
                prompt=prompts,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=max_len,
            )
            if prompt_embeds_mask is None:
                prompt_embeds_mask = torch.ones(
                    prompt_embeds.shape[:2], device=self.device, dtype=torch.long
                )
            neg_embeds, neg_mask = self._pipe.encode_prompt(
                prompt=negative_prompts,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=max_len,
            )
            if neg_mask is None:
                neg_mask = torch.ones(
                    neg_embeds.shape[:2], device=self.device, dtype=torch.long
                )
            return {
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": neg_embeds,
                "negative_prompt_embeds_mask": neg_mask,
            }

    def _decode(self, latents):
        import torch

        with torch.no_grad():
            height = self.config["pipeline"]["height"]
            width = self.config["pipeline"]["width"]
            unpacked = self._pipe._unpack_latents(
                latents.to(self.dtype), height, width, self._vae_scale_factor
            )
            latents_mean = getattr(self.vae.config, "latents_mean", None)
            latents_std = getattr(self.vae.config, "latents_std", None)
            if latents_mean is not None and latents_std is not None:
                mean = torch.tensor(
                    latents_mean, device=unpacked.device, dtype=unpacked.dtype
                ).view(1, -1, 1, 1, 1)
                std = torch.tensor(
                    latents_std, device=unpacked.device, dtype=unpacked.dtype
                ).view(1, -1, 1, 1, 1)
                unpacked = unpacked * std + mean
            else:
                scaling = getattr(self.vae.config, "scaling_factor", 1.0)
                unpacked = unpacked / scaling
            out = self.vae.decode(unpacked, return_dict=False)[0]
            if out.ndim == 5:
                out = out.squeeze(2)
            return (out / 2 + 0.5).clamp(0, 1)

    def _prepare_initial_latents(self, batch_size: int, seed: int | None):
        import torch

        height = self.config["pipeline"]["height"]
        width = self.config["pipeline"]["width"]
        h = 2 * (int(height) // (self._vae_scale_factor * 2))
        w = 2 * (int(width) // (self._vae_scale_factor * 2))
        shape = (batch_size, 1, self._num_channels_latents, h, w)
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        latents = torch.randn(
            shape, generator=generator, device=self.device, dtype=self.dtype
        )
        return self._pipe._pack_latents(
            latents, batch_size, self._num_channels_latents, h, w
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample_trajectory(
        self,
        prompts: list[str],
        negative_prompts: list[str],
        metadata: list[dict[str, Any]],
        *,
        K: int,
        seed: int | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ):
        import torch

        self.transformer.eval()
        if generation_overrides is None:
            with torch.no_grad():
                return self.adapter.sample_trajectory(
                    prompts, negative_prompts, metadata, K=K, seed=seed
                )
        train_steps = int(self.config["pipeline"]["num_inference_steps"])
        val_steps = int(generation_overrides.get("num_inference_steps", train_steps))
        saved_algo = self.adapter.algo_cfg
        self._set_scheduler_timesteps(val_steps)
        # 纯 ODE：窗口置空 → 每步 stochastic=False，logprob 全 0
        self.adapter.algo_cfg = {
            **self.config["algo"],
            "sde_window_size": 0,
            "sde_window_range": None,
        }
        try:
            with torch.no_grad():
                return self.adapter.sample_trajectory(
                    prompts, negative_prompts, metadata, K=K, seed=seed
                )
        finally:
            self.adapter.algo_cfg = saved_algo
            self._set_scheduler_timesteps(train_steps)

    def compute_transition_logprob(
        self,
        data,
        *,
        use_reference: bool = False,
        train: bool = False,
    ):
        # train=True keeps the module in training mode so dropout (if any)
        # stays active during the with-grad recompute inside train_step.
        self.transformer.train(train)
        reference_forward = None
        if use_reference:
            if not self._lora_enabled:
                raise RuntimeError(
                    "beta > 0 requires LoRA: the reference policy is the base "
                    "model with the adapter disabled; full-parameter training "
                    "has no reference copy"
                )
            reference_forward = self._build_no_adapter_forward()
        curr, means, stds, refs = self.adapter.compute_transition_logprob(
            data,
            use_reference=use_reference,
            reference_forward_fn=reference_forward,
        )
        return {
            "curr_logprob": curr,
            "current_mean": means,
            "std_dev": stds,
            "reference_mean": refs,
        }

    def _build_no_adapter_forward(self):
        return build_no_adapter_forward(
            self.transformer, self._forward_transformer_with_img_shapes
        )

    def train_step(self, data, loss_cfg):
        import torch

        from nemo_rl.algorithms.loss.diffusion_grpo import DiffusionGRPOLossFn

        self.transformer.train()
        if self._loss_fn is None:
            self._loss_fn = DiffusionGRPOLossFn(loss_cfg)
        use_reference = float(loss_cfg["beta"]) > 0

        # Gradient accumulation over sample-dimension chunks so the
        # with-grad T-step recompute only holds `micro` samples' activations
        # at a time. Chunk losses are weighted by sample count, which matches
        # the full-batch masked_mean when masks are uniform across samples.
        total = int(data["generation_logprobs"].shape[0])
        # None → the whole rollout batch in one backward pass.
        micro = self.config["train_micro_batch_size"] or total
        self.optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        metrics_acc: dict[str, float] = {}
        for start in range(0, total, micro):
            end = min(start + micro, total)
            chunk = data.slice(start, end)
            weight = (end - start) / total
            recompute = self.compute_transition_logprob(
                chunk, use_reference=use_reference, train=True
            )
            loss, metrics = self._loss_fn(
                curr_logprob=recompute["curr_logprob"],
                generation_logprob=chunk["generation_logprobs"],
                advantages=chunk["advantages"],
                timestep_mask=chunk["timestep_mask"],
                sample_mask=chunk["sample_mask"],
                current_mean=recompute["current_mean"],
                reference_mean=recompute["reference_mean"],
                std_dev=recompute["std_dev"],
            )
            (loss * weight).backward()
            loss_acc += float(loss.detach().item()) * weight
            accumulate_metrics(metrics_acc, metrics, weight)
        if self.world_size > 1 and torch.distributed.is_initialized():
            # Data-parallel: average trainable grads across ranks so every
            # optimizer step applies the identical global update.
            for p in self.transformer.parameters():
                if p.requires_grad and p.grad is not None:
                    torch.distributed.all_reduce(
                        p.grad, op=torch.distributed.ReduceOp.AVG
                    )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in self.transformer.parameters() if p.requires_grad),
            max_norm=float(self.config["optimizer"]["max_grad_norm"]),
        )
        self.optimizer.step()
        return {
            "loss": loss_acc,
            "grad_norm": float(grad_norm.item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            **metrics_acc,
        }

    def report_trainable_checksum(self) -> float:
        """Sum of all trainable params — DP ranks must agree after each step."""
        import torch

        with torch.no_grad():
            return float(
                sum(
                    p.double().sum().item()
                    for p in self.transformer.parameters()
                    if p.requires_grad
                )
            )

    def save_checkpoint(self, path: str) -> None:
        import torch

        # DP ranks hold identical weights after the all-reduced step; only
        # rank 0 writes to avoid concurrent writes to the same path.
        if self.rank != 0:
            return
        os.makedirs(path, exist_ok=True)
        if self._lora_enabled:
            self.transformer.save_pretrained(path)
        else:
            torch.save(self.transformer.state_dict(), f"{path}/transformer.pt")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")

    def load_checkpoint(self, path: str) -> bool:
        """Restore weights + optimizer saved by :meth:`save_checkpoint`.

        Every DP rank loads the same rank-0-written checkpoint, which matches
        the invariant that ranks hold identical weights and optimizer state.
        """
        import torch

        if self._lora_enabled:
            from peft.utils import set_peft_model_state_dict
            from safetensors.torch import load_file

            adapter_sd = load_file(
                os.path.join(path, "adapter_model.safetensors"),
                device=str(self.device),
            )
            set_peft_model_state_dict(self.transformer, adapter_sd)
        else:
            sd = torch.load(
                os.path.join(path, "transformer.pt"), map_location=self.device
            )
            self.transformer.load_state_dict(sd)
        self.optimizer.load_state_dict(
            torch.load(os.path.join(path, "optimizer.pt"), map_location=self.device)
        )
        return True

    def prepare_for_generation(self) -> None:
        self.transformer.eval()

    def prepare_for_training(self) -> None:
        self.transformer.train()

    def shutdown(self) -> bool:
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return True

    def report_device_id(self) -> str:
        import torch

        if torch.cuda.is_available():
            try:
                return torch.cuda.get_device_name(torch.cuda.current_device())
            except Exception:
                return "cuda"
        return "cpu"
