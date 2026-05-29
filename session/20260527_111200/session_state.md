# Session State

- Session: 20260527_111200
- Repo: /home/ubuntu/workspace/NVIDIA/RL/NeMoRL
- Branch: diffusion/sde-algo
- Started: 2026-05-27 11:12 local
- Updated: 2026-05-27 11:14 local

## Goal
Implement diffusion-GRPO (Qwen-Image Flow-GRPO) in NeMo-RL, mirroring `verl-omni`'s `flowgrpo_trainer/`. Single-GPU 3080 Ti smoke must pass with `tiny-random/Qwen-Image`. Compare against verl-omni TSV trends.

## Current Subtask
S5 (`QwenImagePipelineAdapter` in `nemo_rl/models/diffusion/pipeline.py`) with end-to-end logprob parity test.

## Loaded Skills
- `auto-research` — base research workflow.
- `config-conventions` — YAML defaults, TypedDicts, no hidden code defaults.
- `testing` — coverage pragma for Ray actors; nightly conventions.
- `session-memory` — this checkpoint.

## Current Status
- S0 done: `docs/design-docs/diffusion-grpo.zh.md` + index registration.
- S1 done: `DiffusionPolicyConfig` / `DiffusionGRPOAlgoConfig` / `DiffusionLossConfig` / `DiffusionPipelineCfg` / `DiffusionAlgoCfg` / `DiffusionLoraCfg` in `interfaces.py`; `compute_window_mask` in `sde.py`. 10 SDE tests pass.
- S2 done: `nemo_rl/algorithms/loss/diffusion_grpo.py:DiffusionGRPOLossFn`; 10 unit tests pass.
- S3 done: `nemo_rl/data/datasets/text_to_image_prompt.py`; 6 tests pass.
- S4 done: `nemo_rl/environments/image_reward_environment.py` + `DummyImageReward` + Ray pool; 4 tests pass (2 ray-based).
- Total new tests: 30, all passing under `uv run --frozen --group test`.
- `uv` installed under `~/.local/bin`. Megatron-core 0.17.0 installed from wheel (the full mcore extra build fails on this box due to nvidia-resiliency-ext; wheel suffices for unit-test import paths). Diffusers 0.37.1 + peft 0.19.1 installed.

## Plan
- [ ] S5: `pipeline.py:QwenImagePipelineAdapter` + `tests/unit/models/diffusion/test_qwen_image_pipeline_adapter.py`. Core requirement: shared `_for_each_window_step()` between sampling and recompute; parity test with a tiny in-test transformer (fp32 < 1e-4, bf16 < 1e-2).
- [ ] S6: `nemo_rl/models/diffusion/workers/diffusion_worker.py:DiffusionPolicyWorker` — Ray actor mirroring `dtensor_policy_worker.py` shape.
- [ ] S7: `nemo_rl/models/diffusion/policy.py:DiffusionPolicy` — controller facade.
- [ ] S8: `nemo_rl/algorithms/diffusion_grpo.py:diffusion_grpo_train`.
- [ ] S9: `examples/configs/diffusion_grpo_qwen_image{,_tiny}.yaml` + `examples/run_diffusion_grpo.py`.
- [ ] S10: `tests/functional/diffusion/test_diffusion_grpo_smoke.py` + `test_logprob_parity.py`.
- [ ] S11: `docs/guides/diffusion-grpo.md` + cleanup.

## Assumptions
- `tiny-random/Qwen-Image` (HF 2025-08-05) is loadable through `QwenImagePipeline.from_pretrained`. Verify in S6 worker init; if pipeline files are missing, fall back to hand-constructed tiny.
- LoRA via PEFT is the default smoke path. Single-GPU FSDP2 effectively no-op; OK.
- Reward workers default CPU placement for smoke; configurable to GPU for production.
- Reference transformer is the base model with LoRA adapter disabled (no extra copy) in LoRA mode.

## Blockers
- None known. The full `--extra mcore` build fails locally (nvidia-resiliency-ext needs build deps), so we used `uv pip install megatron-core` standalone wheel. Tests do not require the full mcore extra in unit scope.
