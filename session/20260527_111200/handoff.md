# Handoff

## Resume From Here
Implementing diffusion-GRPO on branch `diffusion/sde-algo`. The plan file at `/home/ubuntu/.claude/plans/stateless-dazzling-swing.md` is approved and lists S0–S11. S0–S4 are done; the design doc at `docs/design-docs/diffusion-grpo.zh.md` is the alignment source. Next: S5 (pipeline adapter) with logprob parity test.

## Next Actions
1. Write `nemo_rl/models/diffusion/pipeline.py:QwenImagePipelineAdapter`. Use diffusers 0.37.1 `QwenImagePipeline` and `FlowMatchEulerDiscreteScheduler`. Share `_for_each_window_step()` between `sample_trajectory` and `compute_transition_logprob`. Cite verl-omni `pipelines/qwen_image_flow_grpo/diffusers_training_adapter.py` for CFG/norm-rescale conventions.
2. Write `tests/unit/models/diffusion/test_qwen_image_pipeline_adapter.py` constructing a tiny `QwenImageTransformer2DModel(patch_size=2, num_layers=1, num_attention_heads=1, attention_head_dim=16)` (or smaller). Sample 4 steps with manual_seed and recompute; assert `max|Δlogprob| < 1e-4` fp32, `< 1e-2` bf16.
3. Run via `PATH="$HOME/.local/bin:$PATH" uv run --frozen --group test python -m pytest tests/unit/models/diffusion/test_qwen_image_pipeline_adapter.py -v`.

## Watch Outs
- The user is explicit: "verl-omni" is the parity target, not the older "verl" repo. Always look in `/home/ubuntu/workspace/NVIDIA/RL/verl-omni`.
- Do NOT touch token-GRPO files: `grpo.py`, `lm_policy.py`, the policy workers under `nemo_rl/models/policy/workers/`, `environments/interfaces.py`.
- The repo uses `uv`; tests need `--group test`. Megatron is required because `nemo_rl/algorithms/loss/__init__.py` eagerly imports `loss_functions.py` which transitively imports megatron-core. Wheel install is sufficient on this box.
- User wants Chinese design output. Code stays English.
- Single-GPU 3080 Ti, 16 GB; default smoke = `tiny-random/Qwen-Image`, LoRA, T=8, K=2, B=1, 128×128.
- `# pragma: no cover` MUST appear on the line declaring any `@ray.remote` class/function (testing skill).

## Quick Verification Commands
```bash
export PATH="$HOME/.local/bin:$PATH"
git status -s
uv run --frozen --group test python -m pytest \
  tests/unit/models/diffusion/test_sde.py \
  tests/unit/algorithms/loss/test_diffusion_grpo_loss.py \
  tests/unit/data/test_text_to_image_prompt_dataset.py \
  tests/unit/environments/test_image_reward_environment.py \
  -v 2>&1 | tail -40
```
Expected: 30 passed.
