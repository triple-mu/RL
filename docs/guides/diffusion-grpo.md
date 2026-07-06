# Diffusion-GRPO (Qwen-Image, flow-grpo)

Status: experimental, single-node only (multi-GPU data parallelism within the
node). The Chinese design document at
[`design-docs/diffusion-grpo.zh.md`](../design-docs/diffusion-grpo.zh.md) is the
authoritative description of the algorithm, data contract, and the alignment
against [`verl-omni`](https://github.com/volcengine/verl-omni)'s
`flowgrpo_trainer/` recipe.

Configs are validated at startup against
`nemo_rl.algorithms.diffusion_grpo.DiffusionMasterConfig` (pydantic v2 schema;
field defaults live on the BaseModel classes in
`nemo_rl/models/diffusion/interfaces.py`). Training auto-resumes from the
newest complete `step_N` directory under `checkpointing.checkpoint_dir`; note
the dataloader position is not restored on resume.

## What it does

`nemo_rl/algorithms/diffusion_grpo.py:diffusion_grpo_train` implements
group-relative policy optimization for flow-matching diffusion models. The
training loop mirrors the token-GRPO phases (rollout → reward → advantage →
train → validate → checkpoint), but the *rollout* is an SDE denoising
trajectory (`x_t → x_{t+1}`), the *log-prob* is the SDE Gaussian density per
step, and the *reward* comes from an image reward environment rather than a
text verifier.

## Single-GPU smoke

```bash
bash tests/functional/diffusion_grpo_smoke.sh
```

The smoke driver runs 5 steps against `tiny-random/Qwen-Image` with the
`DummyImageReward` plugin, configured by
[`examples/configs/diffusion_grpo_qwen_image_tiny.yaml`](../../examples/configs/diffusion_grpo_qwen_image_tiny.yaml).

The nightly recipe (real `Qwen/Qwen-Image` + PickScore, 8-GPU DP) lives at
`examples/configs/recipes/diffusion/grpo-qwen-image-1n8g-dp8-lora.yaml` with
its driver in `tests/test_suites/diffusion/`.

## Production config

[`examples/configs/diffusion_grpo_qwen_image.yaml`](../../examples/configs/diffusion_grpo_qwen_image.yaml)
targets `Qwen/Qwen-Image` with LoRA rank 32 on an 8-GPU node. Override with
Hydra-style args, e.g.:

```bash
uv run python examples/run_diffusion_grpo.py \
    --config examples/configs/diffusion_grpo_qwen_image.yaml \
    grpo.num_generations_per_prompt=8 \
    policy.algo.noise_level=0.5
```

## Algorithm knobs

The most-tuned hyperparameters (with verl-omni equivalents in parentheses):

- `policy.algo.noise_level` (`actor_rollout_ref.rollout.algo.noise_level`):
  magnitude of SDE noise injected inside the active window. Larger = more
  diversity, lower visual quality.
- `policy.algo.sde_window_size` (`actor_rollout_ref.rollout.algo.sde_window_size`):
  how many consecutive denoising steps participate in the policy gradient.
  Smaller = cheaper training step, less coverage.
- `policy.algo.sde_window_range` (`actor_rollout_ref.rollout.algo.sde_window_range`):
  `[start, end]` envelope from which the active window is sampled.
- `loss_fn.ratio_clip_{min,max}` (`actor.diffusion_loss.clip_ratio`): PPO-style
  ratio clipping bounds.
- `loss_fn.adv_clip_max` (`actor.diffusion_loss.adv_clip_max`): pre-ratio
  advantage clamp; matches `FlowGRPOLoss.compute_loss`.
- `loss_fn.beta`: coefficient for the Gaussian-mean KL term against the
  reference policy. Set to 0 to disable KL (default smoke config).

## Reward plugins

`nemo_rl/environments/image_reward_environment.py` ships three plugins:

- `dummy` — deterministic (prompt hash + image mean), CPU-only; smoke tests.
- `jpeg_compressibility` — `-jpeg_kb/500`, ported from verl-omni; used by the
  `_tiny_jpeg*` comparison configs.
- `pickscore` — the PickScore_v1 CLIP-H preference model (~4GB download on
  first use; works on both transformers 4.x and 5.x). Used by the h200 config
  and the nightly recipe.

To register a new reward plugin:

```python
from nemo_rl.environments.image_reward_environment import register_image_reward

class MyReward:
    name = "my_reward"
    weight = 1.0
    def score(self, images, prompts, metadata):
        return {"score": ...}

register_image_reward("my_reward", lambda: MyReward())
```

Then reference it in the YAML:

```yaml
env:
  image_reward:
    plugins:
      - name: my_reward
        weight: 0.7
      - name: dummy
        weight: 0.3
```

## Comparing against verl-omni

The Chinese design document includes a section-by-section mapping of NeMo-RL
symbols against verl-omni's diffusion stack. A practical TSV-based comparison
recipe is documented there under "verl-omni 基线对照"; in short, run
`verl-omni examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh` with the same
`tiny-random/Qwen-Image` substitution and compare `policy_loss`, `mean_ratio`,
`clipfrac`, and `kl_loss` curves rather than absolute numbers.
