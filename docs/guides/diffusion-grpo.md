# Diffusion-GRPO (Qwen-Image, flow-grpo)

Status: experimental, single-node only (multi-GPU data parallelism within the
node). The Chinese design document at
[`design-docs/diffusion-grpo.zh.md`](../design-docs/diffusion-grpo.zh.md) is the
authoritative description of the algorithm, the data contract, and the
parallelism design.

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

With `cluster.gpus_per_node > 1` the policy runs data-parallel: rollout
prompts are scattered across workers and gradients are all-reduced, so
samples-per-step scales with the GPU count. `grpo.num_prompts_per_step` must
be a multiple of the worker count; the `train/dp_checksum_spread` metric
should stay at exactly `0.0` (it monitors that all ranks hold identical
weights).

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

Point `data.train.prompt_file` / `data.val.prompt_file` at your prompt jsonl
files; `tools/export_diffusion_prompts.py` exports a deduplicated,
fixed-seed train/val split from a Hugging Face dataset.

## Algorithm knobs

The most-tuned hyperparameters:

- `policy.algo.noise_level`: magnitude of SDE noise injected inside the
  active window. Larger = more diversity, lower visual quality.
- `policy.algo.sde_window_size`: how many consecutive denoising steps
  participate in the policy gradient. Smaller = cheaper training step, less
  coverage.
- `policy.algo.sde_window_range`: `[start, end]` envelope from which the
  active window is sampled.
- `loss_fn.ratio_clip_{min,max}`: PPO-style ratio clipping bounds.
- `loss_fn.adv_clip_max`: pre-ratio advantage clamp.
- `loss_fn.beta`: coefficient for the Gaussian-mean KL term against the
  reference policy. Set to 0 to disable KL (default smoke config).
- `policy.train_micro_batch_size`: chunk size for the with-grad logprob
  recompute during training; bounds peak memory independently of the global
  batch size.

## Reward plugins

`nemo_rl/environments/image_reward_environment.py` ships three plugins:

- `dummy` — deterministic (prompt hash + image mean), CPU-only; smoke tests.
- `jpeg_compressibility` — rule-based `-jpeg_kb/500`, zero external
  dependencies.
- `pickscore` — the PickScore_v1 CLIP-H preference model (~4GB download on
  first use; works on both transformers 4.x and 5.x). Used by the exemplar
  config and the nightly recipe. Scores on CPU workers by default; to score
  on a GPU, reserve one by lowering `cluster.gpus_per_node` and set
  `env.image_reward.num_gpus_per_worker: 1.0`.

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

Images arrive as CPU `NCHW` float tensors in `[0, 1]`; a GPU-scoring plugin
moves them to its device and returns CPU scores.
