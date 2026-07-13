# Diffusion GRPO (Qwen-Image, Flow-GRPO)

This guide is the authoritative documentation for the `diffusion_grpo` algorithm
path: group-relative policy optimization for flow-matching text-to-image models,
following [Flow-GRPO](https://arxiv.org/abs/2505.05470)
([yifan123/flow_grpo](https://github.com/yifan123/flow_grpo)). The first
supported model is [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image).

Status: experimental, single-node only (multi-GPU data parallelism within the
node, LoRA training). The original module-level design notes live in
[`design-docs/diffusion-grpo.zh.md`](../design-docs/diffusion-grpo.zh.md)
(Chinese, historical); where the two disagree, this guide reflects the current
code.

## Motivation and Scope

NeMo RL's token GRPO is built around token outputs: `message_log` →
vLLM/SGLang/Megatron generation → token log-probs → `ClippedPGLossFn`. None of
that fits a diffusion policy, so `diffusion_grpo` is a separate algorithm path
(`nemo_rl/algorithms/diffusion_grpo.py:diffusion_grpo_train`) that leaves the
existing token GRPO code untouched:

| Dimension | LLM/VLM GRPO | Diffusion GRPO |
|---|---|---|
| Action | next token | latent transition `x_t → x_{t+1}` |
| Trajectory | token sequence | latent trajectory `[B, T+1, ...]` |
| Log-prob | categorical token log-prob | per-step SDE Gaussian log-density |
| Reward input | text / message log | image tensor + prompt + metadata |
| Generation backend | vLLM / SGLang / Megatron | the diffusion policy workers themselves |
| Refit | policy weights synced to the generation engine | not needed (sampling and training share the workers) |
| Advantage shape | expanded over tokens | expanded over timesteps |
| Training subtlety | sequence packing affects normalization | the SDE window decides which steps enter the loss |
| Advantage estimator name | `grpo` | `diffusion_grpo` |

The training loop mirrors the token-GRPO phases (rollout → reward → advantage →
train → validate → checkpoint), but the rollout is an SDE denoising trajectory,
the log-prob is the SDE Gaussian density per step, and the reward comes from an
image reward environment rather than a text verifier. Advantages reuse
`calculate_baseline_and_std_per_prompt` (per-prompt group baseline over the K
generations of each prompt).

With `cluster.gpus_per_node > 1` the policy runs data-parallel: rollout prompts
are scattered across workers and gradients are all-reduced, so samples-per-step
scale with the GPU count. `grpo.num_prompts_per_step` must be a multiple of the
worker count, and the `train/dp_checksum_spread` metric must stay at exactly
`0.0` (it monitors that all ranks hold identical trainable weights).

Configs are validated at startup against
`nemo_rl.algorithms.diffusion_grpo.DiffusionMasterConfig` (pydantic v2 schema;
field defaults live on the `BaseModel` classes in
`nemo_rl/models/diffusion/interfaces.py`). Training auto-resumes from the
newest complete `step_N` directory under `checkpointing.checkpoint_dir`; note
that the dataloader position is not restored on resume.

## Quickstart

Diffusion dependencies (diffusers, peft, paddleocr) live behind the optional
`diffusion` extra: use `uv run --extra diffusion ...` (or `uv sync --extra
diffusion` once).

### Single-GPU functional test (~16 GB)

```bash
bash tests/functional/diffusion_grpo.sh
```

The functional test runs 5 training steps of `tiny-random/Qwen-Image` with the
`jpeg_compressibility` reward (a real, optimizable signal), mirroring
[`examples/configs/diffusion_grpo_qwen_image_tiny_jpeg.yaml`](../../examples/configs/diffusion_grpo_qwen_image_tiny_jpeg.yaml),
then asserts metric health via `tests/check_metrics.py` (ratio window, bounded
grad norm, checkpoint written). It is wired into the L1 functional CI suite.

### OCR task on 8 GPUs (the reference recipe)

The Flow-GRPO OCR text-rendering task is the reference convergence recipe.
Export the dataset once, then launch:

```bash
uv run python tools/export_ocr_prompts.py --out-dir examples/data/diffusion/ocr
uv run --extra diffusion examples/run_diffusion_grpo.py \
    --config examples/configs/diffusion_grpo_qwen_image_ocr.yaml
```

[`examples/configs/diffusion_grpo_qwen_image_ocr.yaml`](../../examples/configs/diffusion_grpo_qwen_image_ocr.yaml)
targets `Qwen/Qwen-Image` with LoRA rank 64 on an 8-GPU node
(32 prompts × 16 generations = 512 samples/step), mirroring the verl-omni
`run_qwen_image_ocr_lora.sh` hyperparameters. Override any field with
Hydra-style dotted args, e.g. `policy.algo.noise_level=1.0
logger.wandb_enabled=True`.

The general-purpose exemplar (PickScore reward, pickapic prompts) is
[`examples/configs/diffusion_grpo_qwen_image.yaml`](../../examples/configs/diffusion_grpo_qwen_image.yaml);
nightly recipes live under `examples/configs/recipes/diffusion/` with drivers
in `tests/test_suites/diffusion/`.

## Data and Rewards

### Prompt files

`data.train.prompt_file` / `data.val.prompt_file` accept two formats
(`nemo_rl/data/datasets/text_to_image_prompt.py`):

- `.txt` — one prompt per line.
- `.jsonl` — one object per line:

  ```json
  {"prompt": "A sign that says \"HELLO\"", "negative_prompt": " ", "metadata": {"ground_truth": "HELLO"}}
  ```

  `negative_prompt` is optional (defaults to `" "`, Qwen-Image's recommended
  usage). `metadata` is optional and passed through to reward plugins — the
  `ocr` reward reads `metadata["ground_truth"]`.

Two export tools generate these files:

- `tools/export_ocr_prompts.py` — downloads the Flow-GRPO OCR dataset and
  writes jsonl with the quoted text as `metadata.ground_truth`. Yields
  **19,653 train / 1,018 val** prompts. (The upstream txt files lack a trailing
  newline, so `wc -l` on them undercounts by one.)
- `tools/export_diffusion_prompts.py` — exports a deduplicated, fixed-seed
  train/val split from a Hugging Face text dataset (used for the pickapic
  prompts in the exemplar config).

### Reward plugins

`nemo_rl/environments/image_reward_environment.py` ships four plugins:

| Plugin | Signal | Dependencies |
|---|---|---|
| `dummy` | deterministic (prompt hash + image mean); unit/functional tests only | none |
| `jpeg_compressibility` | rule-based `-jpeg_kb/500` | none |
| `pickscore` | PickScore_v1 CLIP-H preference model (~4 GB download on first use; transformers 4.x and 5.x) | transformers |
| `ocr` | PaddleOCR (en) + edit distance vs `metadata["ground_truth"]`: substring hit → 1.0, else `1 - dist/len(gt)` | paddleocr 2.x via the `diffusion` extra |

Rewards are scored by a Ray worker pool, one actor group per plugin. Each
plugin returns a component dict; the environment sums components weighted by
the per-plugin `weight` and logs `reward/<plugin>/<component>_mean` plus
`reward/total_mean`. Placement is controlled by `env.image_reward.num_{cpus,gpus}_per_worker`;
`num_workers_per_plugin` adds replicas for slow CPU rewards (the OCR config
uses 16 replicas to keep up with 512 images/step). Plugins score on CPU by
default; to score on a GPU, reserve one by lowering `cluster.gpus_per_node` and
set `num_gpus_per_worker: 1.0`.

To register a custom reward, use the `register_image_reward` extension point:

```python
from nemo_rl.environments.image_reward_environment import register_image_reward

class MyReward:
    name = "my_reward"
    weight = 1.0

    def score(self, images, prompts, metadata):
        # images: CPU NCHW float tensor in [0, 1]; a GPU-scoring plugin moves
        # them to its device and returns CPU tensors.
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
      - name: jpeg_compressibility
        weight: 0.3
```

## Configuration Reference

Annotated walk-through of the exemplar sections (defaults live on the schema in
`nemo_rl/models/diffusion/interfaces.py`; the YAML files document them):

### `grpo`

- `num_prompts_per_step` — unique prompts per rollout step (the GRPO group
  count). Must be a multiple of the DP worker count.
- `num_generations_per_prompt` — K images sampled per prompt for the group
  baseline.
- `ppo_epochs` — number of optimization epochs over each rollout batch;
  `generation_logprobs` stay fixed while `curr_logprob` is recomputed per
  epoch.
- `use_leave_one_out_baseline` — leave-one-out group baseline (recommended).
- `val_generation.num_inference_steps: 40` — validation always samples with
  the **deterministic ODE** (no SDE noise, no log-prob collection) regardless
  of the training rollout settings; only its denoising step count is
  configurable here. Validation uses K=1 and a fixed seed so `val/reward_mean`
  and the saved sample images are comparable across steps.

### `loss_fn`

- `ratio_clip_min` / `ratio_clip_max` — PPO-style clip: ratio clipped to
  `[1 - min, 1 + max]`. See the red lines below for window-mode values.
- `adv_clip_max` — advantages clamped to ± this before the ratio product.
- `beta` — Gaussian-mean KL coefficient against the reference policy; `0`
  disables KL entirely (the reference transformer is not even loaded).
  Nonzero `beta` currently requires LoRA (the reference is the base model with
  adapters disabled).
- `aggregate_logprobs_per_sample` — `false` (default) uses per-`(B, T)`
  elements (Flow-GRPO paper formulation); `true` sums log-probs over T first
  (verl-omni's per-sample formulation).

### `policy`

- `train_micro_batch_size` — samples per backward pass (gradient
  accumulation). Bounds peak memory of the with-grad log-prob recompute
  independently of the global batch; `null` recomputes the full local batch at
  once.
- `optimizer.max_grad_norm` — global gradient-norm clipping threshold applied
  after the DP all-reduce, before `optimizer.step()`.
- `seed` — required for multi-GPU DP: all ranks must initialize LoRA
  identically (missing seed is a startup error).
- `pipeline.num_inference_steps` — denoising steps T per training rollout
  (validation overrides its own count, see above).
- `pipeline.true_cfg_scale` — classifier-free guidance scale (`1.0` disables
  CFG).

### `policy.algo` (SDE window)

- `noise_level` — magnitude of the SDE noise injected inside the active
  window. Larger = more exploration, lower visual quality.
- `sde_type` — `"sde"` (Flow-GRPO Gaussian log-prob) or `"cps"` (simplified).
- `sde_window_size` — how many consecutive denoising steps participate in the
  policy gradient; `null` = all steps.
- `sde_window_range` — `[start, end)` envelope: the window start is sampled
  **uniformly at random from `[start, end - size]` on every rollout** (from a
  dedicated `seed + 2` random stream — the initial latents use `seed` and the
  SDE noise `seed + 1`, so the three streams never overlap). `null` fixes the
  window start at 0.

Steps outside the active window are sampled with the deterministic mean and
excluded from the loss. The training data is sliced down to the active window
before the log-prob recompute, so a small window also makes the training step
proportionally cheaper.

### `env.image_reward`, `cluster`, `checkpointing`

- `env.image_reward.plugins` — list of `{name, weight}` entries; components
  are weighted and summed into the total reward.
- `env.image_reward.num_workers_per_plugin` — reward actor replicas per
  plugin; raise for slow CPU rewards such as OCR.
- `cluster.gpus_per_node` — all GPU bundles go to DP policy workers (one full
  model replica each).
- `checkpointing.save_period` — save every N steps; training auto-resumes
  from the newest complete `step_N` (LoRA adapter or full state dict +
  optimizer state, written by rank 0 only).

## Debugging and Sanity Procedures

### Metrics reference

| Metric | Meaning |
|---|---|
| `train/loss`, `train/policy_loss` | total loss (policy + `beta`·KL) and its policy-gradient part |
| `train/approx_kl` | first-order KL estimate between sampling and current policy |
| `train/clipfrac`, `train/clipfrac_higher`, `train/clipfrac_lower` | fraction of elements hitting the ratio clip (total / upper / lower) |
| `train/mean_ratio`, `train/ratio_min`, `train/ratio_max` | importance-ratio statistics inside the window mask |
| `train/grad_norm`, `train/lr` | pre-clip global gradient norm; current learning rate |
| `train/advantage_mean`, `train/advantage_std`, `train/reward_mean` | rollout batch statistics |
| `train/dp_checksum_spread` | max − min of per-rank trainable-parameter checksums; must be exactly `0.0` |
| `reward/<plugin>/<component>_mean`, `reward/total_mean` | per-component and weighted-total reward means |
| `timing/*` | per-phase wall-clock durations |
| `val/reward_mean` | deterministic-ODE validation reward |

### L0-a: on-policy ratio check

With `grpo.ppo_epochs=1` and `policy.train_micro_batch_size=null`, the
training-time log-prob recompute sees exactly the trajectories that were just
sampled, before any optimizer step — so the first-step `train/mean_ratio`
must be ≈ 1 (and `train/approx_kl` ≈ 0):

```bash
uv run --frozen --extra diffusion python examples/run_diffusion_grpo.py \
    --config examples/configs/diffusion_grpo_qwen_image_tiny_jpeg.yaml \
    grpo.ppo_epochs=1 \
    policy.train_micro_batch_size=null
```

Measured on an H200, the tiny jpeg config yields `train/mean_ratio` = 0.99996. Any
real deviation from 1 means the sampling path and the training recompute path
have numerically diverged (scheduler indexing, dtype, CFG, or window handling)
— the classic Flow-GRPO failure mode, cf. flow_grpo issues
[#172](https://github.com/yifan123/flow_grpo/issues/172) and
[#211](https://github.com/yifan123/flow_grpo/issues/211). Both paths share one
denoise-step helper (`sde_step_with_logprob`, fp32 math) precisely to keep this
invariant; treat a broken ratio check as a bug, never as something to tune
around.

### L0-b: fake-reward overfit

`jpeg_compressibility` is a trivially optimizable reward. On the tiny + jpeg
config (`bash tests/functional/diffusion_grpo.sh`),
`train/reward_mean` must visibly rise within the 5 training steps. If the ratio
check passes but a fake reward does not go up, the optimization plumbing
(advantages, masking, optimizer wiring) is broken — do not proceed to real
rewards.

### L1: healthy metric ranges during real training

- `train/clipfrac_higher` vs `train/clipfrac_lower` should stay roughly
  symmetric. Persistent asymmetry means a systematic ratio bias and is a
  precursor of reward over-optimization.
- `train/approx_kl` should stay small and bounded — a steady upward drift
  means the policy is escaping the trust region.
- `train/grad_norm` should be spike-free; isolated spikes are absorbed by
  `optimizer.max_grad_norm`, recurring spikes call for a lower learning rate
  or noise level.
- `train/dp_checksum_spread` must be exactly `0.0` on every step.
- `val/reward_mean` should trend with `train/reward_mean`; training reward
  rising while validation reward stalls indicates over-optimization of the
  reward model.

### Config red lines

- **Group size ≥ 16.** `grpo.num_generations_per_prompt` below 16 makes the
  group baseline/std too noisy for stable advantages.
- **Window mode needs a tiny clip.** With `sde_window_size` <
  `num_inference_steps`, set `ratio_clip_{min,max}` to ~1e-4–1e-5 (the OCR
  config uses 1e-4). The default 0.2 clip destabilizes window ("Fast") mode —
  see the flow_grpo FAQ.
- **Rollout and training batch semantics must match.** Every rollout sample is
  trained each step: the global batch is `num_prompts_per_step ×
  num_generations_per_prompt`, and `policy.train_micro_batch_size` is only
  gradient accumulation, never sub-sampling. Keep `num_prompts_per_step`
  divisible by the DP worker count.

## Known Limitations and Roadmap

- **Single-node data parallelism, LoRA-first.** Each DP worker holds a full
  model replica; there is no FSDP2/TP sharding on this path yet. Large-scale
  full-parameter training (FSDP2) is planned for a later phase.
- **`loss_fn.beta > 0` requires LoRA.** The KL reference policy is implemented
  as the base model with adapters disabled; full-parameter training currently
  has no reference path.
- **Image generation only.** Image editing and video models (e.g. Wan) are on
  the roadmap (Phase 2/3), not in this release.
- **No built-in GenEval reward server.** Detector-based rewards such as
  GenEval require an external service; integrate them via
  `register_image_reward`.
- **Resume is minimal.** Auto-resume restores model/optimizer state from the
  newest `step_N`, but not the dataloader position.
