# NeMo-RL Diffusion-GRPO 设计文档（中文）

最后更新：2026-05-27。

本文档说明如何在 NeMo-RL 中引入新的 `diffusion_grpo` 算法路径，首期支持 [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) 文生图模型的 Flow-GRPO 训练（[paper](https://arxiv.org/abs/2505.05470)、[code](https://github.com/yifan123/flow_grpo)）。设计在数学上对齐
[`verl-omni`](https://github.com/volcengine/verl-omni) `examples/flowgrpo_trainer/`，在工程上保留 NeMo-RL 的 `BatchedDataDict` + `RayWorkerGroup` 风格，避免污染既有 token GRPO 路径。

## 1. 动机与目标

NeMo-RL 现有 GRPO 围绕 token 输出设计：`message_log` → vLLM/SGLang/Megatron 生成 → token logprob → `ClippedPGLossFn`。该路径不适合 diffusion 图像/视频输出：
action 是 latent transition `x_t → x_{t+1}`，logprob 是 SDE Gaussian 密度，reward 输入是 image tensor，且不需要 generation backend 与 policy 之间的权重 refit。

首期目标：

- Qwen-Image 文生图 GRPO 训练，覆盖 SDE flow-matching rollout、per-timestep logprob、per-prompt group baseline。
- LoRA 训练为默认，full-parameter 通过配置开启。
- 可选 reference transformer KL；KL 系数为 0 时不加载 reference。
- 支持 PickScore / ImageReward 风格的 image reward（首期落地 `DummyImageReward`，其余留接口）。
- 在单卡 RTX 3080 Ti (16 GB) + `tiny-random/Qwen-Image` 上跑通 smoke。
- **不改动** 既有 token GRPO 路径（`grpo.py / lm_policy.py / dtensor_policy_worker.py / environments/interfaces.py`）。

非目标：critic / value model（GRPO 范式不需要）；vLLM/SGLang 生成后端用于 diffusion；image-edit / video（后续 PR）；Megatron-Core 后端的 diffusion 适配（首期仅 dtensor/FSDP2 风格）。

## 2. 与 token GRPO 的对照

| 维度 | LLM/VLM GRPO | Diffusion GRPO |
|---|---|---|
| action | next token | latent transition `x_t → x_{t+1}` |
| trajectory | token sequence | latent trajectory `[B, T+1, ...]` |
| logprob | categorical token logprob | SDE Gaussian log-density |
| reward 输入 | text / message log | image tensor + prompt + metadata |
| generation backend | vLLM / SGLang / Megatron | diffusion policy worker 本身 |
| refit | 需要将权重同步到生成引擎 | 不需要 |
| advantage 维度 | token expanded | timestep expanded |
| 训练时机微妙处 | seq packing 影响归一 | SDE window 决定哪些步参与 loss |
| advantage estimator 名称 | `grpo` | `diffusion_grpo` |

因此新增独立算法路径 `diffusion_grpo` 比把分支逻辑塞进 `nemo_rl/algorithms/grpo.py` 更清晰，也更容易和 NeMo-RL
既有 token GRPO 解耦演进。

## 3. 总体架构

```text
nemo_rl/algorithms/diffusion_grpo.py
  │
  ├── TextToImagePromptDataset
  │      → prompt / negative_prompt / metadata
  │
  ├── DiffusionPolicy
  │      → RayWorkerGroup
  │         → DiffusionPolicyWorker
  │            → QwenImagePipelineAdapter
  │               ├── trainable transformer (DiT)
  │               ├── frozen text_encoder
  │               ├── frozen vae
  │               ├── optional frozen transformer_ref（LoRA 模式：禁 adapter 复用 base）
  │               └── FlowMatch scheduler + SDE logprob kernel
  │
  ├── ImageRewardEnvironment
  │      → _RewardWorker pool
  │         → DummyImageReward / PickScoreReward / ImageRewardModelReward
  │
  ├── GRPO advantage：复用 calculate_baseline_and_std_per_prompt
  ├── DiffusionGRPOLossFn
  └── Logger / Timer / CheckpointManager
```

采样和训练共享同一组 diffusion policy workers，不引入 vLLM / SGLang，也不需要 policy-to-generation refit。

## 4. Actor、Reward、Critic

### Actor

可训练 diffusion transformer / DiT。Qwen-Image 对应 `QwenImageTransformer2DModel`。
LoRA 为默认（rank=4/8/16 等），full-parameter 通过 `policy.dtensor_cfg.lora_cfg.enabled=false` 启用。

### 冻结组件

- `text_encoder`：负责 prompt encoding，无梯度。
- `vae`：负责 latent decode，无梯度。
- `scheduler`：flow-matching timestep 与 sigma；状态量，不更新。
- `transformer_ref`：可选；仅在 `loss_fn.beta > 0` 时启用。LoRA 模式下通过禁用 PEFT adapter 复用 base 模型权重，无额外显存开销；full-parameter 模式下保留独立冻结副本。

未来扩展：image editing 增加 reference image encoder；video diffusion 增加 temporal latent 相关处理。

### Reward

独立 Ray reward worker pool。输入 `(images, prompts, metadata)`，输出 `(weighted_total_reward, component_metrics)`。首期插件：

- `DummyImageReward`：确定性 reward（prompt 哈希 + image 均值），用于 smoke 与单测；
- `PickScoreReward` / `ImageRewardModelReward`：仅落骨架与延迟加载守卫，后续 PR 完成。

### Critic

不引入 critic。GRPO 用同一 prompt 下 K 张图的 reward 分布计算 group baseline 与 std。
直接复用 NeMo-RL 既有 `nemo_rl.algorithms.utils.calculate_baseline_and_std_per_prompt`。

## 5. 数据契约

### DiffusionDatumSpec

```python
class DiffusionDatumSpec(TypedDict):
    prompt: str
    negative_prompt: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]
    idx: int
    loss_multiplier: float
    task_name: NotRequired[str]
```

`negative_prompt` 默认 `" "`，对齐 flow_grpo 的 Qwen-Image recipe。

### DiffusionTrajectorySpec（rollout 产物）

```python
class DiffusionTrajectorySpec(TypedDict):
    prompts: list[str]
    negative_prompts: list[str]
    metadata: list[dict[str, Any]]
    images: torch.Tensor                    # [B*K, 3, H, W]
    latents: torch.Tensor                   # [B*K, T+1, C, H', W']
    timesteps: torch.Tensor                 # [B*K, T] 或 [T]
    generation_logprobs: torch.Tensor       # [B*K, T]
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_embeds_mask: torch.Tensor
```

### DiffusionTrainDataSpec（loss 输入）

```python
class DiffusionTrainDataSpec(TypedDict):
    latents: torch.Tensor                   # [B*K, T+1, C, H', W']
    timesteps: torch.Tensor
    generation_logprobs: torch.Tensor       # 旧 policy log-prob（即 old_log_probs）
    advantages: torch.Tensor                # [B*K, T]，由 [B*K] 扩展
    timestep_mask: torch.Tensor             # SDE window mask
    sample_mask: torch.Tensor               # = loss_multiplier
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_embeds_mask: torch.Tensor
    reference_policy_mean: NotRequired[torch.Tensor]
    current_policy_mean: NotRequired[torch.Tensor]
    std_dev: NotRequired[torch.Tensor]
```

Diffusion 训练不依赖 `input_ids`、`output_ids`、`token_mask` 或 `message_log`。

## 6. 模块清单

### `nemo_rl/algorithms/diffusion_grpo.py`

主训练入口 `diffusion_grpo_train(policy, env, train_dataset, val_dataset, loss_fn, logger, checkpointer, save_state, master_config)`。相位：

```text
batch prompts
   ├─ repeat K → B*K
   ▼ policy.sample_trajectory
collect trajectory + generation_logprobs
   ▼ env.score_images
rewards [B*K]
   ▼ calculate_baseline_and_std_per_prompt + advantage 扩到 timestep
train_data
   ▼ policy.train
loss / metrics
   ▼ log / validate / checkpoint
```

### `nemo_rl/models/diffusion/interfaces.py`

已存在；本期追加 `DiffusionPolicyConfig`、`DiffusionGRPOAlgoConfig`、`DiffusionLossConfig` 三个
`TypedDict`（遵循 `config-conventions` 技能：YAML 是默认源，不在 Python 写 defaults）。

### `nemo_rl/models/diffusion/sde.py`

已存在；本期追加 `compute_window_mask(num_steps, window_start, window_size) -> Tensor`，供
pipeline 与 loss 共享，避免重复定义 window 语义。

### `nemo_rl/models/diffusion/pipeline.py`

实现 `QwenImagePipelineAdapter`，对应 verl-omni `verl_omni/pipelines/qwen_image_flow_grpo/`。

公共方法：

```python
def encode_condition(prompts, negative_prompts) -> dict[str, Tensor]
def sample_trajectory(prompts, negative_prompts, metadata, *, K) -> DiffusionTrajectorySpec
def compute_transition_logprob(data, *, use_reference) -> tuple[Tensor, Tensor, Tensor]
def decode(latents) -> Tensor
```

要点：

- prompt 与 negative prompt 一起 encode 后再 split（flow_grpo 的 CFG concat 路径）。
- 使用 Qwen-Image 的 flow-matching `mu` shift。
- true classifier-free guidance + norm rescale。
- **采样与训练 logprob recompute 必须共享同一段 `_for_each_window_step()` helper**，
  避免 rollout-time 与 train-time 数值分叉。
- SDE window 外的步骤走确定性 mean，且 `generation_logprobs[:, t] = 0`、`timestep_mask[:, t] = 0`。

### `nemo_rl/models/diffusion/workers/diffusion_worker.py`

`@ray.remote class DiffusionPolicyWorker`。**镜像** `nemo_rl/models/policy/workers/dtensor_policy_worker.py`
的结构（`configure_worker / _default_options / __init__ / prepare_for_generation / prepare_for_training /
save_checkpoint / shutdown`），但 **不继承** —— 后者绑定 HF causal-LM `forward`。

方法：

```python
def sample_trajectory(prompts, negative_prompts, metadata, *, K, seed) -> DiffusionTrajectorySpec
def compute_transition_logprob(data, *, use_reference) -> dict[str, Tensor]
def train_step(data, loss_cfg) -> dict[str, float]
def save_checkpoint(path) -> None
```

初始化步骤：

1. `init_process_group("nccl")`（用 `RayWorkerBuilder` 注入的 env vars）。
2. `QwenImagePipeline.from_pretrained(..., torch_dtype=bfloat16)`，拆出
   `transformer / text_encoder / vae / scheduler`。
3. 冻结 `text_encoder` 与 `vae`。
4. LoRA 经 PEFT 注入到 transformer；或 full-parameter。
5. FSDP2 `fully_shard` 包 trainable transformer；单卡时无开销但保留代码路径。
6. Reference 实现：LoRA 模式 = base 模型禁用 adapter；full-param 模式 = 独立冻结副本。
7. 构造 `QwenImagePipelineAdapter`；AdamW 仅训练 trainable params。

### `nemo_rl/models/diffusion/policy.py`

`DiffusionPolicy` 是 controller 侧 façade。要求：

- 使用 `RayVirtualCluster` / `RayWorkerBuilder` / `RayWorkerGroup`；不裸写 `@ray.remote` 列表。
- 不接入 vLLM / SGLang refit。
- 提供 `sample_trajectory / compute_transition_logprob / train / prepare_for_generation /
  prepare_for_training / save_checkpoint / shutdown`。

通过 `run_all_workers_sharded_data` 切分 batch；`run_all_workers_single_data` 处理 broadcast 类参数。

### `nemo_rl/algorithms/loss/diffusion_grpo.py`

`DiffusionGRPOLossFn`。**不实现** `nemo_rl/algorithms/loss/interfaces.py:LossFunction` Protocol，其
`__call__(data, global_valid_seqs, global_valid_toks, **kwargs)` 与 token 语义绑定（next_token_logprobs
或 logits + `LossType.{TOKEN,SEQUENCE}_LEVEL`），不适配 diffusion timestep 概念。把 diffusion loss
放进同一接口要么让接口变臃肿，要么强行映射会让 metrics 命名错位。

签名：

```python
def __call__(
    self,
    curr_logprob: Tensor,            # [B*K, T]
    generation_logprob: Tensor,      # [B*K, T]，对应 verl-omni 的 old_log_prob
    advantages: Tensor,              # [B*K, T]
    timestep_mask: Tensor,           # [B*K, T]
    sample_mask: Tensor,             # [B*K]，= loss_multiplier
    *,
    current_mean: Tensor | None = None,
    reference_mean: Tensor | None = None,
    std_dev: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]
```

公式（与 verl-omni `verl_omni/trainer/diffusion/diffusion_algos.py:FlowGRPOLoss.compute_loss` 对齐）：

```python
advantages = advantages.clamp(-adv_clip_max, adv_clip_max)     # 与 verl-omni 一致
log_ratio   = curr_logprob - generation_logprob
ratio       = log_ratio.exp()
unclipped   = -advantages * ratio
clipped     = -advantages * ratio.clamp(1 - c_min, 1 + c_max)
pg          = torch.maximum(unclipped, clipped)
policy_loss = masked_mean(pg, timestep_mask * sample_mask)

if beta > 0:
    kl       = (current_mean - reference_mean) ** 2 / (2 * std_dev ** 2)
    kl_loss  = masked_mean(kl.mean(dim=spatial_dims), timestep_mask * sample_mask)
    loss     = policy_loss + beta * kl_loss
else:
    loss     = policy_loss
```

返回 metrics：`policy_loss / kl_loss / approx_kl / clipfrac / clipfrac_higher / clipfrac_lower /
mean_ratio / ratio_min / ratio_max`。命名与 verl-omni 的 `actor/ppo_kl`、`actor/pg_clipfrac*`、
`actor/ratio_mean` 等保持语义对齐，但前缀按 NeMo-RL 的 `train/` 风格挂载。

### `nemo_rl/environments/image_reward_environment.py`

新增 image-native reward API：

```python
class BaseImageReward(Protocol):
    name: str
    weight: float
    def score(self, images: Tensor, prompts: list[str],
              metadata: list[dict]) -> dict[str, Tensor]: ...

@ray.remote
class _RewardWorker: ...

class ImageRewardEnvironment:
    def score_images(self, images, prompts, metadata
                     ) -> tuple[Tensor, dict[str, Any]]: ...
    def shutdown(self) -> bool: ...
```

**不继承** `nemo_rl/environments/interfaces.py:EnvironmentInterface`，其 `step(message_log_batch,
metadata)` 假设 token-style 输入，与 image reward 不兼容。

首期插件：`DummyImageReward`（CPU、无 GPU 依赖、确定性）。其它插件留接口与 import 守卫。

### Dataset、Config、Entrypoint

- `nemo_rl/data/datasets/text_to_image_prompt.py:TextToImagePromptDataset`：
  - `.txt`：每行一个 prompt；
  - `.jsonl`：`{"prompt": str, "negative_prompt"?: str, "metadata"?: dict}`。
- `examples/configs/diffusion_grpo_qwen_image.yaml`：生产级配置。
- `examples/configs/diffusion_grpo_qwen_image_tiny.yaml`：单卡 smoke 配置（K=2、T=8、128×128、LoRA rank=4）。
- `examples/run_diffusion_grpo.py`：复用 `run_grpo.py` 的 Hydra 解析框架，但调用 `diffusion_grpo_train`。

## 7. 可复用 NeMo-RL 组件

| 组件 | 路径 |
|---|---|
| `RayVirtualCluster` | `nemo_rl/distributed/virtual_cluster.py` |
| `RayWorkerBuilder` / `RayWorkerGroup` | `nemo_rl/distributed/worker_groups.py` |
| `BatchedDataDict` | `nemo_rl/distributed/batched_data_dict.py` |
| `calculate_baseline_and_std_per_prompt` | `nemo_rl/algorithms/utils.py`（签名 `(prompts, rewards, valid_mask, leave_one_out_baseline, std_rewards)`） |
| `CheckpointManager` | `nemo_rl/utils/checkpoint.py` |
| `Logger` | `nemo_rl/utils/logger.py` |
| `Timer` | `nemo_rl/utils/timer.py` |
| Hydra config 加载与 override helper | `nemo_rl/utils/config.py` |
| YAML 结构基线 | `examples/configs/grpo_math_1B.yaml`（去掉 generation / megatron / async / dynamic-sampling / message-log 相关段） |
| dtensor worker 结构模板 | `nemo_rl/models/policy/workers/dtensor_policy_worker.py` |

## 8. 训练流程详解

### Rollout

1. 读取 prompt batch（size = `num_prompts_per_step`）。
2. 每个 prompt 重复 K = `num_generations_per_prompt` 次。
3. 编码 prompt 与 negative prompt。
4. 准备初始 latent `x_0 ~ N(0, I)`。
5. 在 SDE window 内：对每个 t，transformer forward → CFG → norm rescale → `sde_step_with_logprob`，
   记录 `(latents[t+1], generation_logprobs[t])`。窗外步走确定性 mean，logprob 与 mask 置 0。
6. VAE decode 末步 latent → image tensor。

### Reward

1. 把 `(images, prompts, metadata)` 派给 Ray reward worker pool。
2. 每个 plugin 返回 component dict。
3. 按 `weight` 聚合为 total reward；记录每个 component 的 mean/std。

### Advantage

1. 同一原始 prompt 的 K 张图共享 prompt id。
2. 调 `calculate_baseline_and_std_per_prompt(prompts, rewards, valid_mask, leave_one_out_baseline=True)`，
   把 prompt 编码成 `(B*K, 1)` 整数张量即可。
3. `advantages = (r - baseline) / std.clamp_min(1e-6)`，再 `.unsqueeze(-1).expand(-1, T)`。

### Training

对每个 microbatch、每个训练 timestep：

1. 当前 transformer 对 `x_t` forward（有梯度）。
2. 调 `sde_step_with_logprob(..., prev_sample=latents[:, t+1])` 重算 `curr_logprob`。
3. 可选：base 模型禁 LoRA adapter 再 forward 一次得到 `reference_mean`。
4. 调 `DiffusionGRPOLossFn` 得 scalar loss + metrics。
5. backward + 梯度累积。
6. clip grad norm → optimizer step。

## 9. 单卡 smoke 规格（3080 Ti 16 GB）

| 项目 | 值 |
|---|---|
| 模型 | `tiny-random/Qwen-Image`（HF 上 2025-08-05 发布的 2 层小型版） |
| 分辨率 | 128×128 图像 / 16×16 latent |
| T（denoising 步数） | 8 |
| SDE 窗口 | `[0, 8)` 全开 |
| K | 2 |
| B（每步 prompts） | 1 |
| LoRA | rank=4, alpha=8, targets `to_q, to_k, to_v` |
| 精度 | bf16 模型权重，fp32 SDE math（`sde.py` 强制） |
| Reward | `DummyImageReward`（CPU） |
| 预估显存 | < 4 GB |

如果 HF 上的 `tiny-random/Qwen-Image` 缺少完整 pipeline 文件，pipeline.py 提供 fallback：用
`QwenImageTransformer2DModel(num_layers=2, num_attention_heads=2, attention_head_dim=16,
in_channels=16, out_channels=16)` 手工实例化并 cache 到 `~/.cache/nemo-rl/tiny-qwen-image/`。

## 10. 与 verl-omni 的对照与基线比对方案

verl-omni 在 `examples/flowgrpo_trainer/`、`verl_omni/trainer/diffusion/`、
`verl_omni/pipelines/qwen_image_flow_grpo/` 提供了完整的 Flow-GRPO 实现。

| 关注点 | verl-omni | NeMo-RL (本设计) |
|---|---|---|
| Advantage 估计器 | `algorithm.adv_estimator=flow_grpo` | `calculate_baseline_and_std_per_prompt`（公式一致） |
| Loss 名 | `actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo` | `DiffusionGRPOLossFn` |
| Loss 公式 | `FlowGRPOLoss.compute_loss` | 同 |
| 关键 hyper | `clip_ratio`, `adv_clip_max`, `use_kl_loss`, `kl_loss_coef`, `noise_level`, `sde_type`, `sde_window_size`, `sde_window_range`, `num_inference_steps`, `true_cfg_scale` | 同（YAML 字段名）|
| Rollout 后端 | `vllm_omni` | DiffusionPolicyWorker 自身（不引入 vLLM-omni） |
| Pipeline 包装 | `pipelines/qwen_image_flow_grpo/diffusers_training_adapter.py`、`vllm_omni_rollout_adapter.py` | `pipeline.py` 单文件复用 `_for_each_window_step` |
| Reward | `verl_omni/workers/reward_manager/` | `ImageRewardEnvironment` |
| 主循环 | `verl_omni/trainer/diffusion/ray_diffusion_trainer.py` | `nemo_rl/algorithms/diffusion_grpo.py` |
| 配置入口 | `verl_omni/trainer/config/diffusion_trainer.yaml` | `examples/configs/diffusion_grpo_qwen_image*.yaml` |

基线对照运行：

1. 在本地 `/home/ubuntu/workspace/NVIDIA/RL/verl-omni` 修改 `examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh`：把 `actor_rollout_ref.model.path` 换成
   `tiny-random/Qwen-Image`、`num_inference_steps=8`、`rollout.n=2`、`data.train_batch_size=1`、
   用 dummy reward 或最小 reward。`conda activate torch && bash <script>`。
2. 收集每步指标到 `results/baselines/verl_omni_flowgrpo.tsv`：`step / policy_loss / mean_ratio /
   clipfrac / approx_kl / kl_loss / reward_mean`。
3. NeMo-RL 跑 `uv run python examples/run_diffusion_grpo.py --config-name
   diffusion_grpo_qwen_image_tiny.yaml`，指标记到 `results/diffusion_grpo/nemo_rl.tsv`。
4. **期望对齐口径**：trend 一致（`mean_ratio ≈ 1.0 ± clip_ratio`、`policy_loss` 同号且同量级、
   `kl_loss` 接近 0 或随 step 二次型增长）；**不要求** 数值绝对相等（模型权重 / seed / 批序不同）。

## 11. 测试

### 单元测试

- `tests/unit/models/diffusion/test_sde.py`（已有 scaffold）：
  - `test_sde_recompute_matches_sampling_fp32`：sample 一次后用 `prev_sample` recompute；
    `max|Δlogprob| < 1e-5`。
  - `test_sde_window_mask`：`compute_window_mask` 形状与边界正确。
- `tests/unit/algorithms/loss/test_diffusion_grpo_loss.py`：
  - 零 advantage → 零 policy_loss；
  - clipped 分支按公式触发；
  - KL 二次性（`reference_mean` 偏移 δ → kl_loss = δ² / (2·std²)）；
  - mask 归一化覆盖率正确。
- `tests/unit/environments/test_image_reward_environment.py`：dummy reward 确定性 + 加权聚合。
- `tests/unit/data/test_text_to_image_prompt_dataset.py`：txt / jsonl 解析、缺省 negative_prompt。
- `tests/unit/models/diffusion/test_qwen_image_pipeline_adapter.py`：极小 transformer 上 4 步 sample
  后 recompute，fp32 < 1e-4、bf16 < 1e-2（issue [#793](https://github.com/NVIDIA-NeMo/RL/issues/793)
  的回归测试）。

### 集成测试

- `tests/functional/diffusion/test_diffusion_grpo_smoke.py`：5 步训练，断言 loss 有限、LoRA
  state_dict 在 checkpoint 中非全零、reward_mean 在合理区间。`@pytest.mark.nightly` 与
  `@pytest.mark.timeout(900)`（遵循 `testing` 技能）。
- `tests/functional/diffusion/test_logprob_parity.py`：单 worker 内 sample → recompute parity（fp32
  < 1e-4、bf16 < 1e-2）。

## 12. 风险与缓解

1. **FSDP2 + PEFT LoRA + diffusers transformer 兼容性**。单卡时 FSDP2 退化为无操作；先用
   plain `requires_grad_` + AdamW，多卡再启 FSDP2。worker 初始化加 assert：trainable params
   > 0 且占比 ≤ 1%（LoRA 模式）。
2. **logprob fp32/bf16 数值漂移**（[#793](https://github.com/NVIDIA-NeMo/RL/issues/793)）。
   `sde.py` 已强制 fp32 数学；用 `test_logprob_parity.py` 守门，fp32 < 1e-4 强制、bf16 < 1e-2 警告。
3. **diffusers 0.38.0.dev0 API 漂移**。本地 conda `torch` 环境是 editable 安装。在 `pipeline.py`
   顶部注释固定接口契约（`scheduler.index_for_timestep / .sigmas / .timesteps`），并加
   `test_flow_match_scheduler_contract` 单测。
4. **PEFT 未必装在 conda torch env**。worker 启动前 try-import；缺失则 fallback 到 tiny 模型
   全量训练并 warning。
5. **reward pool 占显存**。smoke 默认 `_RewardWorker` CPU placement；生产配置允许 `num_gpus=0.25`。
6. **tiny-random/Qwen-Image 接口不全**。若 from_pretrained 失败，pipeline.py fallback 用
   `QwenImageTransformer2DModel` 手工实例化并 cache。

## 13. 实施顺序与依赖

```text
S0 设计文档（本文件）          ── 对齐源
S1 configs + window helper      ── 几十行
S2 loss + tests              ─┐
S3 dataset + tests           ─┼─ 互不依赖，可并行
S4 reward env + tests        ─┘
S5 pipeline adapter + parity test
S6 diffusion worker
S7 diffusion policy
S8 main loop (diffusion_grpo.py)
S9 config + entrypoint
S10 smoke + parity（functional）
S11 docs（含 docs/index 注册、英文 quickstart）
```

S2 / S3 / S4 在 S1 完成后可并行；S5 与 S2 / S3 / S4 也无强依赖；S5 仅依赖 S1 的 window helper。
S6 起强串行。每 commit 控制 < 500 LOC，方便 review。所有 commit 在分支 `diffusion/sde-algo`
上递进，最终统一开 PR。

## 14. 范围之外（明确不做）

- 不改既有 token GRPO 路径（`grpo.py / lm_policy.py / dtensor_policy_worker.py /
  environments/interfaces.py`）。
- 不在 diffusion 路径上接入 vLLM / SGLang / Megatron。
- 不实现 PickScore / ImageReward 的完整加载（仅留 skeleton + 配置开关）。
- 不做 image-edit / video / 多 reward async 流水（后续 PR）。
- 不做 Megatron-Core 后端的 diffusion 适配（首期仅 dtensor / FSDP2 风格）。

## 15. 参考

- NeMo-RL [logprob mismatch issue #793](https://github.com/NVIDIA-NeMo/RL/issues/793)
- NeMo-RL [LLaDA PR #878](https://github.com/NVIDIA-NeMo/RL/pull/878)
- NeMo-RL [Audio GRPO docs](https://docs.nvidia.com/nemo/rl/nightly/guides/grpo-audio.html)
- NeMo-AutoModel [diffusion fine-tuning docs](https://docs.nvidia.com/nemo/automodel/nightly/guides/diffusion/finetune.html)
- Flow-GRPO 论文 ([arXiv:2505.05470](https://arxiv.org/abs/2505.05470))
- [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo) 上游代码
- verl-omni 本地路径：`/home/ubuntu/workspace/NVIDIA/RL/verl-omni`，关键文件：
  `verl_omni/trainer/diffusion/diffusion_algos.py`、`ray_diffusion_trainer.py`、
  `pipelines/qwen_image_flow_grpo/`、`examples/flowgrpo_trainer/`、`docs/algo/flowgrpo.md`。
