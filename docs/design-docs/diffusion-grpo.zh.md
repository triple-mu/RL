# NeMo-RL Diffusion-GRPO 设计文档（中文）

本文档说明 NeMo-RL 中 `diffusion_grpo` 算法路径的设计，首期支持
[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) 文生图模型的 Flow-GRPO 训练
（[paper](https://arxiv.org/abs/2505.05470)、[code](https://github.com/yifan123/flow_grpo)）。
工程上沿用 NeMo-RL 的 `BatchedDataDict` + `RayWorkerGroup` 风格，且不改动既有 token GRPO 路径。

## 1. 动机与目标

NeMo-RL 现有 GRPO 围绕 token 输出设计：`message_log` → vLLM/SGLang/Megatron 生成 →
token logprob → `ClippedPGLossFn`。该路径不适合 diffusion 图像/视频输出：
action 是 latent transition `x_t → x_{t+1}`，logprob 是 SDE Gaussian 密度，reward 输入是
image tensor，且不需要 generation backend 与 policy 之间的权重 refit。

目标：

- Qwen-Image 文生图 GRPO 训练，覆盖 SDE flow-matching rollout、per-timestep logprob、
  per-prompt group baseline。
- LoRA 训练为默认，full-parameter 通过配置开启。
- 可选 reference transformer KL；KL 系数为 0 时不加载 reference。
- 可插拔 image reward：内置 `DummyImageReward`（确定性，供测试）、
  `JpegCompressibilityReward`（规则型）、`PickScoreReward`（偏好模型）；
  其它 reward 通过注册接口扩展。
- 单节点数据并行（DP）：按 prompt 分片 rollout、梯度 all-reduce，样本吞吐随卡数线性扩展。
- 训练侧 micro-batch 梯度累积，使带梯度的 T 步 logprob 重算的显存与全局 batch 解耦。
- 最小断点恢复：从 checkpoint 目录最新完整 `step_N` 自动续跑。
- 提供 tiny 模型的单卡 smoke 路径，用于快速回归与 CI。
- **不改动** 既有 token GRPO 路径（`grpo.py / lm_policy.py / dtensor_policy_worker.py /
  environments/interfaces.py`）。

非目标：critic / value model（GRPO 范式不需要）；vLLM/SGLang 生成后端用于 diffusion；
image-edit / video（后续 PR）；Megatron-Core 后端的 diffusion 适配（首期仅 dtensor/FSDP2 风格）。

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

因此新增独立算法路径 `diffusion_grpo` 比把分支逻辑塞进 `nemo_rl/algorithms/grpo.py` 更清晰，
也更容易和 NeMo-RL 既有 token GRPO 解耦演进。

## 3. 总体架构

```text
nemo_rl/algorithms/diffusion_grpo.py
  │
  ├── TextToImagePromptDataset
  │      → prompt / negative_prompt / metadata
  │
  ├── DiffusionPolicy（controller 侧 façade，负责 DP scatter/gather）
  │      → RayWorkerGroup（N 个 worker = N 卡数据并行）
  │         → DiffusionPolicyWorker（每卡一个，NCCL 进程组成员）
  │            → QwenImagePipelineAdapter
  │               ├── trainable transformer (DiT)
  │               ├── frozen text_encoder
  │               ├── frozen vae
  │               ├── optional frozen transformer_ref（LoRA 模式：禁 adapter 复用 base）
  │               └── FlowMatch scheduler + SDE logprob kernel
  │
  ├── ImageRewardEnvironment
  │      → _RewardWorker pool
  │         → DummyImageReward / JpegCompressibilityReward / PickScoreReward
  │
  ├── GRPO advantage：复用 calculate_baseline_and_std_per_prompt（全局计算）
  ├── DiffusionGRPOLossFn
  └── Logger / Timer / checkpoint（rank 0 落盘 + 最新 step 自动恢复）
```

采样和训练共享同一组 diffusion policy workers，不引入 vLLM / SGLang，也不需要
policy-to-generation refit。

## 4. Actor、Reward、Critic

### Actor

可训练 diffusion transformer / DiT。Qwen-Image 对应 `QwenImageTransformer2DModel`。
LoRA 为默认，full-parameter 通过 `lora_cfg.enabled=false` 启用。

### 冻结组件

- `text_encoder`：负责 prompt encoding，无梯度。
- `vae`：负责 latent decode，无梯度。
- `scheduler`：flow-matching timestep 与 sigma；状态量，不更新。
- `transformer_ref`：可选；仅在 `loss_fn.beta > 0` 时启用。LoRA 模式下通过禁用 PEFT adapter
  复用 base 模型权重，无额外显存开销；full-parameter 模式下保留独立冻结副本。

未来扩展：image editing 增加 reference image encoder；video diffusion 增加 temporal latent
相关处理。

### Reward

独立 Ray reward worker pool。输入 `(images, prompts, metadata)`，输出
`(weighted_total_reward, component_metrics)`。内置插件：

- `DummyImageReward`：确定性 reward（prompt 哈希 + image 均值），用于 smoke 与单测；
- `JpegCompressibilityReward`：`-jpeg_kb/500`，规则型压缩率 reward，零外部依赖；
- `PickScoreReward`：PickScore_v1（CLIP-H 偏好模型）。走整模型 forward 取
  `logits_per_image` 对角线（同时兼容 transformers 4.x/5.x —— 5.x 的
  `get_*_features` 返回 `ModelOutput` 而非张量）；重型加载延迟到 Ray actor 内。
- 其它 reward（ImageReward / HPS 等）通过 `register_image_reward` 插件接口扩展。

图像张量约定：`NCHW`、float、值域 `[0, 1]`，跨 Ray 边界时驻留 CPU；GPU 打分插件自行搬运。

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

`negative_prompt` 默认 `" "`，对齐 Qwen-Image 的推荐用法。

### DiffusionTrajectorySpec（rollout 产物）

```python
class DiffusionTrajectorySpec(TypedDict):
    prompts: list[str]
    negative_prompts: list[str]
    metadata: list[dict[str, Any]]
    images: torch.Tensor                    # [B*K, 3, H, W]
    latents: torch.Tensor                   # [B*K, T+1, C, H', W']
    timesteps: torch.Tensor                 # [B*K, T]
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
    generation_logprobs: torch.Tensor       # 旧 policy logprob
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

主训练入口 `diffusion_grpo_train(...)` 与配置根 `DiffusionMasterConfig`。相位：

```text
batch prompts
   ├─ repeat K → B*K
   ▼ policy.sample_trajectory        （DP：按 prompt scatter，gather 后拼接）
collect trajectory + generation_logprobs
   ▼ env.score_images                （images 先搬 CPU 再过 Ray）
rewards [B*K]
   ▼ calculate_baseline_and_std_per_prompt + advantage 扩到 timestep（全局）
train_data
   ▼ policy.train                    （DP：同序 scatter，worker 内梯度 all-reduce）
loss / metrics（含 DP 权重一致性监控）
   ▼ log / validate / checkpoint（rank 0）
```

### 配置 schema

配置类（`DiffusionPolicyConfig`、`DiffusionGRPOAlgoConfig`、`DiffusionLossConfig`、
`DiffusionPipelineCfg`、`DiffusionAlgoCfg`、`DiffusionLoraCfg`）采用 pydantic
`BaseModel(extra="allow")`（config-conventions v2：默认值集中在 BaseModel 字段上，
exemplar YAML 作为文档）。入口以 `DiffusionMasterConfig.model_validate` 校验后用
`model_dump()` dict 跨 Ray 边界传输。

### `nemo_rl/models/diffusion/sde.py`

SDE 步进的数学核心：

- `sde_step_with_logprob`：给定模型输出与（可选的）`prev_sample`，计算下一步 latent 与
  transition 的 Gaussian log-density。同一函数同时服务 rollout（采样新 latent）与训练
  （对 rollout 轨迹重算 logprob），保证两侧数值一致。SDE 数学强制 fp32。
- `compute_window_mask(num_steps, window_start, window_size)`：SDE 随机窗口 mask，供
  pipeline 与 loss 共享，避免重复定义 window 语义。

### `nemo_rl/models/diffusion/pipeline.py`

`QwenImagePipelineAdapter`：把 diffusers 加载的 Qwen-Image 组件接入 GRPO 的
rollout / logprob-recompute 双路径。

公共方法：

```python
def encode_condition(prompts, negative_prompts) -> dict[str, Tensor]
def sample_trajectory(prompts, negative_prompts, metadata, *, K, seed) -> DiffusionTrajectorySpec
def compute_transition_logprob(data, *, use_reference) -> tuple[Tensor, ...]
def decode(latents) -> Tensor
```

要点：

- prompt 与 negative prompt 一起 encode 后再 split（CFG concat 路径）。
- 使用 Qwen-Image 的 flow-matching `mu` shift。
- true classifier-free guidance + norm rescale。
- **采样与训练 logprob recompute 共享同一段 denoise-step helper**，避免 rollout-time 与
  train-time 数值分叉。
- SDE window 外的步骤走确定性 mean，且 `generation_logprobs[:, t] = 0`、`timestep_mask[:, t] = 0`。
- **随机性设计**：初始 latent 与 SDE 注入噪声必须使用不同派生 seed 的 generator
  （实现为 `seed` 与 `seed + 1`）。两个同 seed 的新 generator 产生相同的随机流，
  若共用，step-0 的注入噪声会是初始 latent 的逐元素重排，破坏 transition logprob
  的独立高斯假设。

### `nemo_rl/models/diffusion/workers/diffusion_worker.py`

`@ray.remote class DiffusionPolicyWorker`。**镜像** `dtensor_policy_worker.py` 的结构
（`configure_worker / __init__ / prepare_for_generation / prepare_for_training /
save_checkpoint / shutdown`），但 **不继承** —— 后者绑定 HF causal-LM `forward`。

方法：

```python
def sample_trajectory(prompts, negative_prompts, metadata, *, K, seed) -> DiffusionTrajectorySpec
def compute_transition_logprob(data, *, use_reference) -> dict[str, Tensor]
def train_step(data, loss_cfg) -> dict[str, float]
def save_checkpoint(path) -> None          # 仅 rank 0 写
def load_checkpoint(path) -> None          # 断点恢复
def report_trainable_checksum() -> float   # DP 一致性监控
```

初始化步骤：

1. 从 `RayWorkerGroup` 注入的 `RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT` 环境变量
   读取分布式身份，`init_process_group("nccl")` 组成 DP 进程组。
2. `QwenImagePipeline.from_pretrained(...)`，拆出 `transformer / text_encoder / vae / scheduler`。
3. 冻结 `text_encoder` 与 `vae`；按配置开启 transformer gradient checkpointing。
4. LoRA 经 PEFT 注入到 transformer；或 full-parameter。多卡时 **必须提供统一 seed**，
   保证各 rank 的 LoRA 随机初始化逐位一致（缺失时直接报错，避免静默发散）。
5. Reference 实现：LoRA 模式 = base 模型禁用 adapter；full-param 模式 = 独立冻结副本。
6. 构造 `QwenImagePipelineAdapter`；AdamW 仅训练 trainable params。

`train_step` 的 micro-batch 梯度累积：按 `train_micro_batch_size` 沿样本维切 chunk，
逐 chunk 做带梯度的 T 步 logprob 重算 + loss backward（按 chunk 样本数加权），
显存峰值只与 chunk 大小相关；全部 chunk 累积后（DP 下先对 trainable 梯度
all-reduce 取均值）统一 clip grad norm 并 `optimizer.step()`。

### `nemo_rl/models/diffusion/policy.py`

`DiffusionPolicy` 是 controller 侧 façade，同时承担 DP 的数据编排：

- 使用 `RayVirtualCluster` / `RayWorkerBuilder` / `RayWorkerGroup`；worker 数 =
  `cluster.gpus_per_node × cluster.num_nodes`。
- `sample_trajectory`：把 prompt batch 均分 scatter 给各 worker（每 worker 派生独立
  seed，保证初始噪声跨 rank 去相关且可复现），gather 后沿 batch 维拼接；不同 worker
  的 prompt embedding 序列长度可能不同，拼接时右 pad 到最大长度（对应 mask 置 0）。
  prompt 数不能整除 worker 数时回退到单 worker 并告警（K=1 的验证路径属于合法回退）。
- `train`：训练数据按与 rollout 相同的连续切分 scatter，每个 worker 训练自己生成的
  样本；样本数不能整除 worker 数时直接报错（入口另有
  `num_prompts_per_step % 卡数 == 0` 的启动校验）。
- `trainable_checksums`：收集各 rank trainable 参数校验和；训练循环把
  `max - min` 记为 `train/dp_checksum_spread` 指标，恒为 0 才说明梯度同步正确。
- 不接入 vLLM / SGLang refit。

### `nemo_rl/algorithms/loss/diffusion_grpo.py`

`DiffusionGRPOLossFn`。**不实现** `nemo_rl/algorithms/loss/interfaces.py:LossFunction`
Protocol，其 `__call__(data, global_valid_seqs, global_valid_toks, **kwargs)` 与 token 语义
绑定（next_token_logprobs 或 logits + `LossType.{TOKEN,SEQUENCE}_LEVEL`），不适配
diffusion timestep 概念。把 diffusion loss 放进同一接口要么让接口变臃肿，要么强行映射
会让 metrics 命名错位。

签名：

```python
def __call__(
    self,
    curr_logprob: Tensor,            # [B*K, T]
    generation_logprob: Tensor,      # [B*K, T]，旧 policy logprob
    advantages: Tensor,              # [B*K, T]
    timestep_mask: Tensor,           # [B*K, T]
    sample_mask: Tensor,             # [B*K]，= loss_multiplier
    *,
    current_mean: Tensor | None = None,
    reference_mean: Tensor | None = None,
    std_dev: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]
```

公式（clipped policy-gradient + 可选 Gaussian KL）：

```python
advantages  = advantages.clamp(-adv_clip_max, adv_clip_max)
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

支持 per-element（`[B*K, T, ...]` 逐元素）与 per-sample（沿 T 求和到 `[B*K]`）两种
logprob 聚合粒度。返回 metrics：`policy_loss / kl_loss / approx_kl / clipfrac /
clipfrac_higher / clipfrac_lower / mean_ratio / ratio_min / ratio_max`，按 NeMo-RL 的
`train/` 前缀挂载。

### `nemo_rl/environments/image_reward_environment.py`

image-native reward API：

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

**不继承** `nemo_rl/environments/interfaces.py:EnvironmentInterface`，其
`step(message_log_batch, metadata)` 假设 token-style 输入，与 image reward 不兼容。

每个 plugin 一个 Ray actor；plugin 返回 component dict，环境按 `weight` 加权求和为
total reward，并输出各 component 的 mean 指标。GPU/CPU 放置由
`num_gpus_per_worker / num_cpus_per_worker` 配置。

### Dataset、Config、Entrypoint

- `nemo_rl/data/datasets/text_to_image_prompt.py:TextToImagePromptDataset`：
  - `.txt`：每行一个 prompt；
  - `.jsonl`：`{"prompt": str, "negative_prompt"?: str, "metadata"?: dict}`。
- `tools/export_diffusion_prompts.py`：从 HF 数据集导出去重、过滤、固定 seed 切分的
  train/val prompt jsonl。
- `examples/configs/diffusion_grpo_qwen_image.yaml`：多卡 DP exemplar（含字段文档）。
- `examples/configs/diffusion_grpo_qwen_image_tiny.yaml`：tiny 模型单卡 smoke 配置。
- `examples/configs/recipes/diffusion/`：nightly recipe（继承 exemplar，仅覆盖差异项）。
- `examples/run_diffusion_grpo.py`：配置加载（OmegaConf + Hydra overrides）→
  `DiffusionMasterConfig` 校验 → driver 侧 set_seed（DataLoader shuffle 可复现）→
  组建 cluster/policy/env/dataloader（训练集 `drop_last=True`，避免尾批打破 DP 整除性）
  → `diffusion_grpo_train`。

## 7. 可复用 NeMo-RL 组件

| 组件 | 路径 |
|---|---|
| `RayVirtualCluster` | `nemo_rl/distributed/virtual_cluster.py` |
| `RayWorkerBuilder` / `RayWorkerGroup` | `nemo_rl/distributed/worker_groups.py` |
| `BatchedDataDict` | `nemo_rl/distributed/batched_data_dict.py` |
| `calculate_baseline_and_std_per_prompt` | `nemo_rl/algorithms/utils.py` |
| `Logger` | `nemo_rl/utils/logger.py` |
| `Timer` | `nemo_rl/utils/timer.py` |
| Hydra config 加载与 override helper | `nemo_rl/utils/config.py` |
| dtensor worker 结构模板 | `nemo_rl/models/policy/workers/dtensor_policy_worker.py` |

## 8. 训练流程详解

### Rollout

1. 读取 prompt batch（size = `num_prompts_per_step`）。
2. controller 按 worker 数均分 prompts；每个 worker 拿到独立派生 seed。
3. worker 内每个 prompt 重复 K = `num_generations_per_prompt` 次，编码 prompt 与
   negative prompt，准备初始 latent `x_0 ~ N(0, I)`。
4. 在 SDE window 内：对每个 t，transformer forward → CFG → norm rescale →
   `sde_step_with_logprob`，记录 `(latents[t+1], generation_logprobs[t])`。
   窗外步走确定性 mean，logprob 与 mask 置 0。
5. VAE decode 末步 latent → image tensor。
6. controller gather 各 worker 轨迹并拼接（embedding 序列右 pad 对齐）。

### Reward

1. images 搬到 CPU 后把 `(images, prompts, metadata)` 派给 Ray reward worker pool。
2. 每个 plugin 返回 component dict；按 `weight` 聚合为 total reward。

### Advantage（全局）

1. 同一原始 prompt 的 K 张图共享 prompt id；跨 worker gather 后在 controller 统一计算。
2. 调 `calculate_baseline_and_std_per_prompt(...)`（leave-one-out baseline）。
3. `advantages = (r - baseline) / std.clamp_min(1e-6)`，再 `.unsqueeze(-1).expand(-1, T)`。

### Training

1. 训练数据按 rollout 的同序连续切分发回各 worker。
2. worker 按 `train_micro_batch_size` 切 chunk：当前 transformer 对轨迹重算
   `curr_logprob`（有梯度；可选 reference forward），调 `DiffusionGRPOLossFn`，
   加权 backward 累积。
3. DP 下对 trainable 梯度 all-reduce（均值）→ clip grad norm → optimizer step。
4. 支持 `ppo_epochs > 1`：同一批 rollout 数据重复训练，`generation_logprobs` 固定、
   `curr_logprob` 每轮重算。

### Validation 与 checkpoint

- 验证用 K=1 与固定 seed 采样，保证 `val/reward_mean` 与落盘样本图跨训练步可比；
  `max_val_samples` 限制验证规模，`num_val_samples_to_print` 控制每次验证保存的
  样本图数量（PNG + prompt/reward 清单）。支持 `val_at_start / val_period / val_at_end`。
- checkpoint 仅 rank 0 写（LoRA adapter 或全量 state_dict + optimizer state）。
- 恢复：启动时扫描 `checkpoint_dir` 下最新完整 `step_N` 自动续跑（dataloader 位置
  不恢复，留给 StatefulDataLoader 后续接入）。

## 9. 并行设计

- **数据并行（已实现）**：GRPO 的吞吐瓶颈是每步样本量。N 个 worker 各持完整模型副本，
  rollout 按 prompt 分片、训练按同序样本分片，梯度 all-reduce 后各 rank 应用相同更新。
  正确性不变量：所有 rank 的 trainable 参数逐位一致，以 `train/dp_checksum_spread == 0`
  持续监控。前提：统一的权重初始化 seed（缺失即报错）。
- **micro-batch 梯度累积（已实现）**：训练侧带梯度的 T 步重算是显存大头；按样本维
  chunk 化后，显存峰值与 `train_micro_batch_size` 相关而与全局 batch 无关。
- **FSDP2 参数分片（预留）**：LoRA 模式下无必要（冻结权重 + 极小可训练集）；
  full-parameter 训练需要优化器状态分片时接入，代码结构（worker 独占单卡、
  NCCL 进程组已就位）为其预留。
- **TP / SP / CP（不做）**：单卡可容纳推理与 LoRA 训练时收益为负；高分辨率
  （长 latent 序列）或 full-parameter 场景出现后再评估。

## 10. 测试

### 单元测试

- `tests/unit/models/diffusion/test_sde.py`：
  - sample 一次后用 `prev_sample` recompute，logprob 精确一致（含 `cps` 变体）；
  - `compute_window_mask` 的 full/partial/末端 clamp/非法参数用例；
  - 真实 diffusers `FlowMatchEulerDiscreteScheduler` 满足
    `timesteps/sigmas/index_for_timestep` 契约（防 diffusers API 漂移）。
- `tests/unit/models/diffusion/test_qwen_image_pipeline_adapter.py`：极小 transformer 上
  sample 后 recompute 的 logprob parity（fp32 < 1e-4、bf16 < 1e-2；issue
  [#793](https://github.com/NVIDIA-NeMo/RL/issues/793) 的回归测试）。
- `tests/unit/models/diffusion/test_policy_merge.py`：DP 轨迹 gather 的拼接与
  ragged 序列维 pad 语义。
- `tests/unit/algorithms/loss/test_diffusion_grpo_loss.py`：零 advantage → 零 policy_loss；
  clipped 分支按公式触发；KL 二次性；mask 归一化。
- `tests/unit/algorithms/test_diffusion_grpo.py`：advantage 计算与主循环辅助逻辑。
- `tests/unit/environments/test_image_reward_environment.py`：dummy reward 确定性、
  加权聚合、PickScore 的配对/分块/返回契约（fake CLIP，不下载权重）。
- `tests/unit/data/test_text_to_image_prompt_dataset.py`：txt / jsonl 解析、缺省
  negative_prompt。

### 集成测试

- nightly 三件套：`examples/configs/recipes/diffusion/` recipe +
  `tests/test_suites/diffusion/` driver + `tests/test_suites/nightly.txt` 条目
  （真实模型多卡 DP + PickScore 短程训练，断言 `mean_ratio ≈ 1` 且末步 loss 有界）。
- 单卡 shell smoke：`tests/functional/diffusion_grpo_smoke.sh`（tiny 模型数步训练 +
  checkpoint 产物断言）。
- DP 正确性：多卡 smoke 观察 `train/dp_checksum_spread` 恒为 0。

## 11. 风险与缓解

1. **FSDP2 + PEFT LoRA + diffusers transformer 兼容性**。首期用 plain
   `requires_grad_` + AdamW + DP；接入 FSDP2 前先在 tiny 模型上验证 PEFT 兼容性。
   worker 初始化 assert：trainable params > 0。
2. **logprob fp32/bf16 数值漂移**（[#793](https://github.com/NVIDIA-NeMo/RL/issues/793)）。
   `sde.py` 强制 fp32 数学；单测以 fp32 < 1e-4、bf16 < 1e-2 守门。
3. **diffusers API 漂移**。`pipeline.py` 顶部注释固定接口契约
   （`scheduler.index_for_timestep / .sigmas / .timesteps`），配 scheduler 契约单测。
4. **随机流重叠**。初始 latent 与 SDE 噪声、以及 DP 各 rank 之间都必须使用派生
   seed 隔离随机流（见 §6 pipeline 要点与 §9），否则出现结构性相关噪声或跨 rank
   重复样本，且难以从 loss 曲线察觉。
5. **reward pool 资源放置**。placement group 占满全部 GPU 时，GPU reward worker 会
   无法调度；exemplar 默认 reward 走 CPU，需要 GPU 打分时预留一张卡
   （降低 `cluster.gpus_per_node` 并设 `num_gpus_per_worker: 1`）。
6. **静默配置错误**。DP 相关的整除性、缺失 seed 等在入口/worker 初始化处显式校验
   报错，不允许静默退化。

## 12. 模块依赖与实施划分

```text
S0 设计文档（本文件）          ── 对齐源
S1 configs + window helper
S2 loss + tests              ─┐
S3 dataset + tests           ─┼─ 互不依赖，可并行
S4 reward env + tests        ─┘
S5 pipeline adapter + parity test
S6 diffusion worker
S7 diffusion policy（含 DP scatter/gather）
S8 main loop (diffusion_grpo.py)
S9 config + entrypoint
S10 smoke + nightly recipe
S11 docs（含 docs/index 注册、英文 quickstart）
```

S2 / S3 / S4 在 S1 完成后可并行；S5 仅依赖 S1 的 window helper；S6 起强串行。
每个提交控制在可 review 的规模并按主题拆分。

## 13. 范围之外（明确不做）

- 不改既有 token GRPO 路径（`grpo.py / lm_policy.py / dtensor_policy_worker.py /
  environments/interfaces.py`）。
- 不在 diffusion 路径上接入 vLLM / SGLang / Megatron。
- 不内置 ImageReward / HPS 的加载（走 `register_image_reward` 插件接口）。
- 不做 image-edit / video / 多 reward async 流水（后续 PR）。
- 不做 Megatron-Core 后端的 diffusion 适配（首期仅 dtensor / FSDP2 风格）。

## 14. 参考

- Flow-GRPO 论文（[arXiv:2505.05470](https://arxiv.org/abs/2505.05470)）与
  [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo) 上游实现
- NeMo-RL [logprob mismatch issue #793](https://github.com/NVIDIA-NeMo/RL/issues/793)
- NeMo-RL [LLaDA PR #878](https://github.com/NVIDIA-NeMo/RL/pull/878)
- NeMo-RL [Audio GRPO docs](https://docs.nvidia.com/nemo/rl/nightly/guides/grpo-audio.html)
- NeMo-AutoModel [diffusion fine-tuning docs](https://docs.nvidia.com/nemo/automodel/nightly/guides/diffusion/finetune.html)
