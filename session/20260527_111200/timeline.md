# Timeline

## 2026-05-27 11:14
- Started session for diffusion-GRPO implementation
- Goal: Implement diffusion-GRPO in NeMo-RL + compare against verl-omni baseline
- S0-S11 completed: 39 unit tests pass

## 2026-05-27 11:50
- NeMo-RL smoke verified on RTX 3080 Ti Laptop GPU
- 5 GRPO steps completed, step_5/adapter_model.safetensors written (7.6 KB)
- Notable: K=2 + leave-one-out baseline → loss = 0 (math property); K=4 → non-zero loss (-0.116, +0.958, -0.038)
- 30 dependencies fixed iteratively: fp32/bf16 dtype boundary (transformer + VAE), Ray pickle (deferred torch import), Ray sharding_annotations bypass, CPU image handoff to reward, Logger stdout fallback

## 2026-05-27 13:50
- User asked: have you also run verl-omni for comparison? Honest answer: no.
- Inspected verl-omni install requirements: vllm 0.20.2 + vllm-omni (git build) + verl + verl-omni. Decided Docker route.
- No published verl-omni image. Found verlai/verl:vllm020.dev1 (11 GB, parent project) as the closest base.

## 2026-05-27 13:55 — Docker migration
- /var/lib/docker on / (7.5 GB free), needed 11 GB pull
- User granted NOPASSWD sudo + authorized stopping cti container + /data switch
- Stopped docker → rsync /var/lib/docker (only 2.6 MB actually copied — du report was inflated) → edited daemon.json data-root=/data/docker → restarted docker
- Old images (lmsysorg/sglang, nvcr.io/nvidia/pytorch *) not visible in new root. Pre-existing /data/docker had paddlecloud images (45+71 GB).
- cti container lost (was based on nvcr.io/nvidia/pytorch:23.04). User accepted.

## 2026-05-27 13:56 — Docker pull
- Started: docker pull verlai/verl:vllm020.dev1 (background task bfbq3ydgd)
- Progress: overlay2 79→93 GB (~14 GB layers pulled, image not yet registered)
- Drafted scripts/run_verlomni_1gpu_smoke.sh with matched parameters to NeMo-RL smoke:
  - K=2, T=8, h=w=128, true_cfg_scale=1.0, noise_level=0.7
  - LoRA rank=4 alpha=8 targets [to_q,to_k,to_v]
  - lr=1e-4 weight_decay=0, no KL, 5 train steps
  - reward: rule-based jpeg_compressibility (verl-omni's only zero-config rule reward)
- Wait condition set up: until docker images verlai/verl --format {{.ID}} non-empty (background task bfbsafb3t)
