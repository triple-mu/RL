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
"""Tests for the diffusion worker's NeMo Automodel model-lifecycle stack.

These tests never load a real checkpoint: the Automodel entry points are
monkeypatched (pipeline loading) or exercised on tiny pure-torch modules
(LoRA). They skip cleanly when ``nemo_automodel`` is not installed, mirroring
how the diffusers-dependent tests handle minimal installs.
"""

import pytest
import torch

pytest.importorskip("nemo_automodel")

from nemo_rl.models.diffusion.workers.diffusion_worker import (
    load_diffusion_pipeline,
)


class _FakePipe:
    pass


def test_load_diffusion_pipeline_uses_automodel_loader(monkeypatch):
    from nemo_automodel._diffusers import auto_diffusion_pipeline as adp

    seen: dict = {}

    def fake_from_pretrained(model_name, **kwargs):
        seen["model_name"] = model_name
        seen.update(kwargs)
        return _FakePipe(), {}

    monkeypatch.setattr(
        adp.NeMoAutoDiffusionPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    pipe, managers = load_diffusion_pipeline(
        "Qwen/Qwen-Image", dtype=torch.bfloat16, device=torch.device("cpu")
    )

    assert isinstance(pipe, _FakePipe)
    assert managers == {}
    assert seen["model_name"] == "Qwen/Qwen-Image"
    assert seen["torch_dtype"] is torch.bfloat16
    assert seen["device"] == torch.device("cpu")
    assert seen["load_for_training"] is True
    # Stage one keeps the worker's manual DP all-reduce: no parallel managers.
    assert seen["parallel_scheme"] is None
    assert seen["peft_cfg"] is None


def test_load_diffusion_pipeline_returns_two_tuple(monkeypatch):
    from nemo_automodel._diffusers import auto_diffusion_pipeline as adp

    fake_managers = {"transformer": object()}
    monkeypatch.setattr(
        adp.NeMoAutoDiffusionPipeline,
        "from_pretrained",
        staticmethod(lambda model_name, **kwargs: (_FakePipe(), fake_managers)),
    )

    pipe, managers = load_diffusion_pipeline(
        "Qwen/Qwen-Image", dtype=torch.float32, device=torch.device("cpu")
    )
    assert isinstance(pipe, _FakePipe)
    assert managers is fake_managers


# ---------------------------------------------------------------------------
# LoRA: schema -> PeftConfig mapping and fail-loud injection check
# ---------------------------------------------------------------------------
class _TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.to_q = torch.nn.Linear(4, 4)
        self.attn.to_k = torch.nn.Linear(4, 4)


class _TinyTransformer(torch.nn.Module):
    def __init__(self, num_blocks: int = 2) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            _TinyBlock() for _ in range(num_blocks)
        )


def _lora_cfg_dict(**overrides):
    cfg = {
        "enabled": True,
        "rank": 2,
        "alpha": 4,
        "target_modules": ["*.attn.to_q", "*.attn.to_k"],
        "dropout": 0.0,
        "exclude_modules": None,
    }
    cfg.update(overrides)
    return cfg


def test_build_peft_config_maps_schema_fields():
    from nemo_rl.models.diffusion.workers.diffusion_worker import build_peft_config

    peft_cfg = build_peft_config(_lora_cfg_dict(rank=64, alpha=128))
    assert peft_cfg.dim == 64
    assert peft_cfg.alpha == 128
    assert peft_cfg.target_modules == ["*.attn.to_q", "*.attn.to_k"]
    assert peft_cfg.exclude_modules == []
    assert peft_cfg.dropout == 0.0
    # The loader pins LoRA weights to the base dtype (bf16); leave it unset here.
    assert peft_cfg.lora_dtype is None


def test_full_path_targets_inject_and_pass_fail_loud_check():
    from nemo_automodel.components._peft.lora import (
        LinearLoRA,
        apply_lora_to_linear_modules,
    )

    from nemo_rl.models.diffusion.workers.diffusion_worker import (
        assert_lora_targets_hit,
        build_peft_config,
    )

    model = _TinyTransformer(num_blocks=3)
    hits = apply_lora_to_linear_modules(model, build_peft_config(_lora_cfg_dict()))
    assert hits == 6  # 3 blocks x {to_q, to_k}
    assert (
        sum(isinstance(m, LinearLoRA) for m in model.modules()) == 6
    )
    assert_lora_targets_hit(model, ["*.attn.to_q", "*.attn.to_k"])


def test_bare_suffix_targets_match_nothing_and_fail_loud():
    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules

    from nemo_rl.models.diffusion.workers.diffusion_worker import (
        assert_lora_targets_hit,
        build_peft_config,
    )

    model = _TinyTransformer()
    # peft-style bare suffixes: ModuleMatcher anchors the whole FQN, so this
    # silently injects nothing — exactly the failure the check must catch.
    hits = apply_lora_to_linear_modules(
        model, build_peft_config(_lora_cfg_dict(target_modules=["to_q", "to_k"]))
    )
    assert hits == 0
    with pytest.raises(RuntimeError, match="full-path"):
        assert_lora_targets_hit(model, ["to_q", "to_k"])


def test_partially_missed_targets_fail_loud_and_name_offenders():
    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules

    from nemo_rl.models.diffusion.workers.diffusion_worker import (
        assert_lora_targets_hit,
        build_peft_config,
    )

    model = _TinyTransformer()
    targets = ["*.attn.to_q", "*.attn.to_v"]  # to_v does not exist
    apply_lora_to_linear_modules(
        model, build_peft_config(_lora_cfg_dict(target_modules=targets))
    )
    with pytest.raises(RuntimeError, match=r"to_v"):
        assert_lora_targets_hit(model, targets)


def test_lora_schema_defaults_are_full_path_patterns():
    from nemo_rl.models.diffusion.interfaces import DiffusionLoraCfg

    cfg = DiffusionLoraCfg()
    assert cfg.target_modules, "defaults must not be empty"
    assert all(t.startswith("*.") for t in cfg.target_modules)


def test_exemplar_yaml_targets_are_full_path_patterns():
    yaml = pytest.importorskip("yaml")

    with open("examples/configs/diffusion_grpo_qwen_image_ocr.yaml") as f:
        cfg = yaml.safe_load(f)
    targets = cfg["policy"]["lora_cfg"]["target_modules"]
    assert len(targets) == 12
    assert all(t.startswith("*.") for t in targets)
