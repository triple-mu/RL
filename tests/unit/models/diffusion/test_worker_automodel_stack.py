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
