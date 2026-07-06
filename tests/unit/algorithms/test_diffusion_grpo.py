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
"""Tests for diffusion-GRPO resume helpers."""

import os

from nemo_rl.algorithms.diffusion_grpo import _latest_checkpoint


def _make_ckpt(root, name: str, complete: bool = True) -> str:
    path = os.path.join(root, name)
    os.makedirs(path)
    if complete:
        with open(os.path.join(path, "optimizer.pt"), "wb"):
            pass
    return path


def test_latest_checkpoint_missing_dir_returns_none(tmp_path):
    assert _latest_checkpoint(str(tmp_path / "does_not_exist")) is None


def test_latest_checkpoint_picks_highest_complete_step(tmp_path):
    _make_ckpt(tmp_path, "step_1")
    expected = _make_ckpt(tmp_path, "step_10")
    # Incomplete checkpoint (no optimizer.pt yet) must be skipped even
    # though its step number is the highest.
    _make_ckpt(tmp_path, "step_30", complete=False)
    _make_ckpt(tmp_path, "not_a_checkpoint")
    assert _latest_checkpoint(str(tmp_path)) == (expected, 10)


def test_latest_checkpoint_all_incomplete_returns_none(tmp_path):
    _make_ckpt(tmp_path, "step_5", complete=False)
    assert _latest_checkpoint(str(tmp_path)) is None
