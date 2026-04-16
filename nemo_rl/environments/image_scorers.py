# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Lightweight image reward scorers used by diffusion GRPO."""

import torch
from transformers import CLIPModel, CLIPProcessor


class PickScoreScorer(torch.nn.Module):
    """PickScore reward model for prompt-image alignment.

    Adapted from the upstream Flow-GRPO scorer so we can keep NeMo-RL's
    newer diffusion stack without inheriting Flow-GRPO's older dependency pins.
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        super().__init__()
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"

        self.device = device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def __call__(self, prompts, images) -> torch.Tensor:
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        image_outputs = self.model.vision_model(**image_inputs)
        image_embs = self.model.visual_projection(image_outputs.pooler_output)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_outputs = self.model.text_model(**text_inputs)
        text_embs = self.model.text_projection(text_outputs.pooler_output)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()

        # Match the upstream normalization used by Flow-GRPO.
        return scores / 26
