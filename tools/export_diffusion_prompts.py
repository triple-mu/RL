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
"""Export text-to-image prompts from a HF dataset to train/val jsonl files.

Produces the `{"prompt": ...}` jsonl format consumed by
`nemo_rl.data.datasets.text_to_image_prompt.TextToImagePromptDataset`.
Prompts are deduplicated, length-filtered, shuffled with a fixed seed, and
split into disjoint train/val files.

Example (pick-a-pic v2 prompts for PickScore reward training):

    uv run python tools/export_diffusion_prompts.py \
        --dataset yuvalkirstain/pickapic_v2_no_images --split train \
        --column caption --train-size 4000 --val-size 64 \
        --out-dir /data/datasets/qwen_image_grpo
"""

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HF dataset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--column", default="caption", help="prompt column name")
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=200_000,
        help="max rows to stream before shuffling/splitting",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    seen: set[str] = set()
    prompts: list[str] = []
    need = args.train_size + args.val_size
    for i, row in enumerate(ds):
        if i >= args.scan_limit or len(prompts) >= args.scan_limit:
            break
        text = (row.get(args.column) or "").strip()
        if not (args.min_chars <= len(text) <= args.max_chars):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)
        # Keep scanning past `need` so the fixed-seed shuffle draws from a
        # wider pool, but stop once the pool is 5x oversampled.
        if len(prompts) >= need * 5:
            break

    if len(prompts) < need:
        raise SystemExit(
            f"only found {len(prompts)} unique prompts; need {need}. "
            "Increase --scan-limit or relax the length filters."
        )

    random.Random(args.seed).shuffle(prompts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train_prompts.jsonl": prompts[: args.train_size],
        "val_prompts.jsonl": prompts[args.train_size : need],
    }
    for fname, split_prompts in splits.items():
        path = out_dir / fname
        with path.open("w", encoding="utf-8") as f:
            for p in split_prompts:
                f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
        print(f"wrote {len(split_prompts)} prompts -> {path}")


if __name__ == "__main__":
    main()
