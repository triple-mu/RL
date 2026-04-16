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

"""Entry point for Diffusion GRPO training (e.g., Qwen-Image with PickScore rewards).

Usage:
    python examples/run_diffusion_grpo.py --config examples/configs/diffusion_grpo_qwen_image.yaml
    python examples/run_diffusion_grpo.py --config examples/configs/diffusion_grpo_qwen_image.yaml \
        diffusion_grpo.num_images_per_prompt=8 loss_fn.ratio_clip_min=0.1
"""

import argparse
import os
import pprint

from omegaconf import OmegaConf

from nemo_rl.algorithms.diffusion_grpo import (
    DiffusionMasterConfig,
    diffusion_grpo_train,
    setup,
)
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run Diffusion GRPO training with configuration"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "diffusion_grpo_qwen_image.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: DiffusionMasterConfig = OmegaConf.to_container(config, resolve=True)

    print("Final config:")
    pprint.pprint(config)

    # Set up log directory
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")

    # Setup all components
    policy, dataloader, loss_fn, logger, save_state = setup(config)

    print("Running Diffusion GRPO training")
    diffusion_grpo_train(
        policy=policy,
        dataloader=dataloader,
        loss_fn=loss_fn,
        logger=logger,
        save_state=save_state,
        config=config,
    )


if __name__ == "__main__":
    main()
