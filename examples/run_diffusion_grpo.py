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
"""Entrypoint for diffusion-GRPO training (Qwen-Image flow-grpo)."""

import argparse
import os
import pprint

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo_rl.algorithms.diffusion_grpo import diffusion_grpo_train
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data.datasets.text_to_image_prompt import (
    TextToImagePromptDataset,
    text_to_image_collate_fn,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.image_reward_environment import ImageRewardEnvironment
from nemo_rl.models.diffusion.policy import DiffusionPolicy
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import Logger, get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run diffusion-GRPO training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: examples/configs/diffusion_grpo_qwen_image_tiny.yaml)",
    )
    return parser.parse_known_args()


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "diffusion_grpo_qwen_image_tiny.yaml",
        )

    cfg = load_config(args.config)
    if overrides:
        cfg = parse_hydra_overrides(cfg, overrides)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print("Final config:")
    pprint.pprint(cfg)

    cfg["logger"]["log_dir"] = get_next_experiment_dir(cfg["logger"]["log_dir"])
    print(f"📊 log_dir: {cfg['logger']['log_dir']}")

    # Seed the driver process too: DataLoader(shuffle=True) draws from the
    # global RNG, so without this the prompt order differs across runs.
    set_seed(int(cfg["grpo"].get("seed", 42)))

    init_ray()

    cluster_cfg = cfg["cluster"]
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[cluster_cfg["gpus_per_node"]]
        * cluster_cfg["num_nodes"],
        use_gpus=True,
        max_colocated_worker_groups=1,
    )

    policy = DiffusionPolicy(cluster=cluster, config=cfg["policy"])

    env_cfg = cfg["env"]["image_reward"]
    env = ImageRewardEnvironment(
        plugin_specs=env_cfg["plugins"],
        num_cpus_per_worker=env_cfg.get("num_cpus_per_worker", 1),
        num_gpus_per_worker=env_cfg.get("num_gpus_per_worker", 0.0),
    )

    train_ds = TextToImagePromptDataset(cfg["data"]["train"]["prompt_file"])
    val_ds = (
        TextToImagePromptDataset(cfg["data"]["val"]["prompt_file"])
        if "val" in cfg["data"]
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["grpo"]["num_prompts_per_step"],
        shuffle=True,
        collate_fn=text_to_image_collate_fn,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg["grpo"]["num_prompts_per_step"],
            shuffle=False,
            collate_fn=text_to_image_collate_fn,
        )
        if val_ds is not None
        else None
    )

    logger = Logger(cfg["logger"])

    try:
        diffusion_grpo_train(
            policy=policy,
            env=env,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            algo_cfg=cfg["grpo"],
            loss_cfg=cfg["loss_fn"],
            policy_cfg=cfg["policy"],
            logger=logger,
            checkpoint_dir=cfg["checkpointing"].get("checkpoint_dir")
            if cfg["checkpointing"].get("enabled")
            else None,
            save_period=int(cfg["checkpointing"].get("save_period", 0)),
            val_image_dir=os.path.join(cfg["logger"]["log_dir"], "val_images"),
            num_val_images_to_save=int(
                cfg["logger"].get("num_val_samples_to_print", 0)
            ),
        )
    finally:
        env.shutdown()
        policy.shutdown()


if __name__ == "__main__":
    main()
