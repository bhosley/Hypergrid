from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import random
import argparse
import json
import os

import torch
import torch.nn as nn
from pathlib import Path
from typing import Callable

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls

import hypergrid.rllib as HRL

HRL


def get_policy_mapping_fn(
    checkpoint_dir: Path | str | None,
    num_agents: int,
) -> Callable:
    try:
        policies = sorted(
            [
                path
                for path in (checkpoint_dir / "policies").iterdir()
                if path.is_dir()
            ]
        )

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return policies[agent_id % len(policies)].name

        print("Loading policies from:", checkpoint_dir)
        for agent_id in range(num_agents):
            print(
                "Agent ID:", agent_id, "Policy ID:", policy_mapping_fn(agent_id)
            )

        return policy_mapping_fn

    except Exception:
        return lambda agent_id, *args, **kwargs: f"policy_{agent_id}"


def find_checkpoint_dir(search_dir: Path | str | None) -> Path | None:
    try:
        checkpoints = (
            Path(search_dir).expanduser().glob("**/rllib_checkpoint.json")
        )
        if checkpoints:
            return sorted(checkpoints, key=os.path.getmtime)[-1].parent
    except Exception:
        return None


# ---  --- #


from ray.rllib.core.rl_module.torch import TorchRLModule

num_missions = 4


class CustomTorchModule(TorchRLModule):
    def setup(self) -> None:
        # Manually build sub-nets for each part of the Dict:
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
        )
        self.ori_encoder = nn.Sequential(
            nn.Linear(self.observation_space["direction"].nvec.sum(), 16),
            nn.ReLU(),
        )
        self.mis_encoder = nn.Embedding(num_missions, 8)

        # Define a fusion trunk and heads:
        self.trunk = nn.Sequential(
            nn.Linear(128 + 16 + 8, 64),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(64, self.action_space.n)
        self.value_head = nn.Linear(64, 1)

    def _forward_train(self, batch: dict, **kwargs):
        img = batch["obs"]["image"].float() / 255.0
        ori = batch["obs"]["direction"].float()
        miss = batch["obs"]["mission"].long()
        z_img = self.img_encoder(img)
        z_ori = self.ori_encoder(ori)
        z_mis = self.mis_encoder(miss)
        z = torch.cat([z_img, z_ori, z_mis], dim=1)
        logits = self.pi_head(self.trunk(z))
        self._value_out = self.value_head(self.trunk(z)).squeeze(1)
        return logits, [], {}


def get_algorithm_config(
    algo: str = "PPO",
    env: str = "HyperGrid-Empty-v0",
    env_config: dict = {},
    num_agents: int = 2,
    num_workers: int = 0,
    num_gpus: int = 0,
    lr: float | None = NotProvided,
    batch_size: int | None = NotProvided,
    # lstm: bool = False, TODO: implement LSTM model
    # centralized_critic: bool = False, TODO: implement centralized critic
    **kwargs,
) -> AlgorithmConfig:
    """
    Return the RL algorithm configuration dictionary.
    """
    config = (
        get_trainable_cls(algo)
        .get_default_config()
        .api_stack(
            enable_env_runner_and_connector_v2=True,
            enable_rl_module_and_learner=True,
        )
        .debugging(seed=random.randint(0, 1000000))
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1,
            num_gpus_per_env_runner=num_gpus
            if torch.cuda.is_available()
            else 0,
        )
        .environment(env=env, env_config={**env_config, "agents": num_agents})
        .framework("torch")
        .multi_agent(
            policies={f"policy_{i}" for i in range(num_agents)},
            policy_mapping_fn=get_policy_mapping_fn(None, num_agents),
            policies_to_train=[f"policy_{i}" for i in range(num_agents)],
        )
        .training(lr=lr, train_batch_size=batch_size)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    f"policy_{i}": RLModuleSpec(module_class=CustomTorchModule)
                    for i in range(num_agents)
                }
            )
        )
    )

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        help="The name of the RLlib-registered algorithm to use.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="HyperGrid-Empty-v0",
        help="HyperGrid environment to use.",
    )
    parser.add_argument(
        "--env-config",
        type=json.loads,
        default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents in environment.",
    )
    parser.add_argument("--lr", type=float, help="Learning rate for training.")
    # parser.add_argument(
    #     '--lstm', action='store_true',
    #     help="Use LSTM model.")
    # parser.add_argument(
    #     '--centralized-critic', action='store_true',
    #     help="Use centralized critic for training.")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of rollout workers."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0, help="Number of GPUs to train on."
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1e7,
        help="Total number of timesteps to train.",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        help="Checkpoint directory for loading pre-trained policies.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="~/ray_results/",
        help="Directory for saving checkpoints, results, and trained policies.",
    )

    args = parser.parse_args()
    config = get_algorithm_config(**vars(args))

    # config.build()
    config.build_algo()
