from __future__ import annotations

import argparse
import json
import hypergrid.rllib as HRC
import os
import random
import ray

# import ray.train
# import ray.tune
import torch

import torch.nn as nn
from pathlib import Path
from typing import Callable

from gymnasium.spaces import MultiDiscrete

from ray.rllib.algorithms import AlgorithmConfig, PPOConfig

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec

from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule

# from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical # Old API Stack
from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical
from ray.rllib.utils.from_config import NotProvided
# from ray.tune.registry import get_trainable_cls  # , register_env

# from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
# from multigrid.core.constants import Direction
HRC

# ### Helper Methods


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


def preprocess_batch(batch: dict) -> torch.Tensor:
    image = batch["obs"]["image"]
    direction = batch["obs"]["direction"]
    # direction = 2 * torch.pi * (direction / len(Direction))
    # direction = torch.stack([torch.cos(direction), torch.sin(direction)], dim=-1)
    # direction = direction[..., None, None, :].expand(*image.shape[:-1], 2)
    batch_size, ndims = direction.shape
    # 1) reshape to [b, 1, 1, …, 1, d]
    direction = direction.view(batch_size, *([1] * ndims), ndims)
    # 2) broadcast to [b, v0, v1, …, vD, d]
    direction = direction.expand(*image.shape[:-1], ndims)
    x = torch.cat([image, direction], dim=-1).float()
    return x


# ### Models


class MultiGridEncoder(nn.Module):
    def __init__(self, in_channels: int = 23):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        x = x[None] if x.ndim == 3 else x  # add batch dimension
        # x = x.permute(0, 3, 1, 2) # channels-first (N H W C -> N C H W)
        x = x.permute(
            0, -1, *(range(1, x.ndim - 1))
        )  # channels-first (N H W C -> N C H W)
        return self.model(x)


class MultiDiscreteHead(nn.Module):
    def __init__(self, feat_dim, nvec):
        super().__init__()
        # keep your branches in a ModuleList so parameters register
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feat_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, n),
                )
                for n in nvec
            ]
        )

    def forward(self, features):
        # return a *list* of logits tensors
        return [branch(features) for branch in self.branches]


class AgentModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.base = nn.Identity()

        # self.actor = nn.Sequential(
        #     MultiGridEncoder(in_channels=22),
        #     nn.Linear(64, 64), nn.ReLU(),
        #     nn.Linear(64, 7),
        # )
        # self.critic = nn.Sequential(
        #     MultiGridEncoder(in_channels=22),
        #     nn.Linear(64, 64), nn.ReLU(),
        #     nn.Linear(64, 1),
        # )

        # 1) shared encoder
        in_ch = (
            self.observation_space["image"].shape[-1]
            + self.observation_space["direction"].shape[-1]
        )
        self.encoder = MultiGridEncoder(in_channels=in_ch)
        feat_dim = self.encoder.out_dim

        # 2) critic head (unchanged)
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 3) actor head(s)
        act_space = self.action_space
        if isinstance(act_space, MultiDiscrete):
            # one small MLP per branch
            # actor_head = nn.ModuleList([
            #     nn.Sequential(
            #         nn.Linear(feat_dim, 64),
            #         nn.Linear(64, 64),
            #         nn.ReLU(),
            #         nn.Linear(64, n),
            #     )
            #     for n in act_space.nvec
            # ])
            # raise InterruptedError(f"AgentModule.setup (evoked branched agent) identified action space as {type(act_space)}")
            actor_head = MultiDiscreteHead(feat_dim, act_space.nvec)

            class TMC(TorchMultiCategorical):
                def from_logits(
                    cls,
                    logits: "torch.Tensor",
                    input_lens: list[int] = act_space.nvec,
                    temperatures: list[float] = None,
                    **kwargs,
                ):
                    super().from_logits(
                        cls, logits, input_lens, temperatures, **kwargs
                    )

            self.action_dist_cls = TMC
        else:
            actor_head = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, act_space.n),
            )
        self.actor = actor_head

    def _forward(self, batch, **kwargs):
        x = self.base(preprocess_batch(batch))
        return {Columns.ACTION_DIST_INPUTS: self.actor(x)}

    def _forward_train(self, batch, **kwargs):
        x = self.base(preprocess_batch(batch))
        return {
            Columns.ACTION_DIST_INPUTS: self.actor(x),
            Columns.EMBEDDINGS: self.critic(batch.get("value_inputs", x)),
        }

    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            x = self.base(preprocess_batch(batch))
            embeddings = self.critic(batch.get("value_inputs", x))

        return embeddings.squeeze(-1)

    def get_initial_state(self):
        return {}


### Environment

# class CustomTorchModule(TorchRLModule):
# # class CustomTorchModule(TorchRLModule, ValueFunctionAPI):
#     def setup(self) -> None:
#         # Manually build sub-nets for each part of the Dict:
#         self.img_encoder = nn.Sequential(
#             nn.Conv2d(3, 32, 3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(32 * 5 * 5, 128),
#         )
#         self.ori_encoder = nn.Sequential(
#             nn.Linear(self.observation_space["direction"].nvec.sum(), 16),
#             nn.ReLU(),
#         )
#         self.mis_encoder = nn.Embedding(num_missions, 8)

#         # Define a fusion trunk and heads:
#         self.trunk = nn.Sequential(
#             nn.Linear(128 + 16 + 8, 64),
#             nn.ReLU(),
#         )
#         self.pi_head = nn.Linear(64, self.action_space.n)
#         self.value_head = nn.Linear(64, 1)

# def _forward_train(self, batch:dict, **kwargs):
#     img = batch["obs"]["image"].float() / 255.0
#     ori = batch["obs"]["direction"].float()
#     miss = batch["obs"]["mission"].long()
#     z_img = self.img_encoder(img)
#     z_ori = self.ori_encoder(ori)
#     z_mis = self.mis_encoder(miss)
#     z = torch.cat([z_img, z_ori, z_mis], dim=1)
#     logits = self.pi_head(self.trunk(z))
#     self._value_out = self.value_head(self.trunk(z)).squeeze(1)
#     return logits, [], {}

# For VF API
# def compute_values(self, batch, embeddings=None):
#     if embeddings is None:
#         x = self.base(preprocess_batch(batch))
#         embeddings = self.critic(batch.get("value_inputs", x))

#     return embeddings.squeeze(-1)

# def get_initial_state(self):
#     return {}


### Training


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
        PPOConfig()
        # get_trainable_cls(algo)
        # .get_default_config()
        .api_stack(
            enable_env_runner_and_connector_v2=True,
            enable_rl_module_and_learner=True,
        )
        .debugging(seed=random.randint(0, 1000000))
        .framework("torch")
        .environment(env=env, env_config={**env_config, "agents": num_agents})
        .training(lr=lr, train_batch_size=batch_size)
        .multi_agent(
            policies={f"policy_{i}" for i in range(num_agents)},
            policy_mapping_fn=get_policy_mapping_fn(None, num_agents),
            policies_to_train=[f"policy_{i}" for i in range(num_agents)],
        )
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1,
            num_gpus_per_env_runner=num_gpus
            if torch.cuda.is_available()
            else 0,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    f"policy_{i}": RLModuleSpec() for i in range(num_agents)
                }
            )
        )
    )

    return config


def train(
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str = None,
):
    """
    Train an RLlib algorithm.
    """
    checkpoint = find_checkpoint_dir(load_dir)
    if checkpoint:
        tuner = ray.tune.Tuner.restore(checkpoint)
    else:
        tuner = ray.tune.Tuner(
            config.algo_class,
            param_space=config,
            run_config=ray.train.RunConfig(
                storage_path=save_dir,
                stop=stop_conditions,
                verbose=1,
                checkpoint_config=ray.train.CheckpointConfig(
                    checkpoint_frequency=20,
                    checkpoint_at_end=True,
                ),
            ),
        )
    results = tuner.fit()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build-test",
        action="store_true",
        help="Skip training, just test-build the config.",
    )

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
    #     # parser.add_argument(
    #     #     '--lstm', action='store_true',
    #     #     help="Use LSTM model.")
    #     # parser.add_argument(
    #     #     '--centralized-critic', action='store_true',
    #     #     help="Use centralized critic for training.")
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
    if args.build_test:
        config.build_algo()
        exit()

    # if args.env not in HRL.CONFIGURATIONS:
    #     # register env
    #     # TODO: add env registerer
    #     pass

    print()
    print(f"Running with following CLI options: {args}")
    print(
        "\n",
        "-" * 64,
        "\n",
        "Training with following configuration:",
        "\n",
        "-" * 64,
    )
    print()

    stop_conditions = {
        "learners/__all_modules__/num_env_steps_trained_lifetime": args.num_timesteps
    }
    train(config, stop_conditions, args.save_dir, args.load_dir)
