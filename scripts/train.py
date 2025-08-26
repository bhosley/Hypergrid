from __future__ import annotations

import argparse
import json
import os
import random
from dotenv import load_dotenv

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

import ray.tune
import ray.train
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import AlgorithmConfig, PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import (
    TorchMultiCategorical,
    TorchCategorical,
)
from ray.rllib.utils.from_config import NotProvided


import hypergrid.rllib as HRC

assert HRC is not None


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


def get_TorchMultiDiscrete(action_space):
    from ray.rllib.utils.annotations import override

    class TorchMultDiscrete(TorchMultiCategorical):
        input_lengths = list(action_space.nvec)

        @override(TorchMultiCategorical)
        def __init__(self, categoricals: list[TorchCategorical], **kwargs):
            super().__init__(categoricals)

        @classmethod
        @override(TorchMultiCategorical)
        def from_logits(cls, logits: "torch.Tensor", **kwargs):
            # If RLlib already supplied input_lens, don't override it.
            kwargs.setdefault("input_lens", cls.input_lengths)
            # Properly delegate to the parent implementation (no manual 'cls' arg).
            return super(TorchMultDiscrete, cls).from_logits(logits, **kwargs)

    return TorchMultDiscrete


class CustomTorchModule(TorchRLModule, ValueFunctionAPI):
    encoder_dims = [128, 16]

    def setup(self) -> None:
        # Manually build sub-nets for each part of the Dict:
        input_img = self.observation_space["image"].shape[-1]
        input_dir = self.observation_space["direction"].shape[0]

        self.action_dist_cls = get_TorchMultiDiscrete(self.action_space)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(input_img, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, self.encoder_dims[0]),
        )
        self.ori_encoder = nn.Sequential(
            nn.Linear(input_dir, self.encoder_dims[1]),
            nn.ReLU(),
        )
        # Define a fusion trunk and heads:
        self.trunk = nn.Sequential(
            nn.Linear(sum(self.encoder_dims), 64),
            nn.ReLU(),
        )
        # Unpack the action space shape as an int
        # self.pi_head = nn.Linear(64, *self.action_space.shape)
        self.pi_head = nn.Linear(64, int(self.action_space.nvec.sum()))
        self.value_head = nn.Linear(64, 1)

    def _forward(self, batch: dict, **kwargs):
        embeddings = self.heads_and_body(batch)
        logits = self.pi_head(embeddings)
        assert logits.shape[-1] == int(
            self.action_space.nvec.sum()
        ), f"logits dim {logits.shape[-1]} != sum(nvec) {
            int(self.action_space.nvec.sum())
        }"
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    def value_function(self, **kwargs):
        # RLlib's ValueFunctionAPI expects this to return a 1D tensor of V(s)
        # matching the last forward() call.
        return self._value_out

    def compute_values(self, batch, embeddings=None):
        """
        Compute V(s) for a batch.
        If `embeddings` (i.e., trunk output) is provided, reuse it; otherwise,
        recompute the embedding from the obs dict using this module's encoders.
        Returns a 1D tensor of shape [B].
        """
        if embeddings is None:
            embeddings = self.heads_and_body(batch)
        values = self.value_head(embeddings).squeeze(1)
        return values

    def heads_and_body(self, batch):
        img = batch["obs"]["image"].float()
        ori = batch["obs"]["direction"].float()
        # Permute channels from [-1] to [1] - (N D1 D2 C -> N C D1 D2)
        img = img.permute(0, -1, *(range(1, img.ndim - 1)))
        z_img = self.img_encoder(img)
        z_ori = self.ori_encoder(ori)
        z = torch.cat([z_img, z_ori], dim=1)
        embedding = self.trunk(z)
        return embedding


def get_algorithm_config(
    algo: str = "PPO",
    env: str = "SensorSuite",
    env_config: dict = {},
    num_agents: int = 2,
    num_workers: int = 0,
    num_gpus: int = 0,
    lr: float | None = NotProvided,
    batch_size: int | None = NotProvided,
    # lstm: bool = False, TODO: implement LSTM model
    # centralized_critic: bool = False, TODO: implement centralized critic
    make_homo: bool = False,
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
                    f"policy_{i}": RLModuleSpec(module_class=CustomTorchModule)
                    for i in range(num_agents)
                }
            )
        )
    )
    if make_homo:
        config = config.multi_agent(
            policies={"policy_0"},
            policies_to_train=["policy_0"],
            policy_mapping_fn=lambda _, *args, **kwargs: "policy_0",
        )
    else:
        config = config.multi_agent(
            policies={f"policy_{i}" for i in range(num_agents)},
            policy_mapping_fn=get_policy_mapping_fn(None, num_agents),
            policies_to_train=[f"policy_{i}" for i in range(num_agents)],
        )
    return config


def train(
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str = None,
    callbacks: list = [],
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
                callbacks=callbacks,
            ),
        )
    results = tuner.fit()
    return results


def main(
    algo: str = "PPO",
    num_timesteps: int = 1e7,
    save_dir: str = "~/ray_results/",
    load_dir: str = None,
    wandb: bool = False,
    wandb_key: str = None,
    **kwargs,
):
    """"""

    config = get_algorithm_config(**kwargs)
    callbacks = []

    if wandb and not wandb_key:
        load_dotenv()
        wandb_key = os.getenv("WANDB_KEY")
    if wandb_key:
        wandb_tags = [algo, "multiagent"]

        callbacks.append(
            WandbLoggerCallback(
                project="hypergrid",  # your W&B project
                group=f"rllib-{algo}",  # optional grouping
                job_type="train",  # optional
                tags=wandb_tags,  # optional
                log_config=True,  # log the full RLlib/Tune config
                save_code=True,  # snapshot code
                api_key=wandb_key,  # or rely on wandb login/env var
            )
        )

    # if args.env not in HRL.CONFIGURATIONS:
    #     # register env
    #     # TODO: add env registerer
    #     pass

    stop_conditions = {
        "learners/__all_modules__/num_env_steps_trained_lifetime": num_timesteps
    }
    results = train(config, stop_conditions, save_dir, load_dir, callbacks)
    return results


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
    parser.add_argument("--lstm", action="store_true", help="Use LSTM model.")
    parser.add_argument(
        "--centralized-critic",
        action="store_true",
        help="Use centralized critic for training.",
    )
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
    parser.add_argument(
        "--build-test",
        action="store_true",
        help="Skip training, just test-build the config.",
    )
    parser.add_argument(
        "-W",
        "--wandb",
        action="store_true",
        help="Log results to wandb.",
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        help="API key for wandb.",
    )
    parser.add_argument(
        "-S",
        "--silent",
        action="store_true",
        help="Try to silence what we can.",
    )

    args = parser.parse_args()

    if args.build_test:
        config = get_algorithm_config(**vars(args))
        config.build_algo()
        exit()

    if not args.silent:
        print(
            f"\nRunning with following CLI options: {args}\n",
            f"\n{'-' * 64}\n",
            "Training with following configuration:",
            f"\n{'-' * 64}\n",
        )

    main(**vars(args))


"""
python scripts/train2.py --build-test
"""
