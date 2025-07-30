from __future__ import annotations

import argparse
import os
import json
import random
from pathlib import Path
from typing import Callable

import ray
import torch

from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.connectors.env_to_module.flatten_observations import (
    FlattenObservations,
)
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.algorithms import AlgorithmConfig
from ray.tune.registry import get_trainable_cls

from ray.rllib.utils.test_utils import (
    check_train_results,
    check_learning_achieved,
    check_train_results_new_api_stack,
)

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
            env_to_module_connector=(
                lambda env=None, spaces=None, device=None: (
                    FlattenObservations(multi_agent=True)
                )
            ),
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
            run_config=ray.tune.RunConfig(
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
        default=int(1e7),
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

    # TODO: Replace this with a proper reader
    args.env_config = {
        "max_steps": args.num_timesteps,
    }

    config = get_algorithm_config(**vars(args))
    if args.build_test:
        config.build_algo()
        exit()

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
    results = train(config, stop_conditions, args.save_dir, args.load_dir)

    try:
        print("Errored trials:", results.num_errors)
    except Exception:
        print("No 'results.num_errors' found")
    try:
        print("Cleanly terminated trials:", results.num_terminated)
    except Exception:
        print("No 'results.num_terminated' found")
    # Any exceptions from errored trials:
    try:
        print("Error details:", results.errors)
    except Exception:
        print("No 'results.errors' found")

    check_train_results(results)
    check_learning_achieved(results, 0.0)
    check_train_results_new_api_stack(results)
