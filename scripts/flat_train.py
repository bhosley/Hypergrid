import random
from pathlib import Path
from typing import Callable

import torch

from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.connectors.env_to_module.flatten_observations import (
    FlattenObservations,
)
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.algorithms import AlgorithmConfig
from ray.tune.registry import get_trainable_cls


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
