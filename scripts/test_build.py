import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is deprecated.*")

# import argparse
# import json
# import os
import random
import ray
from pathlib import Path
from typing import Callable

# import ray.train
# import ray.tune
import torch
# import torch.nn as nn
# from gymnasium.spaces import MultiDiscrete

from ray.rllib.algorithms import AlgorithmConfig, PPOConfig
from ray.rllib.connectors.env_to_module.flatten_observations import (
    FlattenObservations,
)

# from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec

# from ray.rllib.core.rl_module.apis import ValueFunctionAPI
# from ray.rllib.core.rl_module.torch import TorchRLModule
# from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical
# from ray.tune.registry import get_trainable_cls  # , register_env
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.env.env_context import EnvContext


# import sys
# sys.dont_write_bytecode = True
import hypergrid.rllib as HRC

HRC


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


# Define env-to-module-connector pipeline for the new stack.
def _env_to_module_pipeline(env, spaces, device):
    return FlattenObservations(multi_agent=True)
    # return FlattenObservations(multi_agent=args.num_agents > 0)


def algorithm_config(
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
            # _env_to_module_pipeline
            env_to_module_connector=(
                lambda env=None, spaces=None, device=None: (
                    FlattenObservations(multi_agent=True)
                )
            ),
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    f"policy_{i}": RLModuleSpec(
                        # TODO: Validate AgentModule
                        # module_class=AgentModule
                    )
                    for i in range(num_agents)
                }
            )
        )
    )

    return config


@pytest.fixture
def config():
    cfg = algorithm_config()
    # run full config sanity checks
    cfg.validate()
    return cfg


def test_to_dict_and_contents(config):
    cfg_dict = config.to_dict()
    assert isinstance(cfg_dict, dict)
    # key fields you expect:
    assert "train_batch_size" in cfg_dict
    assert cfg_dict["framework"] == "torch"


def test_additional_config_validators():
    config = algorithm_config()
    config.validate_train_batch_size_vs_rollout_fragment_length()
    assert config.validate_offline_eval_runners_after_construction
    config._validate_env_runner_settings()
    config._validate_callbacks_settings()
    config._validate_evaluation_settings()
    config._validate_framework_settings()
    config._validate_input_settings()
    config._validate_multi_agent_settings()
    config._validate_new_api_stack_settings()
    config._validate_offline_settings()
    config._validate_resources_settings()
    config._validate_to_be_deprecated_settings()


def test_build_and_env_validation():
    config = algorithm_config()
    # 1) Build your Algorithm.
    algo = config.build_algo()
    assert algo is not None
    creator = algo.env_creator
    # 2) Grab the env creator and your raw config dict.
    env_cfg = algo.config["env_config"]
    # 3) Wrap your dict in an EnvContext
    # (worker_index=0, vector_index=0 for the local runner).
    ctx = EnvContext(env_cfg, worker_index=0, vector_index=0)
    # 4) Instantiate the env via the creator.
    env = creator(ctx)
    algo.validate_env(env, ctx)


def test_tuner_init(config):
    # No need to call .fit(); just constructing should pass
    tuner = ray.tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
    )
    assert tuner is not None


def test_one_train_step(config):
    algo = config.build_algo()
    result = algo.train()
    # RLlib always returns a dict with these keys
    # assert "episode_reward_mean" in result
    assert "env_runners" in result
    assert "learners" in result
    assert "perf" in result


# def test_checkpoint_and_restore(config, tmp_path):
#     algo1 = config.build_algo()
#     res1 = algo1.train()
#     ckpt = algo1.save(str(tmp_path))

#     algo2 = config.build_algo()
#     algo2.restore(ckpt)
#     res2 = algo2.train()
#     # same keys, and it shouldn’t error
#     assert "episode_reward_mean" in res2

# from scripts.test_build import get_policy_mapping_fn

# def test_policy_mapping_fn():
#     fn = get_policy_mapping_fn(None, num_agents=4)
#     valid_ids = {f"policy_{i}" for i in range(4)}
#     # try a handful of agent indices
#     for agent_id in [0,1,2,3,4,7,11]:
#         assert fn(agent_id) in valid_ids

# def test_rl_module_spec(config):
#     # this will catch missing module_class or bad specs
#     config._validate_rl_module_settings()
