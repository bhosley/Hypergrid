import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is deprecated.*")

import ray
from ray.rllib.env.env_context import EnvContext

# import sys
# sys.dont_write_bytecode = True
import hypergrid.rllib as HRC

assert HRC is not None

# from scripts.flat_train import get_algorithm_config
from train import get_algorithm_config


@pytest.fixture
def config():
    cfg = get_algorithm_config()
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
    config = get_algorithm_config()
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
    config = get_algorithm_config()
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
