from __future__ import annotations

import argparse
import json
import random
from typing import Callable

import torch
import torch.nn as nn

import ray.tune
import ray.train
from ray.tune.search.optuna import OptunaSearch
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
    checkpoint_dir: None,
    num_agents: int,
) -> Callable:
    # For tuning, use a simple round-robin mapping policy_i per agent.
    return lambda agent_id, *args, **kwargs: f"policy_{agent_id % num_agents}"


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
    """
    A configurable TorchRLModule whose encoder widths/depths and trunk can be tuned.

    Pass hyperparameters via RLModuleSpec(..., model_config_dict={...}), e.g.:
      {
        "img_conv_spec": [{"out_channels": 16, "kernel": 3, "pool": False},
                          {"out_channels": 32, "kernel": 3, "pool": False}],
        "img_latent_dim": 128,
        "ori_mlp_layers": [16],
        "trunk_layers": [64],
        "activation": "relu"
      }
    """

    def __init__(self, *args, **kwargs):
        # Allow RLlib to provide a config dict via 'model_config_dict' or 'config'
        self.config = (
            kwargs.pop("model_config_dict", kwargs.get("config", {})) or {}
        )
        super().__init__(*args, **kwargs)

    def _get_activation(self):
        act = (self.config.get("activation") or "relu").lower()
        if act == "gelu":
            return nn.GELU
        elif act == "tanh":
            return nn.Tanh
        elif act == "elu":
            return nn.ELU
        return nn.ReLU

    @staticmethod
    def _build_mlp(
        input_dim: int, layer_sizes: list[int], activation
    ) -> tuple[nn.Module, int]:
        """Build an MLP; returns (nn.Sequential, output_dim). If layer_sizes is empty, Identity."""
        if not layer_sizes:
            return nn.Identity(), input_dim
        layers: list[nn.Module] = []
        last = input_dim
        for h in layer_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        return nn.Sequential(*layers), last

    def _build_conv_stack(self, in_channels: int, activation) -> nn.Module:
        """
        Build a Conv stack from spec. Spec is a list of dicts with keys:
          - out_channels (int)
          - kernel (int | tuple[int, int])
          - pool (bool) optional -> MaxPool2d(2)
        After convs, flattens and projects to img_latent_dim via LazyLinear.
        """
        conv_spec = self.config.get("img_conv_spec")
        if not conv_spec:
            # Default V1 (mirrors the original simple encoder)
            conv_spec = [
                {"out_channels": 32, "kernel": 3, "pool": False},
            ]
        layers: list[nn.Module] = []
        last_c = in_channels
        for layer in conv_spec:
            k = layer.get("kernel", 3)
            oc = int(layer.get("out_channels", 32))
            layers.append(nn.Conv2d(last_c, oc, k, padding="same"))
            layers.append(activation())
            if layer.get("pool", False):
                layers.append(nn.MaxPool2d(2))
            last_c = oc
        layers.append(nn.Flatten())
        # Lazy projection avoids knowing the post-conv spatial size.
        img_latent_dim = int(self.config.get("img_latent_dim", 128))
        layers.append(nn.LazyLinear(img_latent_dim))
        return nn.Sequential(*layers)

    def setup(self) -> None:
        Act = self._get_activation()

        # Observation subspaces
        input_img_ch = self.observation_space["image"].shape[-1]
        input_ori_dim = self.observation_space["direction"].shape[0]

        # Action distribution class (MultiDiscrete helper remains)
        self.action_dist_cls = get_TorchMultiDiscrete(self.action_space)

        # Encoders
        self.img_encoder = self._build_conv_stack(
            in_channels=input_img_ch, activation=Act
        )
        ori_layers = list(self.config.get("ori_mlp_layers", [16]))
        self.ori_encoder, ori_out = self._build_mlp(
            input_ori_dim, ori_layers, Act
        )

        # Trunk
        trunk_layers = list(self.config.get("trunk_layers", [64]))
        trunk_in = int(self.config.get("img_latent_dim", 128)) + ori_out
        self.trunk, trunk_out = self._build_mlp(trunk_in, trunk_layers, Act)

        # Heads
        self.pi_head = nn.Linear(trunk_out, int(self.action_space.nvec.sum()))
        self.value_head = nn.Linear(trunk_out, 1)

        # Storage for ValueFunctionAPI
        self._value_out = None

    def heads_and_body(self, batch):
        img = batch["obs"]["image"].float()  # (N, H, W, C)
        ori = batch["obs"]["direction"].float()
        # NHWC -> NCHW
        img = img.permute(0, -1, *(range(1, img.ndim - 1)))
        z_img = self.img_encoder(img)
        z_ori = self.ori_encoder(ori)
        z = torch.cat([z_img, z_ori], dim=1)
        embedding = self.trunk(z)
        return embedding

    def _forward(self, batch: dict, **kwargs):
        embeddings = self.heads_and_body(batch)
        logits = self.pi_head(embeddings)
        self._value_out = self.value_head(embeddings).squeeze(1)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    def value_function(self, **kwargs):
        # RLlib's ValueFunctionAPI expects this to return the last-computed V(s)
        return self._value_out

    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = self.heads_and_body(batch)
        values = self.value_head(embeddings).squeeze(1)
        return values


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
    module_config: dict | None = None,
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
                    f"policy_{i}": RLModuleSpec(
                        module_class=CustomTorchModule,
                        model_config_dict=module_config,
                    )
                    for i in range(num_agents)
                }
            )
        )
    )
    if make_homo:
        config = config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "policy_0": RLModuleSpec(
                        module_class=CustomTorchModule,
                        model_config_dict=module_config,
                    )
                }
            )
        ).multi_agent(
            policies={"policy_0"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "policy_0",
            policies_to_train=["policy_0"],
        )
    else:
        config = config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    f"policy_{i}": RLModuleSpec(
                        module_class=CustomTorchModule,
                        model_config_dict=module_config,
                    )
                    for i in range(num_agents)
                }
            )
        ).multi_agent(
            policies={f"policy_{i}" for i in range(num_agents)},
            policy_mapping_fn=get_policy_mapping_fn(None, num_agents),
            policies_to_train=[f"policy_{i}" for i in range(num_agents)],
        )
    return config


def tune_custom_module(
    env: str = "SensorSuite",
    env_config: dict | None = None,
    num_agents: int = 2,
    num_workers: int = 4,
    num_gpus: int = 0,
    num_samples: int = 20,
    storage_path: str = "~/ray_results/",
):
    """
    Run Ray Tune to search over CustomTorchModule hyperparameters.

    Search space includes:
      - training.lr
      - training.train_batch_size
      - module_config: img_conv_spec (V1/V2), img_latent_dim, ori_mlp_layers, trunk_layers, activation
    """
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    # Base (deterministic) config
    base = get_algorithm_config(
        env=env,
        env_config=env_config or {},
        num_agents=num_agents,
        num_workers=num_workers,
        num_gpus=num_gpus,
        # lr and batch_size will be overridden by Tune below
        lr=NotProvided,
        batch_size=NotProvided,
        make_homo=False,
        # module_config will be overridden by Tune below
        module_config=None,
    )
    # Always evaluate with shaping disabled so we optimize on unshaped reward
    base = base.evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=1,
        evaluation_duration=5,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "env_config": {
                **(env_config or {}),
                "goal_shape": False,
                "ally_shape": False,
            }
        },
    )

    # Define search space by overriding fields in the config
    def _sample_module_cfg():
        # Depth 1-4
        depth = random.choice([1, 2, 3, 4])
        # Monotonic non-decreasing channel schedule (progressively wider)
        base = random.choice([16, 24, 32])
        step = random.choice([0, 8, 16])  # growth per layer
        channels = [base + i * step for i in range(depth)]
        # Per-layer kernel/pooling with SAME padding handled in module
        kernels = [random.choice([3, (3, 3), (2, 2)]) for _ in range(depth)]
        pools = [random.choice([False, False, True]) for _ in range(depth)]
        conv_spec = [
            {"out_channels": c, "kernel": k, "pool": p}
            for c, k, p in zip(channels, kernels, pools)
        ]
        last_c = channels[-1]
        # Tie latent/trunk to conv width to avoid under/over-bottlenecking
        img_latent_dim = random.choice([last_c, last_c * 2, last_c * 3])
        # Orientation encoder sized modestly relative to latent
        ori_choice = random.choice([[16], [32], [32, 16], []])
        # Trunk layers depend on latent size
        trunk_options = [
            [img_latent_dim],
            [img_latent_dim, img_latent_dim // 2],
            [2 * img_latent_dim, img_latent_dim],
            # allow deeper trunks as needed (3 layers)
            [img_latent_dim, img_latent_dim, img_latent_dim // 2],
            [2 * img_latent_dim, img_latent_dim, img_latent_dim // 2],
        ]
        trunk_layers = random.choice(trunk_options)
        activation = random.choice(["relu", "gelu"])
        return {
            "img_conv_spec": conv_spec,
            "img_latent_dim": int(img_latent_dim),
            "ori_mlp_layers": ori_choice,
            "trunk_layers": [int(x) for x in trunk_layers],
            "activation": activation,
        }

    search_space = (
        base.copy(copy_frozen=True)
        .training(
            # Learning rate + batch size are tuned here
            lr=tune.loguniform(1e-5, 3e-3),
            train_batch_size=tune.choice([2048, 4096, 8192]),
        )
        .environment(
            env_config=tune.sample_from(
                lambda _: {
                    **(env_config or {}),
                    "goal_shape": random.choice([False, True]),
                    "ally_shape": random.choice([False, True]),
                }
            )
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    f"policy_{i}": RLModuleSpec(
                        module_class=CustomTorchModule,
                        model_config_dict=tune.sample_from(
                            lambda _: _sample_module_cfg()
                        ),
                    )
                    for i in range(num_agents)
                }
            )
        )
    )

    scheduler = ASHAScheduler(
        metric="evaluation/episode_reward_mean",  # Use evaluation metric
        mode="max",
        grace_period=5,
        reduction_factor=3,
        max_t=200,
    )

    search_alg = OptunaSearch()

    tuner = ray.tune.Tuner(
        search_space.algo_class,
        param_space=search_space,
        run_config=ray.train.RunConfig(
            storage_path=storage_path,
            stop={"training_iteration": 50},
            verbose=1,
            checkpoint_config=ray.train.CheckpointConfig(
                checkpoint_frequency=0,
                checkpoint_at_end=False,
            ),
        ),
        tune_config=ray.tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
            search_alg=search_alg,
        ),
    )
    results = tuner.fit()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="SensorSuite",
        help="HyperGrid environment to use.",
    )
    parser.add_argument(
        "--env-config",
        type=json.loads,
        default={},
        help='Environment config JSON string, e.g. {"size":8}',
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents in environment.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of rollout workers."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0, help="Number of GPUs to train on."
    )
    parser.add_argument(
        "--num-samples", type=int, default=20, help="Number of Tune samples."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="~/ray_results/",
        help="Directory for saving Tune results.",
    )
    args = parser.parse_args()
    tune_custom_module(
        env=args.env,
        env_config=args.env_config,
        num_agents=args.num_agents,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        num_samples=args.num_samples,
        storage_path=args.save_dir,
    )
