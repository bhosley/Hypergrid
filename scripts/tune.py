from __future__ import annotations

"""
Basic, working tuner for Hypergrid RLlib experiments.

Goal for step 1:
- Keep it minimal and reliable.
- Reuse the training configuration builder from train.py (get_algorithm_config).
- Provide a small, sane hyperparameter search space (e.g., lr and batch size).
- No W&B, no complex schedulers/algorithms (we can add later).
- CLI mirrors train.py essentials so we can iterate quickly.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import ray.train
from ray.air.integrations.wandb import WandbLoggerCallback

import torch.nn as nn

# NOTE: We intentionally import only the config builder so we don't duplicate logic.
from train import (
    get_algorithm_config,
)  # relative import within the package layout
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from train import CustomTorchModule  # reference only; we won't modify train.py


class TunableTorchModule(CustomTorchModule):
    """
    Subclass of the training module for tuning experiments.
    Starts identical to CustomTorchModule but allows tunable trunk layers.
    We do NOT modify train.py. All changes are local to this subclass.
    """

    def __init__(self, *args, **kwargs):
        # Extract an optional module_config; RLlib passes this through RLModuleSpec.
        self._tune_module_cfg = (
            kwargs.pop("model_config", None)
            or kwargs.pop("module_config", {})
            or {}
        )
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        # If provided, override parent's encoder_dims BEFORE parent setup builds encoders.
        enc = self._tune_module_cfg.get("encoder_dims")
        if enc is not None and isinstance(enc, (list, tuple)) and len(enc) == 2:
            self.encoder_dims = [int(enc[0]), int(enc[1])]

        # Build the parent module (encoders, default trunk/heads, etc.).
        super().setup()

        # Optionally rebuild the image encoder with tunable multi-layer conv stack.
        # Spatial dims are kept constant (padding='same'); flatten = last_out_ch * H * W.
        img_layers = self._tune_module_cfg.get(
            "img_layers"
        )  # e.g., [32], [32,64], [32,64,64]
        img_kernel = self._tune_module_cfg.get("img_kernel")
        img_act = (
            self._tune_module_cfg.get("img_activation") or "relu"
        ).lower()
        # Backward-compat: allow single-channel override if provided (ignored when img_layers given).
        img_channels = self._tune_module_cfg.get("img_conv_channels")
        if img_layers or img_channels or img_kernel or img_act != "relu":
            # Resolve input shape (H, W, C)
            img_shape = self.observation_space["image"].shape
            h, w, c = int(img_shape[0]), int(img_shape[1]), int(img_shape[-1])
            # Activation
            if img_act == "tanh":
                act = nn.Tanh()
            elif img_act == "gelu":
                act = nn.GELU()
            elif img_act == "elu":
                act = nn.ELU()
            else:
                act = nn.ReLU()
            # Layers: prefer explicit list; fallback to single channel override or default [32]
            if isinstance(img_layers, (list, tuple)) and len(img_layers) > 0:
                channels_list = [int(x) for x in img_layers]
            elif img_channels:
                channels_list = [int(img_channels)]
            else:
                channels_list = [32]
            k = int(img_kernel) if img_kernel else 3
            blocks: list[nn.Module] = []
            in_ch = c
            for out_ch in channels_list:
                blocks.append(
                    nn.Conv2d(
                        in_ch,
                        int(out_ch),
                        kernel_size=k,
                        stride=1,
                        padding="same",
                    )
                )
                blocks.append(act)
                in_ch = int(out_ch)
            # Project flattened conv features to the tuned image-encoder output width.
            img_out = int(self.encoder_dims[0])
            blocks.append(nn.Flatten())
            blocks.append(nn.Linear(in_ch * h * w, img_out))
            self.img_encoder = nn.Sequential(*blocks)

        # Tunable trunk: replace parent's trunk with a configurable MLP over the fused embedding.
        layers: list[int] = self._tune_module_cfg.get("trunk_layers", None)
        if not layers:
            # Keep parent's trunk if nothing provided.
            raise AssertionError("Should not get here from tuning")
            return

        # Infer input dim to the trunk. Parent defines encoder_dims and concatenates encodings.
        if hasattr(self, "encoder_dims"):
            in_dim = sum(getattr(self, "encoder_dims"))
        else:
            # Fallback: try to infer from existing trunk's first Linear layer.
            # If unavailable, do nothing (keep parent's trunk).
            try:
                first_linear = next(
                    m for m in self.trunk if isinstance(m, nn.Linear)
                )
                in_dim = first_linear.in_features
            except Exception:
                return

        # Rebuild trunk as [Linear+ReLU]* with provided hidden sizes.
        seq: list[nn.Module] = []
        prev = in_dim
        for h in layers:
            seq.append(nn.Linear(prev, int(h)))
            seq.append(nn.ReLU())
            prev = int(h)
        self.trunk = nn.Sequential(*seq)

        # Heads must match the new trunk output width.
        out_dim = prev
        # Policy head: logits dim equals sum of action space branches.
        try:
            logits_dim = int(self.action_space.nvec.sum())
        except Exception:
            # Fallback for non-MultiDiscrete; use existing head output size.
            try:
                logits_dim = next(
                    m
                    for m in self.pi_head.modules()
                    if isinstance(m, nn.Linear)
                ).out_features
            except Exception:
                return
        self.pi_head = nn.Linear(out_dim, logits_dim)
        self.value_head = nn.Linear(out_dim, 1)


def sampled_module_spec(num_agents: int, make_homo: bool):
    """
    Build RLModuleSpec at variant generation time using a resolved
    top-level hyperparam `model/trunk_layers` so Tune records the choice.
    """
    return tune.sample_from(
        lambda spec: MultiRLModuleSpec(
            rl_module_specs=(
                {
                    "policy_0": RLModuleSpec(
                        module_class=TunableTorchModule,
                        model_config={
                            "trunk_layers": spec.config["model"][
                                "trunk_layers"
                            ],
                            "encoder_dims": spec.config["model"][
                                "encoder_dims"
                            ],
                            "img_layers": spec.config["model"]["img_layers"],
                            "img_conv_channels": spec.config["model"][
                                "img_conv_channels"
                            ],
                            "img_kernel": spec.config["model"]["img_kernel"],
                            "img_activation": spec.config["model"][
                                "img_activation"
                            ],
                        },
                    )
                }
                if make_homo
                else {
                    f"policy_{i}": RLModuleSpec(
                        module_class=TunableTorchModule,
                        model_config={
                            "trunk_layers": spec.config["model"][
                                "trunk_layers"
                            ],
                            "encoder_dims": spec.config["model"][
                                "encoder_dims"
                            ],
                            "img_layers": spec.config["model"]["img_layers"],
                            "img_conv_channels": spec.config["model"][
                                "img_conv_channels"
                            ],
                            "img_kernel": spec.config["model"]["img_kernel"],
                            "img_activation": spec.config["model"][
                                "img_activation"
                            ],
                        },
                    )
                    for i in range(num_agents)
                }
            )
        )
    )


def appy_tunablel_module(base_config, num_agents: int, make_homo: bool):
    """
    Return a config that uses TunableTorchModule for all policies.
    RLModuleSpec is constructed via `tune.sample_from` to consume the resolved
    top-level hyperparam `model/trunk_layers` (so Tune records it).
    """
    return base_config.rl_module(
        rl_module_spec=sampled_module_spec(
            num_agents=num_agents, make_homo=make_homo
        )
    )


def build_param_space(
    base_config,
    env_config_base: dict,
    num_agents: int,
    make_homo: bool,
    *,
    lr_candidates: list[float] | None = None,
    batch_sizes: list[int] | None = None,
) -> Any:
    """
    Take an AlgorithmConfig and return a param_space with Tune sampling objects embedded.
    We keep the search space minimal for this first iteration.
    """
    if lr_candidates is None:
        # Conservative range; we'll broaden later.
        lr_candidates = [5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3]

    if batch_sizes is None:
        # Train batch size should be >= rollout_fragment_length; we'll tune just a few.
        batch_sizes = [4096, 8192, 16384]

    param_space = (
        base_config.environment(
            env_config={
                **env_config_base,
                "goal_shape": tune.choice([False, True]),
                "ally_shape": tune.choice([False, True]),
            }
        )
        .training(
            lr=tune.choice(lr_candidates),
            train_batch_size=tune.choice(batch_sizes),
            num_sgd_iter=tune.choice([2, 4, 6, 8]),
            entropy_coeff=tune.choice([0.0, 0.005, 0.01, 0.02]),
            vf_clip_param=tune.choice([5.0, 10.0]),
            clip_param=tune.choice([0.1, 0.2, 0.3]),
            gamma=tune.choice([0.95, 0.99]),
            lambda_=tune.choice([0.9, 0.95, 0.97, 0.99]),
            grad_clip=tune.choice([None, 0.5, 1.0, 2.0]),
        )
        # Expose model/trunk_layers at the top level so Tune samples it BEFORE building specs.
        .update_from_dict(
            {
                "model": {
                    "trunk_layers": tune.choice(
                        [[64], [128], [128, 64], [256, 128, 64]]
                    ),
                    "encoder_dims": tune.choice(
                        [[128, 16], [128, 32], [256, 16], [256, 32]]
                    ),
                    "img_layers": tune.choice(
                        [[32], [32, 32], [32, 64], [64, 64], [32, 64, 64]]
                    ),
                    "img_conv_channels": tune.choice([16, 32, 64]),
                    "img_kernel": tune.choice([2, 3]),
                    "img_activation": tune.choice(["relu", "tanh"]),
                }
            }
        )
        # Build RLModuleSpec from the resolved top-level hyperparam.
        .rl_module(
            rl_module_spec=sampled_module_spec(
                num_agents=num_agents, make_homo=make_homo
            )
        )
    )

    return param_space


def make_stop_conditions(num_timesteps: int) -> Dict[str, int]:
    # Use the same counter key as in train.py for consistency.
    return {
        "learners/__all_modules__/num_env_steps_trained_lifetime": num_timesteps
    }


def main(
    *,
    algo: str = "PPO",
    env: str = "SensorSuite",
    env_config: dict = None,
    num_agents: int = 2,
    num_workers: int = 2,
    num_gpus: int = 0,
    num_timesteps: int = int(2e6),
    eval_interval: int = 1,
    eval_episodes: int = 5,
    make_homo: bool = False,
    storage_path: str = "~/ray_results/",
    max_trials: int = 8,
    use_wandb: bool = False,
    wandb_key: str | None = None,
    project_name: str = "hypergrid",
    **kwargs,
):
    """
    Build a base Tuner and run a small search. Keep it simple for iteration.
    """
    env_config = env_config or {}

    # Build the base RLlib AlgorithmConfig using the project's canonical function.
    base_config = get_algorithm_config(
        algo=algo,
        env=env,
        env_config=env_config,
        num_agents=num_agents,
        num_workers=num_workers,
        num_gpus=num_gpus,
        make_homo=make_homo,
        **kwargs,
    )
    # Evaluation runs on an unshaped environment (goal_shape=False, ally_shape=False)
    eval_env_config = {**env_config, "goal_shape": False, "ally_shape": False}
    base_config = base_config.evaluation(
        evaluation_interval=eval_interval,
        evaluation_num_env_runners=min(num_workers, 4),
        evaluation_duration=eval_episodes,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "env": env,
            "env_config": {**eval_env_config, "agents": num_agents},
        },
    )

    # Minimal param space for step 1: LR and train_batch_size.
    param_space = build_param_space(
        base_config,
        env_config_base=env_config,
        num_agents=num_agents,
        make_homo=make_homo,
    )

    stop_conditions = make_stop_conditions(num_timesteps)

    # ASHA: early-stop underperforming trials based on clean eval metric.
    scheduler = ASHAScheduler(
        time_attr="learners/__all_modules__/num_env_steps_trained_lifetime",
        max_t=num_timesteps,
        grace_period=max(
            1, num_timesteps // 20
        ),  # ~5% of budget before pruning
        reduction_factor=3,
    )

    # Optuna search algorithm (uses metric/mode from TuneConfig)
    search_alg = OptunaSearch()

    # Make sure Ray is initialized (idempotent).
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Optional Weights & Biases logging (mirrors train.py behavior)
    callbacks = []
    if use_wandb and not wandb_key:
        load_dotenv()
        wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb_tags = [algo, "tune", "multiagent"]
        callbacks.append(
            WandbLoggerCallback(
                project=project_name,
                group=f"rllib-{algo}",
                job_type="tune",
                tags=wandb_tags,
                log_config=True,
                save_code=True,
                api_key=wandb_key,
            )
        )

    # We keep scheduler/search_alg defaults for now to reduce complexity.
    tuner = tune.Tuner(
        base_config.algo_class,  # e.g., PPO
        param_space=param_space,
        run_config=ray.train.RunConfig(
            storage_path=str(Path(storage_path).expanduser()),
            stop=stop_conditions,
            verbose=1,
            checkpoint_config=ray.train.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            callbacks=callbacks,
        ),
        tune_config=tune.TuneConfig(
            metric="evaluation/env_runners/episode_return_mean",
            mode="max",
            num_samples=max_trials,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
    )

    results = tuner.fit()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hypergrid tuner (base, step 1)"
    )

    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--env", type=str, default="SensorSuite")
    parser.add_argument(
        "--env-config",
        type=json.loads,
        default={},
        help="Environment config as JSON string, e.g. '{\"size\": 8}'",
    )
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=int(2e6),
        help="Total env steps for training budget per trial.",
    )
    parser.add_argument("--make-homo", action="store_true")
    parser.add_argument(
        "--storage-path",
        type=str,
        default="~/ray_results/",
        help="Where Tune will write results/checkpoints.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=8,
        help="Number of trials (samples) to run for this base search.",
    )
    parser.add_argument(
        "-W",
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        help="API key for wandb. If omitted, will read WANDB_API_KEY from env/.env.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="hypergrid_tune",
        help="W&B project name.",
    )

    # Allow passthrough of unknown args to get_algorithm_config (**kwargs).
    args, unknown = parser.parse_known_args()

    # Convert unknown args of form --k v into a kwargs dict for get_algorithm_config.
    passthrough: Dict[str, Any] = {}
    key = None
    for token in unknown:
        if token.startswith("--"):
            key = token.lstrip("-").replace("-", "_")
            passthrough[key] = True  # default True for flags
        else:
            if key is None:
                continue
            # Try to JSON-decode; if it fails, keep raw string/int.
            try:
                val = json.loads(token)
            except Exception:
                try:
                    val = int(token)
                except Exception:
                    try:
                        val = float(token)
                    except Exception:
                        val = token
            passthrough[key] = val
            key = None

    main(
        algo=args.algo,
        env=args.env,
        env_config=args.env_config,
        num_agents=args.num_agents,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        num_timesteps=args.num_timesteps,
        make_homo=args.make_homo,
        storage_path=args.storage_path,
        max_trials=args.max_trials,
        use_wandb=args.wandb,
        wandb_key=args.wandb_key,
        project_name=args.project_name,
        **passthrough,
    )

"""
python tune.py -W --num-agents 4 --num-workers 12 --eval-episodes 20 --max-trials 72
"""
