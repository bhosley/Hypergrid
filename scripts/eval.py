from __future__ import annotations

import argparse
import os
import ray
import random
import torch
import wandb

from dotenv import load_dotenv
import numpy as np
from pathlib import Path

from ray.rllib.core.rl_module import RLModule
from hypergrid.envs.sensor_suite import SensorSuiteEnv as ENVCLASS


def flip_random_bit(lst, val=0):
    agent = random.choice(list(lst.keys()))
    # Find indices of all 1s
    indices = [i for i, v in enumerate(lst[(agent)][1:]) if v != val]
    if not indices:
        return lst  # no 1s to flip
    # Pick a random index that doesn't contain the value
    idx = random.choice(indices) + 1
    # Flip it to 0
    lst[agent][idx] = val
    return lst


def change_coverage(sensors, decrease=True):
    arr = np.array(list(sensors.values()))
    vis = np.sum(arr[:, 1:], axis=0)
    if decrease:
        indices = [i for i, v in enumerate(vis) if v > 0]
    else:
        indices = [i for i, v in enumerate(vis) if v == 0]
    if not indices:
        return sensors
    idx = random.choice(indices) + 1
    if decrease:
        for i in sensors.keys():
            sensors[i][idx] = 0
    else:
        # Need ensure
        agent = random.choice(list(range(len(sensors))))
        sensors[agent][idx] = 1
    return sensors


def main(
    load_dir: Path | str,
    num_agents: int,
    target_env=ENVCLASS,
    agent_sensors: dict = None,
    episodes: int = 1,
    use_wandb: bool = False,
    wandb_key: str = None,
    project_name: str = "hypergrid",
    sensor_config: str = None,
    policy_type: str = None,
    **kwargs,
):
    """ """
    if not agent_sensors:
        agent_sensors = {
            i: [
                1,
            ]
            * 7
            for i in range(num_agents)
        }  # Full vis exp
    if use_wandb and not wandb_key:
        load_dotenv()
        wandb_key = os.getenv("WANDB_API_KEY")

    # Construct eval configs
    overall_conf = {"agents": num_agents}
    eval_types = [
        "baseline",
        "agent_loss",
        "sensor_degradation",
        "sensor_improvement",
        "degrade_coverage",
        "improve_coverage",
    ]
    eval_configs = {_: {} for _ in eval_types}
    # Agent loss - Set some agents to terminated
    cas = random.choice(list(range(num_agents)))
    eval_configs["agent_loss"] = {"remove_agents": [cas]}
    # Sensor degradation
    eval_configs["sensor_degradation"] = {
        "agent_sensors": flip_random_bit(agent_sensors)
    }
    # Sensor improve
    eval_configs["sensor_improvement"] = {
        "agent_sensors": flip_random_bit(agent_sensors, val=1)
    }
    # Change in coverage
    eval_configs["degrade_coverage"] = {
        "agent_sensors": change_coverage(agent_sensors)
    }
    eval_configs["improve_coverage"] = {
        "agent_sensors": change_coverage(agent_sensors, decrease=False)
    }

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    # Restore agents
    restored_module = RLModule.from_checkpoint(
        Path(load_dir) / "learner_group/learner/rl_module"
    )
    # Eval each type
    for eval_type, eval_conf in eval_configs.items():
        # env = target_env(**overall_conf, **eval_conf, **kwargs)
        config = kwargs | overall_conf | eval_conf
        env = target_env(**config)
        if wandb_key:
            wandb.login()
            wandb.init(
                project=f"{project_name}_eval_3",
                name=Path(load_dir).parent.name,
                group=policy_type,
            )
            wandb.config.update(
                {
                    **config,
                    "num_agents": num_agents,
                    "policy_type": policy_type,
                    "eval_type": eval_type,
                    "policy_eval": f"{sensor_config}_{eval_type}",
                }
            )
        eval_episode(
            policy_set=restored_module,
            env=env,
            episodes=episodes,
            wandb_key=wandb_key,
        )
        if wandb_key:
            wandb.finish()


def eval_episode(policy_set, env, episodes=1, wandb_key=None):
    for ep in range(episodes):
        obss, infos = env.reset()
        rewards = {_: 0 for _ in range(env.env.num_agents)}
        over = False
        episode_length = 0
        while not over:
            actions = {}
            for i, o in obss.items():
                pol = policy_set[f"policy_{i}"]
                tens_obs = {
                    _: torch.from_numpy(v).unsqueeze(0) for _, v in o.items()
                }
                logits = pol.forward_inference({"obs": tens_obs})[
                    "action_dist_inputs"
                ]
                act_dist = pol.action_dist_cls.from_logits(logits)
                act = act_dist.to_deterministic().sample().T[0]
                actions[i] = act
            # Step
            obss, rews, terms, truncs, infos = env.step(actions)
            # Record rewards
            for i, r in rews.items():
                rewards[i] += r
            episode_length += 1
            over = np.all(list(terms.values())) or np.all(list(truncs.values()))
        # Log metrics
        if wandb_key:
            episode_stats = {
                "episode": ep + 1,
                "metrics/length": episode_length,
                "metrics/returns/mean": np.mean(list(rewards.values())),
                "metrics/returns/min": np.min(list(rewards.values())),
                "metrics/returns/max": np.max(list(rewards.values())),
            }
            agent_stats = {
                f"metrics/returns/policy_{i}": r for i, r in rewards.items()
            }
            wandb.log(episode_stats | agent_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-dir",
        type=str,
        help="Checkpoint directory for loading pre-trained policies.",
    )
    parser.add_argument(
        "-W",
        "--use-wandb",
        action="store_true",
        help="Log results to wandb.",
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        help="API key for wandb.",
    )
    # TODO: add params

    args = parser.parse_args()
    main(**vars(args))
