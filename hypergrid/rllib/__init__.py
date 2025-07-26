"""
This package provides tools for using HyperGrid environments with
the RLlib MultiAgentEnv API.

*****
Usage
*****

Use a specific environment configuration from :mod:`hypergrid.envs` by name:

    >>> import hypergrid.rllib # registers environment configurations with RLlib
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> algorithm_config = PPOConfig().environment(env='HyperGrid-Empty-8x8-v0')

Wrap an environment instance with :class:`.RLlibWrapper`:

    >>> import gymnasium as gym
    >>> import hypergrid.envs
    >>> env = gym.make('HyperGrid-Empty-8x8-v0', agents=2, render_mode='human')

    >>> from hypergrid.rllib import RLlibWrapper
    >>> env = RLlibWrapper(env)

Wrap an environment class with :func:`.to_rllib_env()`:

    >>> from hypergrid.envs import EmptyEnv
    >>> from hypergrid.rllib import to_rllib_env
    >>> MyEnv = to_rllib_env(EmptyEnv, default_config={'dims': [8,8]})
    >>> config = {'agents': 2, 'render_mode': 'None'}
    >>> env = MyEnv(config)
"""

import gymnasium as gym

from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env

from ..hypergrid_env import HyperGridEnv
from ..envs import CONFIGURATIONS
from ..utils.wrappers import OneHotObsWrapper


class RLlibWrapper(MultiAgentEnv):
    """
    Wrapper for a ``HyperGridEnv`` environment that implements the
    RLlib ``MultiAgentEnv`` interface.
    """

    def __init__(self, env: HyperGridEnv):
        super().__init__()
        self.env = env
        self.agents = list(range(len(env.unwrapped.agents)))
        self.possible_agents = self.agents[:]

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        obs, rewards, terminations, truncations, infos = self.env.step(
            *args, **kwargs
        )
        terminations["__all__"] = all(terminations.values())
        truncations["__all__"] = all(truncations.values())
        return obs, rewards, terminations, truncations, infos

    def get_observation_space(self, agent_index: int):
        return self.env.unwrapped.agents[agent_index].observation_space

    def get_action_space(self, agent_index: int):
        return self.env.unwrapped.agents[agent_index].action_space


def to_rllib_env(
    env_cls: type[HyperGridEnv],
    *wrappers: gym.Wrapper,
    default_config: dict = {},
) -> type[MultiAgentEnv]:
    """
    Convert a ``HyperGridEnv`` class to an RLLib ``MultiAgentEnv`` class.

    Note that this is a wrapper around the environment **class**,
    not environment instances.

    Parameters
    ----------
    env_cls : type[HyperGridEnv]
        ``HyperGridEnv`` environment class
    wrappers : gym.Wrapper
        Gym wrappers to apply to the environment
    default_config : dict
        Default configuration for the environment

    Returns
    -------
    rllib_env_cls : type[MultiAgentEnv]
        RLlib ``MultiAgentEnv`` environment class
    """

    class RLlibEnv(RLlibWrapper):
        def __init__(self, config: dict = {}):
            config = {**default_config, **config}
            env = env_cls(**config)
            for wrapper in wrappers:
                env = wrapper(env)
            super().__init__(env)

    RLlibEnv.__name__ = f"RLlib_{env_cls.__name__}"
    return RLlibEnv


# Register environments with RLlib
for name, (env_cls, config) in CONFIGURATIONS.items():
    register_env(
        name, to_rllib_env(env_cls, OneHotObsWrapper, default_config=config)
    )
