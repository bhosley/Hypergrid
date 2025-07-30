import sys

sys.dont_write_bytecode = True

import pytest
import numpy as np

from hypergrid.core.agent import Agent
from hypergrid.hypergrid_env import HyperGrid
import hypergrid.core.actions as ACTS


"""
This file tests the environment's ability to


basic necessary functions of the env,
it will validate minimum functional requirements for a custom env.

Action Handling is tested in a separate pytest file which will test
a custom action schema in conjunction with the custom environment.
"""

ENV_CLASS = HyperGrid

ACT_SPEC = (
    "ActSpec",
    [
        (ACTS.ActionSpec),
        (ACTS.OrthogonalActionSpec),
    ],
)


@pytest.mark.parametrize(*ACT_SPEC)
def test_action_specs(ActSpec):
    test_agent = Agent(
        index=0,
        spatial_ndim=3,
        view_size=5,
        see_through_walls=True,
        action_spec=ActSpec,
    )
    sample_action = test_agent.action_space.sample()
    sample_as_dict = test_agent.action_spec.to_dict(sample_action)
    assert len(sample_as_dict) == 3
    assert list(sample_as_dict.keys()) == ["move", "orient", "interact"]


@pytest.mark.parametrize(*ACT_SPEC)
def test_env_action_handler(ActSpec):
    env = ENV_CLASS(agents=2, agent_action_spec=ActSpec)
    env.reset()
    sample_action = env.action_space.sample()
    env._handle_actions(sample_action)


@pytest.mark.parametrize(*ACT_SPEC)
def test_env_step(ActSpec):
    env = ENV_CLASS(agents=2, agent_action_spec=ActSpec)
    env.reset()
    # Ensure error if no action is supplied
    with pytest.raises((TypeError)):
        env.step()
    random_acts = env.action_space.sample()
    obss, rews, terms, truncs, infos = env.step(random_acts)
    # Each return should be a dict with an entry for each agent:
    for di in (obss, rews, terms, truncs, infos):
        assert isinstance(di, dict)
        assert len(di) == env.num_agents
    env.close()


@pytest.mark.parametrize(*ACT_SPEC)
def test_env_step_edge(ActSpec):
    env = ENV_CLASS(agents=2, agent_action_spec=ActSpec)
    env.reset()
    max_dim = np.max(env.dims)
    mod_act = env.action_space.sample()
    for _, v in mod_act.items():
        v[0] = max_dim + 4
    env.step(mod_act)
    env.close()


@pytest.mark.parametrize(*ACT_SPEC)
def test_env_step_works(ActSpec):
    env = ENV_CLASS(agents=2, agent_action_spec=ActSpec)
    env.reset()
    starting_position = np.array(env.agent_states.pos)
    # get agent action dict
    act = env.action_space.sample()
    # ensure agent move speed >0
    for agent, action in act.items():
        action[0] = env.action_space[agent][0].n - 1
    env.step(act)
    second_position = np.array(env.agent_states.pos)

    # check each agent's behavior:
    for agent in range(env.num_agents):
        moved = False
        could_not_move = False
        if not np.all(starting_position[agent] == second_position[agent]):
            moved = True
        else:
            ahead = env.grid.get(
                env.agent_states.pos[agent] + env.agent_states.dir[agent]
            )
            could_not_move = ahead is not None and not ahead.can_overlap()

        assert moved or could_not_move
