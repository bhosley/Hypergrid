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
        # (ACTS.ActionSpec),
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
def test_env_step_edge(ActSpec, n_dims=2):
    env = ENV_CLASS(agents=n_dims, n_dims=n_dims, agent_action_spec=ActSpec)
    env.reset()
    max_dim = np.max(env.dims)
    # Define an action so that the agents proceed forward in one direction
    mod_act = env.action_space.sample()
    for _, v in mod_act.items():
        v[0] = 1
    # Should ensure that they directions are different. But how depends on
    # ACT_SPEC. no clear way to pre-define that.

    # Repeat the same action enough times to ensure wall encounter
    try:
        for _ in range(max_dim):
            env.step(mod_act)
    except IndexError:
        raise IndexError(
            f"Dirs = {env.agent_states.dir}",
            f"Poss = {env.agent_states.pos}",
        )

    env.close()
    # FIXME: Agents seem to sometimes, somehow breach the barrier
    #   only seen in non-ortho ActionSpace so far.
    #   inconsistent, not reproducible yet.


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

        # FIXME: Not sure if this will fix orient problem
        # Check to ensure that orientation isn't all 0:
        a_dir = env.agents[agent].action_spec.to_dict(action)["orient"]
        if not any(a_dir):
            from random import randint

            i = randint(1, len(a_dir))
            action[i] = 1

    # FIXME: It appears that it is non-ortho causing an index error occasionally
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

        assert moved or could_not_move, (
            f"start pos: {starting_position[agent]}",
            f"start pos: {second_position[agent]}",
            f"ahead: {ahead}",
            f"Action: {act[agent]}Action: {act[agent]}",
            # f"ahead overlap: {ahead.can_overlap()}",
        )

    # TODO: The prior test demonstrates that agent constraints aren't checked
