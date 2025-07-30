import sys

sys.dont_write_bytecode = True

import pytest
import numpy as np
from numbers import Number

from hypergrid.core.agent import Agent
from hypergrid.core.world_object import WorldObj
from hypergrid.hypergrid_env import HyperGrid

"""
This file tests the basic necessary functions of the env,
it will validate minimum functional requirements for a custom env.

Action Handling is tested in a separate pytest file which will test
a custom action schema in conjunction with the custom environment.
"""

ENV_CLASS = HyperGrid

# ---------------------------- Initialization Tests ----------------------------


def test_env_has_core_methods():
    """Environment exposes the default petting zoo methods."""
    env = ENV_CLASS()
    for method in ("reset", "step", "render", "close"):
        assert hasattr(env, method), f"Missing {method}() method"
    env.close()


@pytest.mark.parametrize(
    "invalid_dims",
    [
        None,
        [1, 2, None],  # Nonetype errors
        "foo",
        [1, 2, "3"],  # String input
        0.5,
        [1, 2, 0.5],  # Half n or d_n not supported
        -3,
        [1, 2, -3],  # Negative n or d_n not supported
    ],
)
def test_env_invalid_dims_raise_error(invalid_dims):
    """Passing an unsupported dims type should raise ValueError."""
    with pytest.raises((ValueError, TypeError)):
        ENV_CLASS(dims=invalid_dims)
        ENV_CLASS(n_dims=invalid_dims)


@pytest.mark.parametrize(
    "dims, expected_shape",
    [
        (2, (10, 10)),  # int → uniform 2-D grid
        (3, (10, 10, 10)),  # another int
        ([4, 5], (4, 5)),  # list → per-dimension sizes
        ((6, 7, 8), (6, 7, 8)),  # tuple → an n-D grid
    ],
)
def test_env_dims_normalization(dims, expected_shape):
    env = ENV_CLASS(dims=dims)
    shape = env.grid.shape
    assert shape == (
        *expected_shape,
        WorldObj.dim,
    ), f"dims={dims} returned shape {shape} instead of {expected_shape}"
    ds = env.grid.dims
    assert ds == [*expected_shape], (
        f"dims={ds} returned dims {ds} instead of {expected_shape}"
    )
    env.close()


@pytest.mark.parametrize(
    "n_dims, expected_shape",
    [
        (2, (10, 10)),
        (3, (10, 10, 10)),
    ],
)
def test_env_n_dims_normalization(n_dims, expected_shape):
    env = ENV_CLASS(n_dims=n_dims)
    shape = env.grid.shape
    assert shape == (
        *expected_shape,
        WorldObj.dim,
    ), f"n_dims={n_dims} returned shape {shape} instead of {expected_shape}"
    dims = env.grid.dims
    assert dims == [*expected_shape], (
        f"n_dims={n_dims} returned dims {dims} instead of {expected_shape}"
    )
    env.close()


# ---------------------------- Agent Building Tests ----------------------------


@pytest.mark.parametrize(
    "invalid_input",
    [None, "foo", 0.5, -3, [1, 1]],
)
def test_env_invalid_agents_raise_error(invalid_input):
    """Passing an unsupported dims type should raise ValueError."""
    with pytest.raises((ValueError, TypeError)):
        ENV_CLASS(agents=invalid_input)


@pytest.mark.parametrize(
    "agents_param, expected_out",
    [
        (2, 2),
        ((Agent(index=0, spatial_ndim=2), Agent(index=0, spatial_ndim=2)), 2),
    ],
)
def test_env_agents_params(agents_param, expected_out):
    env = ENV_CLASS(agents=agents_param)
    assert len(env.agents) == expected_out
    assert all((isinstance(a, Agent) for a in env.agents))
    env.close()


@pytest.mark.parametrize(
    "agents_param, agent_view_size, see_through_walls",
    [
        (2, 3, True),
        (2, (3, 3), True),
        (2, (3, 5), True),
        (2, 3, (True, True)),
        (3, (3, 5, 3), True),
    ],
)
def test_env_new_agents_views(agents_param, agent_view_size, see_through_walls):
    env = ENV_CLASS(
        agents=agents_param,
        agent_view_size=agent_view_size,
        see_through_walls=see_through_walls,
    )
    assert len(env.agent_see_through_walls) == agents_param
    assert len(env.agent_view_sizes) == agents_param
    env.close()


@pytest.mark.parametrize(
    "agents_param, attr",
    [
        (2, "color"),
        (2, "pos"),
        (2, "orientation"),
        (2, "dir"),
    ],
)
def test_env_agent_state_accessors(agents_param, attr):
    env = ENV_CLASS(agents=agents_param)
    attrs = getattr(env.agent_states, attr)
    for i, agent in enumerate(env.agents):
        assert np.all(getattr(agent, attr) == attrs[i])


# ----------------------- Action/Observation Space Tests -----------------------


@pytest.mark.parametrize(
    "num_agents",
    [1, 2],
)
def test_env_action_space(num_agents):
    env = ENV_CLASS(agents=num_agents)
    assert len(env.action_space) == num_agents
    assert len(env.observation_space) == num_agents


# ------------------------------ Env Reset Tests -------------------------------


def test_env_reset():
    env = ENV_CLASS()
    obss, infos = env.reset()
    assert isinstance(infos, dict)
    assert obss is not None
    assert len(obss) == env.num_agents
    for a, obs in obss.items():
        assert np.all(
            obs["image"].shape
            == np.array(
                [*(env.agent_view_sizes[a],) * env.n_dims, WorldObj.dim]
            )
        )
        assert len(obs["direction"]) == env.n_dims
        # assert obs["mission"] is not None
    env.close()


# ------------------------------- Env Step Tests -------------------------------


def test_env_step():
    env = ENV_CLASS()
    env.reset()
    with pytest.raises((TypeError)):
        # Ensure error if no action is supplied
        env.step()
    random_acts = {i.index: i.action_space.sample() for i in env.agents}
    obss, rews, terms, truncs, infos = env.step(random_acts)
    for agent in env.agents:
        i = agent.index
        assert isinstance(truncs[i], (np.bool_, bool))
        assert isinstance(terms[i], (np.bool_, bool))
        assert isinstance(rews[i], Number)
        assert isinstance(infos, dict)
        obs = obss[i]
        assert np.all(
            obs["image"].shape
            == np.array(
                [*(env.agent_view_sizes[i],) * env.n_dims, WorldObj.dim]
            )
        )
        assert len(obs["direction"]) == env.n_dims
        # assert obs["mission"] is not None
    env.close()


# TODO: Verify prevent agent spawning inside wall
# TODO: Verify goal intent
