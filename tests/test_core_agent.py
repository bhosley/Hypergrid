import sys

sys.dont_write_bytecode = True

import pytest
import numpy as np

from gymnasium.spaces import MultiDiscrete

from hypergrid.core.agent import Agent, AgentState
from hypergrid.core.world_object import WorldObj
from hypergrid.core.mission import MissionSpace


# ------------------------- Single Agent Instantiation -------------------------


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        "foo",
        0.5,
        -3,
    ],
)
def test_agent_invalid_int_params_raise_value_error(invalid_input):
    """Passing an unsupported dims type should raise ValueError."""
    with pytest.raises((ValueError, TypeError)):
        Agent(index=invalid_input, spatial_ndim=2, view_size=5)
        Agent(index=0, spatial_ndim=invalid_input, view_size=5)
        Agent(index=0, spatial_ndim=2, view_size=invalid_input)


@pytest.mark.parametrize(
    "invalid_input",
    [None, "foo"],
)
def test_agent_invalid_flt_params_raise_value_error(invalid_input):
    """Passing an unsupported dims type should raise ValueError."""
    with pytest.raises((ValueError, TypeError)):
        Agent(index=0, cost_collision_agent=invalid_input)
        Agent(index=0, cost_collision_env=invalid_input)
        Agent(index=0, cost_interaction=invalid_input)


@pytest.mark.parametrize(
    "spatial_ndim, view_size",
    [
        (1, 3),
        (2, 5),
        (3, 7),
    ],
)
def test_agent_initialization_and_obs_space(spatial_ndim, view_size):
    """Agent should initialize correctly and expose the right observation & action spaces."""
    agent = Agent(index=0, spatial_ndim=spatial_ndim, view_size=view_size)

    # Check observation space keys and types
    assert "image" in agent.observation_space.spaces
    assert "orientation" in agent.observation_space.spaces
    assert "mission" in agent.observation_space.spaces
    # Check observation space shapes and types
    image_space = agent.observation_space["image"]
    orient_space = agent.observation_space["orientation"]
    mission_space = agent.observation_space["mission"]

    assert image_space.dtype == np.uint8
    assert image_space.shape == (agent.view_size,) * spatial_ndim + (
        WorldObj.dim,
    )
    assert isinstance(orient_space, MultiDiscrete)
    assert list(orient_space.nvec) == [3] * spatial_ndim
    assert isinstance(mission_space, MissionSpace)

    # Check action space type and parameters
    assert isinstance(agent.action_space, MultiDiscrete)
    assert len(agent.action_space.nvec) == 1 + spatial_ndim + 1
    # assert list(agent.action_space.nvec) == [3] * spatial_ndim
    # ^ Too specific, testing length for now

    # Check initial state values
    assert isinstance(agent.state, AgentState)
    # assert (  agent.state.pos == tuple([1] * spatial_ndim)  ).all()
    assert np.all(agent.state.pos == tuple([1] * spatial_ndim))
    assert not agent.state.terminated
    assert agent.state.carrying == None
    # default orientation mapped to 0 movement
    # assert (  agent.state.orientation == (1,) * spatial_ndim  ).all()
    assert np.all(agent.state.orientation == (1,) * spatial_ndim)


@pytest.mark.parametrize(
    "dims, new_pos, new_orient",
    [
        (2, (4, 4), (-1, 1)),
    ],
)
def test_agent_state_setters(dims, new_pos, new_orient):
    agent = Agent(index=0, spatial_ndim=dims)

    # Test position setter via alias
    agent.pos = new_pos
    # assert ( agent.pos == new_pos ).all()
    assert agent.pos == new_pos

    # Test orientation setter on state
    agent.orientation = new_orient
    # assert ( agent.orientation == new_orient ).all()
    assert np.all(agent.orientation == new_orient)
    # Test Dir alias for orientation
    # assert ( agent.dir == new_orient ).all()
    assert np.all(agent.dir == new_orient)
    second_orient = tuple(reversed(new_orient))
    agent.dir = second_orient
    # assert ( agent.orientation == second_orient ).all()
    # assert ( agent.dir == second_orient ).all()
    assert np.all(agent.orientation == second_orient)
    assert np.all(agent.dir == second_orient)

    # Test terminated setter
    agent.terminated = True
    assert agent.terminated == True
    agent.terminated = False
    assert agent.terminated == False

    # Test carrying setter
    assert agent.carrying == None
    payload = object()
    agent.carrying = payload
    assert agent.carrying == payload


# Test invalid setters
#     # Invalid position length raises
#     with pytest.raises(ValueError):
#         agent.pos = (1,)  # too short
# # Invalid orientation length raises
# with pytest.raises(ValueError):
#     agent.state.orientation = (0,)  # wrong length

# ------------------------ Vectorized AgentState Tests  ------------------------

from hypergrid.hypergrid_env import HyperGrid

ENV_CLASS = HyperGrid


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        "foo",
        0.5,
        -3,
    ],
)
def test_agentstate_invalid_params_raise_value_error(invalid_input):
    """Passing an unsupported dims type should raise ValueError."""
    with pytest.raises((ValueError, TypeError)):
        AgentState(invalid_input, spatial_ndim=2)
        AgentState(2, spatial_ndim=invalid_input)


def test_agentstate_constants_update():
    _ = ENV_CLASS()
    init_type_idx = int(AgentState.TYPE_IDX)
    init_orient_idx = int(AgentState.ORIENT_END)
    init_total_dim = int(AgentState.total_dim)
    # Evoking AgentState should update index constants
    _ = ENV_CLASS(agents=3, n_dims=3)
    assert init_type_idx == AgentState.TYPE_IDX
    assert init_orient_idx != AgentState.ORIENT_END
    assert init_total_dim != AgentState.total_dim
    # Evoking AgentState changes under n_dims not n agents
    _ = ENV_CLASS(agents=3)
    assert init_type_idx == AgentState.TYPE_IDX
    assert init_orient_idx == AgentState.ORIENT_END
    assert init_total_dim == AgentState.total_dim
