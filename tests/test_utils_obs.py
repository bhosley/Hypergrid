import sys

sys.dont_write_bytecode = True

import pytest
import numpy as np

import hypergrid.utils.obs as OBS

from hypergrid.hypergrid_env import HyperGrid
from hypergrid.core.world_object import WorldObj

ENV_CLASS = HyperGrid

DIMS_AGENTS = (
    "n_agents, n_dims",
    [
        (2, 2),
        (3, 2),
        (2, 3),
    ],
)


@pytest.mark.parametrize(
    "agent_pos, agent_dir, agent_view_sizes",
    [
        (np.array(((1, 0),)), np.array(((0, 0),)), np.array([3])),
        (
            np.array(((1, 1), (0, -1))),
            np.array(((0, 0), (6, -3))),
            np.array([3, 5]),
        ),
        (
            np.array(((-1, 0), (1, 0))),
            np.array(((0, 0), (6, -3))),
            np.array([3, 3]),
        ),
        (
            np.array(((1, 0), (0, 1))),
            np.array(((0, 0), (0, 0))),
            np.array([3, 3]),
        ),
        (
            np.array(((1, 0, 0), (-1, 0, 0))),
            np.array(((0, 0, 0), (0, 0, 0))),
            np.array([3, 3]),
        ),
        (
            np.array(((1, 0, 0), (0, -1, 0))),
            np.array(((0, 0, 0), (0, 0, 0))),
            np.array([3, 3]),
        ),
        (
            np.array(((1, 0, 0), (0, 0, -1))),
            np.array(((0, 0, 0), (0, 0, 0))),
            np.array([3, 3]),
        ),
        (
            np.array(((1, 0, 0, 0), (-1, 0, 0, 0))),
            np.array(((0, 0, 0, 0), (0, 0, 0, 0))),
            np.array([3, 3]),
        ),
        (
            np.array(((1, 1), (0, -1), (-1, 0))),
            np.array(((0, 0), (-4, 6), (6, -3))),
            np.array([3, 3, 3]),
        ),
        # (np.array(),np.array(),np.array()),
    ],
)
def test_obs_get_view_coords(agent_pos, agent_dir, agent_view_sizes):
    n_dims = np.shape(agent_pos)[-1]
    coords = OBS.get_view_coords(agent_pos, agent_dir, agent_view_sizes)

    assert len(coords) == len(agent_pos)  # Confirm matching number of agents
    for i, grid in enumerate(coords):
        assert np.shape(grid) == (*((agent_view_sizes[i],) * n_dims), n_dims)


@pytest.mark.parametrize(*DIMS_AGENTS)
def test_obs_get_agent_obss(n_agents, n_dims):
    env = ENV_CLASS(agents=n_agents, dims=n_dims)
    agent_obs = OBS.gen_obs_grids(
        env.grid.state,  # grid_state,
        env.agent_states,  # agent_state,
        env.agent_view_sizes,  # agent_view_size,
    )
    assert len(agent_obs) == n_agents  # Confirm matching number of agents
    for a, grid in enumerate(agent_obs):
        assert np.shape(grid) == (
            *((env.agent_view_sizes[a],) * n_dims),
            WorldObj.dim,
        )
    env.close()


@pytest.mark.parametrize(*DIMS_AGENTS)
def test_obs_get_see_behind_mask(n_agents, n_dims):
    env = ENV_CLASS(agents=n_agents, dims=n_dims)
    agent_obs = OBS.gen_obs_grids(
        env.grid.state,  # grid_state,
        env.agent_states,  # agent_state,
        env.agent_view_sizes,  # agent_view_size,
    )
    assert len(agent_obs) == n_agents
    for a, a_obs in enumerate(agent_obs):
        see_behind_mask = OBS.get_see_behind_mask(a_obs)
        assert np.shape(see_behind_mask) == (
            *((env.agent_view_sizes[a],) * n_dims),
        )
    env.close()


@pytest.mark.parametrize(*DIMS_AGENTS)
def test_obs_grid_encoding(n_agents, n_dims):
    env = ENV_CLASS(agents=n_agents, dims=n_dims)
    agent_view_sizes = np.array([a.view_size for a in env.agents])
    agent_see_thru_walls = np.array([a.see_through_walls for a in env.agents])

    image = OBS.gen_obs_grid_encoding(
        env.grid.state,
        env.agent_states,
        agent_view_sizes,
        agent_see_thru_walls,
    )

    assert len(image) == n_agents
    for a, a_obs in enumerate(image):
        assert np.shape(a_obs) == (
            *((agent_view_sizes[a],) * n_dims),
            WorldObj.dim,
        )

    env.close()
