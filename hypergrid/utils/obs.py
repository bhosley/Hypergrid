import numba as nb
import numpy as np
from itertools import product
from typing import Iterable

from ..core.agent import AgentState
from ..core.constants import Color, State, Type
from ..core.world_object import Wall, WorldObj

from numpy.typing import NDArray as ndarray

### Constants

WALL_ENCODING = Wall().encode()
UNSEEN_ENCODING = WorldObj(Type.unseen, Color.from_index(0)).encode()
ENCODE_DIM = WorldObj.dim

GRID_ENCODING_IDX = slice(None)

AGENT_DIR_IDX = AgentState._slice_orient
# AGENT_DIR_IDX = AgentState.DIR
AGENT_POS_IDX = AgentState._slice_pos
# AGENT_POS_IDX = AgentState.POS
AGENT_TERMINATED_IDX = AgentState.TERM_IDX
# AGENT_TERMINATED_IDX = AgentState.TERMINATED
AGENT_CARRYING_IDX = AgentState._slice_carry
# AGENT_CARRYING_IDX = AgentState.CARRYING
AGENT_ENCODING_IDX = AgentState.ENCODING

TYPE = WorldObj.TYPE
STATE = WorldObj.STATE

WALL = int(Type.wall)
DOOR = int(Type.door)

OPEN = int(State.open)
CLOSED = int(State.closed)
LOCKED = int(State.locked)

# RIGHT = int(Direction.right)
# LEFT = int(Direction.left)
# UP = int(Direction.up)
# DOWN = int(Direction.down)


### Observation Functions


@nb.njit(cache=True)
def see_behind(world_obj: ndarray[np.int_]) -> bool:
    """
    Can an agent see behind this object?

    Parameters
    ----------
    world_obj : ndarray[int] of shape (encode_dim,)
        World object encoding
    """
    if world_obj is None:
        return True
    if world_obj[TYPE] == WALL:
        return False
    elif world_obj[TYPE] == DOOR and world_obj[STATE] != OPEN:
        return False

    return True


# @nb.njit(cache=True)
def gen_obs_grid_encoding(
    grid_state: ndarray[np.int_],
    agent_state: ndarray[np.int_],
    agent_view_size: ndarray[np.int_],
    see_through_walls: ndarray[np.bool_] | bool,
) -> ndarray[np.int_]:
    """
    Generate encoding for the sub-grid observed by an agent (including visibility mask).

    Parameters
    ----------
    grid_state : ndarray[int] of shape (width, height, grid_state_dim)
        Array representation for each grid object
    agent_state : ndarray[int] of shape (num_agents, agent_state_dim)
        Array representation for each agent
    agent_view_size : int
        Width and height of observation sub-grids
    see_through_walls : bool
        Whether the agent can see through walls
        Default=True based on current implementation

    Returns
    -------
    img : ndarray[int] of shape (num_agents, view_size...n_dims, encode_dim)
        Encoding of observed sub-grid for each agent
    """
    obs_grid = gen_obs_grids(grid_state, agent_state, agent_view_size)
    if isinstance(see_through_walls, bool):
        see_through_walls = [
            see_through_walls,
        ] * len(obs_grid)

    # Generate and apply visibility masks
    for agent, o_grid in enumerate(obs_grid):
        if not see_through_walls[agent]:
            vis_mask = get_vis_mask(o_grid)
            for mask_idx in np.ndindex(vis_mask.shape):
                if not vis_mask[mask_idx]:
                    obs_grid[agent][*mask_idx] = UNSEEN_ENCODING

    # vis_mask = get_vis_mask(obs_grid)
    # num_agents = len(agent_state)
    # for agent in range(num_agents):
    #     #TODO: Return for edits
    #     if not see_through_walls:
    #         for i in range(agent_view_size):
    #             for j in range(agent_view_size):
    #                 if not vis_mask[agent, i, j]:
    #                     obs_grid[agent, i, j] = UNSEEN_ENCODING

    return obs_grid


@nb.njit(cache=True)
def gen_obs_grid_vis_mask(
    grid_state: ndarray[np.int_],
    agent_state: ndarray[np.int_],
    agent_view_size: int,
) -> ndarray[np.int_]:
    """
    Generate visibility mask for the sub-grid observed by an agent.

    Parameters
    ----------
    grid_state : ndarray[int] of shape (width, height, grid_state_dim)
        Array representation for each grid object
    agent_state : ndarray[int] of shape (num_agents, agent_state_dim)
        Array representation for each agent
    agent_view_size : int
        Width and height of observation sub-grids

    Returns
    -------
    mask : ndarray[int] of shape (num_agents, view_size, view_size)
        Encoding of observed sub-grid for each agent
    """
    obs_grid = gen_obs_grids(grid_state, agent_state, agent_view_size)
    return get_vis_mask(obs_grid)


# @nb.njit(cache=True)
# def gen_obs_grid(
#     grid_state: ndarray[np.int_],
#     agent_state: ndarray[np.int_],
#     agent_view_size: int) -> ndarray[np.int_]:
#     """
#     Generate the sub-grid observed by each agent (WITHOUT visibility mask).

#     Parameters
#     ----------
#     grid_state : ndarray[int] of shape (width, height, grid_state_dim)
#         Array representation for each grid object
#     agent_state : ndarray[int] of shape (num_agents, agent_state_dim)
#         Array representation for each agent
#     agent_view_size : int
#         Width and height of observation sub-grids

#     Returns
#     -------
#     obs_grid : ndarray[int] of shape (num_agents, width, height, encode_dim)
#         Observed sub-grid for each agent
#     """
#     num_agents = len(agent_state)
#     obs_width, obs_height = agent_view_size, agent_view_size

#     # Process agent states
#     agent_grid_encoding = agent_state[..., AGENT_ENCODING_IDX]
#     agent_dir = agent_state[..., AGENT_DIR_IDX]
#     agent_pos = agent_state[..., AGENT_POS_IDX]
#     agent_terminated = agent_state[..., AGENT_TERMINATED_IDX]
#     agent_carrying_encoding = agent_state[..., AGENT_CARRYING_IDX]

#     print("Fin Process agents")

#     # Get grid encoding
#     if num_agents > 1:
#         grid_encoding = np.empty((*grid_state.shape[:-1], ENCODE_DIM), dtype=np.int_)
#         grid_encoding[...] = grid_state[..., GRID_ENCODING_IDX]

#         # Insert agent grid encodings
#         for agent in range(num_agents):
#             if not agent_terminated[agent]:
#                 i, j = agent_pos[agent]
#                 grid_encoding[i, j, GRID_ENCODING_IDX] = agent_grid_encoding[agent]
#     else:
#         grid_encoding = grid_state[..., GRID_ENCODING_IDX]

#     print("Fin Grid Encoding")

#     # Get top left corner of observation grids
#     top_left = get_view_exts(agent_dir, agent_pos, agent_view_size)
#     topX, topY = top_left[:, 0], top_left[:, 1]

#     print("Fin 'top left'")

#     # Populate observation grids
#     num_left_rotations = (agent_dir + 1) % 4
#     obs_grid = np.empty((num_agents, obs_width, obs_height, ENCODE_DIM), dtype=np.int_)
#     for agent in range(num_agents):
#         for i in range(0, obs_width):
#             for j in range(0, obs_height):
#                 # Absolute coordinates in world grid
#                 x, y = topX[agent] + i, topY[agent] + j

#                 # Rotated relative coordinates for observation grid
#                 if num_left_rotations[agent] == 0:
#                     i_rot, j_rot = i, j
#                 elif num_left_rotations[agent] == 1:
#                     i_rot, j_rot = j, obs_width - i - 1
#                 elif num_left_rotations[agent] == 2:
#                     i_rot, j_rot = obs_width - i - 1, obs_height - j - 1
#                 elif num_left_rotations[agent] == 3:
#                     i_rot, j_rot = obs_height - j - 1, i

#                 # Set observation grid
#                 if 0 <= x < grid_encoding.shape[0] and 0 <= y < grid_encoding.shape[1]:
#                     obs_grid[agent, i_rot, j_rot] = grid_encoding[x, y]
#                 else:
#                     obs_grid[agent, i_rot, j_rot] = WALL_ENCODING

#     print("Fin Pop Obs Grid")

#     # Make it so each agent sees what it's carrying
#     # We do this by placing the carried object at the agent position
#     # in each agent's partially observable view
#     obs_grid[:, obs_width // 2, obs_height - 1] = agent_carrying_encoding

#     print("Fin Final Obs Grid")

#     return obs_grid


# @nb.njit(cache=True)
def gen_obs_grids(
    grid_state: ndarray[np.int_],
    agent_state: ndarray[np.int_],
    agent_view_sizes: ndarray[np.int_],
) -> ndarray[np.int_]:
    """
    Generate the sub-grid observed by each agent (WITHOUT visibility mask).

    Parameters
    ----------
    grid_state : ndarray[int] of shape (width, height, grid_state_dim)
        Array representation for each grid object
    agent_state : ndarray[int] of shape (num_agents, agent_state_dim)
        Array representation for each agent
    agent_view_sizes : ndarray[int] of shape (num_agents)
        Width and height of observation sub-grids

    Returns
    -------
    obs_grid : ndarray[int] of shape (num_agents, width, height, encode_dim)
        Observed sub-grid for each agent
    """
    num_agents = len(agent_state)

    # Process agent states
    agent_grid_encoding = agent_state.encode
    agent_dir = agent_state.dir
    agent_pos = agent_state.pos
    agent_terminated = agent_state[..., AGENT_TERMINATED_IDX]
    # agent_carrying_encoding = agent_state[..., AGENT_CARRYING_IDX]

    # Get grid encoding
    if num_agents > 1:
        grid_encoding = np.empty(
            (*grid_state.shape[:-1], ENCODE_DIM), dtype=np.int_
        )
        grid_encoding[...] = grid_state[..., GRID_ENCODING_IDX]

        # Insert agent grid encodings
        for agent in range(num_agents):
            if not agent_terminated[agent]:
                pos = agent_pos[agent]
                grid_encoding[*pos] = agent_grid_encoding[agent]
    else:
        grid_encoding = grid_state[..., GRID_ENCODING_IDX]

    agent_obs = []
    agent_view_coords = get_view_coords(agent_dir, agent_pos, agent_view_sizes)

    # for a,view_coords in enumerate(agent_view_coords):
    for view_coords in agent_view_coords:
        # Create an empty view for the agent
        obs_grid = np.empty(
            (*view_coords.shape[:-1], ENCODE_DIM), dtype=np.int_
        )
        # For each view coordinate
        # set view = to grid value at that coordinate
        for obs_idx in np.ndindex(view_coords.shape[:-1]):
            grid_idx = view_coords[obs_idx]
            try:
                obs_grid[obs_idx] = grid_encoding[*grid_idx]
            except IndexError:
                # If out-of-bounds
                obs_grid[obs_idx] = WALL_ENCODING

        agent_obs.append(obs_grid)

    # TODO: Earlier version of letting the agent know that it is carrying something:
    # # Make it so each agent sees what it's carrying
    # # We do this by placing the carried object at the agent position
    # # in each agent's partially observable view
    # obs_grid[:, obs_width // 2, obs_height - 1] = agent_carrying_encoding

    return agent_obs


@nb.njit(cache=True)
def get_see_behind_mask(grid_array: ndarray[np.int_]) -> ndarray[np.int_]:
    """
    Return boolean mask indicating which grid locations can be seen through.

    Parameters
    ----------
    grid_array : ndarray[int] of shape (num_agents, width, height, dim)
        Grid object array for each agent

    Returns
    -------
    see_behind_mask : ndarray[bool] of shape (width, height)
        Boolean visibility mask
    """
    # num_agents, width, height = grid_array.shape[:3]
    # see_behind_mask = np.zeros((num_agents, width, height), dtype=np.bool_)
    # for agent in range(num_agents):
    #     for i in range(width):
    #         for j in range(height):
    #             see_behind_mask[agent, i, j] = see_behind(grid_array[agent, i, j])

    # see_behind_mask = list()
    # for obs_grid in grid_array:
    #     agent_los_mask = np.empty((*obs_grid.shape[:-1], ENCODE_DIM), dtype=np.int_)
    #     for obs_idx in np.ndindex(obs_grid.shape[:-1]):
    #         agent_los_mask[obs_idx] = see_behind(obs_grid[obs_idx])
    #     see_behind_mask.append(agent_los_mask)

    see_behind_mask = np.empty((*grid_array.shape[:-1],))
    # dtypes unsupported by numba
    # see_behind_mask = np.empty((*grid_array.shape[:-1],), dtype=np.int_)
    for obs_idx in np.ndindex(grid_array.shape[:-1]):
        see_behind_mask[obs_idx] = see_behind(grid_array[obs_idx])

    return see_behind_mask


@nb.njit(cache=True)
def get_vis_mask(obs_grid: ndarray[np.int_]) -> ndarray[np.bool_]:
    """
    Generate a boolean mask indicating which grid locations are visible to each agent.

    Parameters
    ----------
    obs_grid : ndarray[int] of shape (num_agents, width, height, dim)
        Grid object array for each agent observation

    Returns
    -------
    vis_mask : ndarray[bool] of shape (num_agents, width, height)
        Boolean visibility mask for each agent
    """
    # num_agents, width, height = obs_grid.shape[:3]
    see_behind_mask = get_see_behind_mask(obs_grid)
    # vis_mask = np.zeros((*obs_grid.shape[:-1],))

    # TODO: Vis Mask should mask obstructed views

    # for agent in range(num_agents):
    #     for j in range(height - 1, -1, -1):
    #         # Forward pass
    #         for i in range(0, width - 1):
    #             if vis_mask[agent, i, j] and see_behind_mask[agent, i, j]:
    #                 vis_mask[agent, i + 1, j] = True
    #                 if j > 0:
    #                     vis_mask[agent, i + 1, j - 1] = True
    #                     vis_mask[agent, i, j - 1] = True

    #         # Backward pass
    #         for i in range(width - 1, 0, -1):
    #             if vis_mask[agent, i, j] and see_behind_mask[agent, i, j]:
    #                 vis_mask[agent, i - 1, j] = True
    #                 if j > 0:
    #                     vis_mask[agent, i - 1, j - 1] = True
    #                     vis_mask[agent, i, j - 1] = True

    # return vis_mask
    return see_behind_mask


# @nb.njit(cache=True)
# def get_view_exts(
#     agent_dir: ndarray[np.int_],
#     agent_pos: ndarray[np.int_],
#     agent_view_size: int) -> ndarray[np.int_]:
#     """
#     Get the extents of the square set of grid cells visible to each agent.

#     Parameters
#     ----------
#     agent_dir : ndarray[int] of shape (num_agents,)
#         Direction of each agent
#     agent_pos : ndarray[int] of shape (num_agents, 2)
#         The (x, y) position of each agent
#     agent_view_size : int
#         Width and height of agent view

#     Returns
#     -------
#     top_left : ndarray[int] of shape (num_agents, 2)
#         The (x, y) coordinates of the top-left corner of each agent's observable view
#     """
#     agent_x, agent_y = agent_pos[:, 0], agent_pos[:, 1]
#     top_left = np.zeros((agent_dir.shape[0], 2), dtype=np.int_)

#     # Facing right
#     top_left[agent_dir == RIGHT, 0] = agent_x[agent_dir == RIGHT]
#     top_left[agent_dir == RIGHT, 1] = agent_y[agent_dir == RIGHT] - agent_view_size // 2

#     # Facing down
#     top_left[agent_dir == DOWN, 0] = agent_x[agent_dir == DOWN] - agent_view_size // 2
#     top_left[agent_dir == DOWN, 1] = agent_y[agent_dir == DOWN]

#     # Facing left
#     top_left[agent_dir == LEFT, 0] = agent_x[agent_dir == LEFT] - agent_view_size + 1
#     top_left[agent_dir == LEFT, 1] = agent_y[agent_dir == LEFT] - agent_view_size // 2

#     # Facing up
#     top_left[agent_dir == UP, 0] = agent_x[agent_dir == UP] - agent_view_size // 2
#     top_left[agent_dir == UP, 1] = agent_y[agent_dir == UP] - agent_view_size + 1

#     return top_left


# @nb.njit(cache=True)
def get_view_exts(*args) -> DeprecationWarning:
    raise DeprecationWarning(
        "get_view_exts is deprecated. Use 'get_view_coords'"
    )


# @nb.njit(cache=True)
def get_view_coords(
    agent_dir: ndarray[np.int_],
    agent_pos: ndarray[np.int_],
    agent_view_sizes: ndarray[np.int_],
) -> ndarray[np.int_]:
    """
    Gets the coordinates within each agent's visual range.

    Parameters
    ----------
    agent_dir : ndarray[int] of shape (num_agents, n_dims)
        Direction of each agent
    agent_pos : ndarray[int] of shape (num_agents, n_dims, 2)
        The (...,d_n) position of each agent
    agent_view_sizes : ndarray[int] of shape (num_agents)
        Width and height of each agent view

    Returns
    -------
    coords : ndarray[int] of shape (num_agents, n_dims, 2)
        The enumeration of (...,d_n) coordinates of each agent's observable view
    """
    # Recover N_dims
    N = agent_dir.shape[-1]
    half_views = np.floor_divide(agent_view_sizes, 2)
    # Relative View Coords
    view_shapes = [
        np.array(
            (
                list(
                    # TODO: itertools.product interferes with njit
                    # a stack exchange post suggests that even so, product is faster
                    product(range(-half, half + 1), repeat=N)
                )
            )
        )
        for half in half_views
    ]
    # Offset View Coords based on agent position and orientation
    offsets = [
        pos + np.multiply(dir, half)
        for dir, half, pos in zip(agent_dir, half_views, agent_pos)
    ]
    coords = [view + offset for view, offset in zip(view_shapes, offsets)]

    oriented_coords = []
    # for a_coord,a_dir in zip(coords,agent_dir):
    for i, (a_coord, a_dir) in enumerate(zip(coords, agent_dir)):
        oriented = a_coord.reshape((*(agent_view_sizes[i],) * N, N))
        if not isinstance(a_dir, Iterable):
            raise TypeError(f"Unsupported dir type: {type(a_dir)}")
        _k = 1
        if a_dir[0] < 0:
            oriented = np.rot90(oriented, k=2, axes=(0, 1))
            a_dir = a_dir * -1
        # For each dir[0:] !=0 rotate,
        for ax, d_0n in enumerate(a_dir[1:]):
            if d_0n < 0:
                oriented = np.rot90(oriented, k=_k, axes=(0, ax + 1))
            if d_0n > 0:
                oriented = np.rot90(oriented, k=-_k, axes=(0, ax + 1))

        oriented_coords.append(oriented)

    return oriented_coords
