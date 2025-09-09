from __future__ import annotations

import gymnasium as gym
import numpy as np
import pygame

from collections.abc import Iterable, Sequence
from gymnasium import spaces
from itertools import repeat
from typing import Any, Callable, Literal, SupportsFloat, TypeAlias

from .core.actions import ActionSpec, OrthogonalActionSpec
from .core.agent import Agent, AgentState
from .core.constants import TILE_PIXELS, Type, Direction
from .core.space import NDSpace
from .core.mission import MissionSpace
from .core.world_object import WorldObj, Goal, Lava
from .utils.obs import gen_obs_grid_encoding
from .utils.random import RandomMixin
### Typing

AgentID: TypeAlias = int
ObsType: TypeAlias = dict[str, Any]

### Environment


class HyperGridEnv(gym.Env, RandomMixin):
    """
    Base class for multi-agent n-D gridworld environments.

    :Agents:

        The environment can be configured with any fixed number of agents.
        Agents are represented by :class:`.Agent` instances, and are
        identified by their index, from ``0`` to ``len(env.agents) - 1``.

    :Observation Space:

        The multi-agent observation space is a Dict mapping from agent index to
        corresponding agent observation space.

        The standard agent observation is a dict with the following entries:

            * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
                Encoding of the agent's view of the environment,
                where each grid object is encoded as a 3 dimensional tuple:
                (:class:`.Type`, :class:`.Color`, :class:`.State`)
            * direction : int
                Agent's direction (0: right, 1: down, 2: left, 3: up)
            * mission : Mission
                Task string corresponding to the current environment configuration

    :Action Space:

        The multi-agent action space is a Dict mapping from agent index to
        corresponding agent action space.

        Agent actions are discrete integers, as enumerated in :class:`.Action`.

    Attributes
    ----------
    agents : list[Agent]
        List of agents in the environment
    grid : Grid
        Environment grid
    observation_space : spaces.Dict[AgentID, spaces.Space]
        Joint observation space of all agents
    action_space : spaces.Dict[AgentID, spaces.Space]
        Joint action space of all agents
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        mission_space: MissionSpace | str = "maximize reward",
        agents: Iterable[Agent] | int = 1,
        dims: int | Sequence[int] | None = None,
        n_dims: int | None = 2,
        grid_size: int | None = 10,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        allow_agent_overlap: bool = True,
        joint_reward: bool = False,
        reward_range: Iterable[int] = (0, 1),
        cooperative_task: bool = False,
        success_termination_mode: Literal["any", "all"] = "any",
        failure_termination_mode: Literal["any", "all"] = "all",
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: Direction | None = None,
        agent_action_spec: ActionSpec | None = OrthogonalActionSpec,
        **kwargs,
    ):
        """
        Parameters
        ----------
        mission_space : MissionSpace
            Space of mission strings (i.e. agent instructions)
        agents : int or Iterable[Agent]
            Number of agents in the env (or provide :class:`Agent` instances)
        grid_size : int
            Size of the environment grid
        n_dims : int
            Number of dimensions of grid_size
        dims : [int]
            Array of size of each dimension
        max_steps : int
            Maximum number of steps per episode
        see_through_walls : bool
            Whether agents can see through walls
        agent_view_size : int
            Size of agent view (must be odd)
        allow_agent_overlap : bool
            Whether agents are allowed to overlap
        joint_reward : bool
            Whether all agents receive the same joint reward
        success_termination_mode : 'any' or 'all'
            Whether to terminate when any agent completes its mission
            or when all agents complete their missions
        failure_termination_mode : 'any' or 'all'
            Whether to terminate when any agent fails its mission
            or when all agents fail their missions
        costs : dict[int]
        render_mode : str
            Rendering mode (human or rgb_array)
        screen_size : int
            Width and height of the rendering window (in pixels)
        highlight : bool
            Whether to highlight the view of each agent when rendering
        tile_size : int
            Width and height of each grid tiles (in pixels)
        """
        gym.Env.__init__(self)
        RandomMixin.__init__(self, self.np_random)

        # Initialize mission space
        if isinstance(mission_space, str):
            self.mission_space = MissionSpace.from_string(mission_space)
        else:
            self.mission_space = mission_space

        # Initialize Grid
        if isinstance(dims, Sequence):
            # If [d_0,d_1,...]
            self.dims = dims
            self.n_dims = len(dims)
        else:
            # If |N| and d
            if isinstance(dims, int) and dims > 0:
                self.n_dims = dims
            elif isinstance(n_dims, int) and n_dims > 0:
                self.n_dims = n_dims
            else:
                raise ValueError("Invalid or missing n_dims or dims parameter")
            # Check for Grid size
            if not isinstance(grid_size, int) or (grid_size < 3):
                grid_size = 10
                raise UserWarning("Invalid Grid Size... Setting to 10")
            # {d_n} for n in N
            self.dims = [grid_size for _ in range(self.n_dims)]
        # Make the space from {d_n}_{n \in N}
        self.grid: NDSpace = NDSpace(self.dims)

        # Initialize agents

        # Given a list of agents
        if isinstance(agents, Iterable) and all(
            (isinstance(a, Agent) for a in agents)
        ):
            self.num_agents = len(agents)
            self.agent_states = AgentState(
                self.num_agents, spatial_ndim=self.n_dims
            )
            self.agents: list[Agent] = sorted(
                agents, key=lambda agent: agent.index
            )
            for agent in self.agents:
                # copy to joint agent state
                self.agent_states[agent.index] = agent.state
                # reference joint agent state
                agent.state = self.agent_states[agent.index]

            # Pre-cache view sizes, and xray vision, ignoring input
            self.agent_view_sizes = np.array([a.view_size for a in self.agents])
            self.agent_see_through_walls = np.array(
                [a.see_through_walls for a in self.agents]
            )

        elif isinstance(agents, int) and (agents > 0):
            self.num_agents = agents

            # Set view sizes
            if isinstance(agent_view_size, int):
                self.agent_view_sizes = np.array(
                    # [*(agent_view_size,)*self.num_agents] )
                    list(repeat(agent_view_size, self.num_agents))
                )
            elif isinstance(agent_view_size, Sequence):
                self.agent_view_sizes = np.array(agent_view_size)
            else:
                raise TypeError(
                    "Invalid arg for agent_view_size, expect int or seq[int]"
                )

            # Set seeing through walls
            if isinstance(see_through_walls, bool):
                self.agent_see_through_walls = np.array(
                    # [*(see_through_walls,)*self.num_agents] )
                    list(repeat(see_through_walls, self.num_agents))
                )
            elif isinstance(see_through_walls, Sequence):
                self.agent_see_through_walls = np.array(see_through_walls)
            else:
                raise TypeError(
                    "Invalid arg for see_through_walls, expect bool or seq[bool]"
                )

        else:
            raise ValueError(f"Invalid argument for agents: {agents}")

        # Given the number of agents, |I|
        if isinstance(agents, int) and (agents > 0):
            # joint agent state (vectorized)
            self.agent_states = AgentState(agents, spatial_ndim=self.n_dims)
            self.agents: list[Agent] = []
            for i in range(agents):
                agent = Agent(
                    index=i,
                    spatial_ndim=self.n_dims,
                    mission_space=self.mission_space,
                    view_size=self.agent_view_sizes[i],
                    see_through_walls=self.agent_see_through_walls[i],
                    action_spec=agent_action_spec,
                )
                agent.state = self.agent_states[i]
                self.agents.append(agent)

        self.agent_action_space = agent_action_spec
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.step_count = 0

        # Other Attributes
        assert isinstance(max_steps, int), (
            f"The argument max_steps must be an integer, got: {type(max_steps)}"
        )
        self.max_steps = max_steps
        self.success_termination_mode = success_termination_mode
        self.failure_termination_mode = failure_termination_mode
        self.joint_reward = joint_reward
        self.reward_range = reward_range
        self.allow_agent_overlap = allow_agent_overlap
        self.cooperative_task = cooperative_task

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov
        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

    # End __init__

    @property
    def observation_space(self) -> spaces.Dict[AgentID, spaces.Space]:
        """
        Return the joint observation space of all agents.
        """
        return spaces.Dict(
            {agent.index: agent.observation_space for agent in self.agents}
        )

    @property
    def action_space(self) -> spaces.Dict[AgentID, spaces.Space]:
        """
        Return the joint action space of all agents.
        """
        return spaces.Dict(
            {agent.index: agent.action_space for agent in self.agents}
        )

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> tuple[dict[AgentID, ObsType] : dict[AgentID, dict[str, Any]]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int or None
            Seed for random number generator

        Returns
        -------
        observations : dict[AgentID, ObsType]
            Observation for each agent
        infos : dict[AgentID, dict[str, Any]]
            Additional information for each agent
        """
        super().reset(seed=seed, **kwargs)
        # Reset agents
        self.mission_space.seed(seed)
        self.mission = self.mission_space.sample()
        self.agent_states = AgentState(
            self.num_agents, spatial_ndim=self.n_dims
        )
        for agent in self.agents:
            agent.state = self.agent_states[agent.index]
            agent.reset(mission=self.mission)

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.dims)

        # These fields should be defined by _gen_grid
        assert np.all(self.agent_states.pos >= 0)
        assert np.all(self.agent_states.dir >= -1)

        # Check that agents don't overlap with other objects
        for agent in self.agents:
            start_cell = self.grid.get(agent.state.pos)
            assert start_cell is None or start_cell.can_overlap()

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        observations = self.gen_obs()

        # Render environment
        if self.render_mode == "human":
            self.render()

        infos = dict(enumerate(repeat({}, self.num_agents)))

        return observations, infos

    def step(
        self,
        actions: dict[AgentID, Sequence[int]],
        infos: dict[AgentID, dict] = None,
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, SupportsFloat],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        """
        Run one timestep of the environment's dynamics
        using the provided agent actions.

        Parameters
        ----------
        actions : dict[AgentID, Action]
            Action for each agent acting at this timestep

        Returns
        -------
        observations : dict[AgentID, ObsType]
            Observation for each agent
        rewards : dict[AgentID, SupportsFloat]
            Reward for each agent
        terminations : dict[AgentID, bool]
            Whether the episode has been terminated for each agent (success or failure)
        truncations : dict[AgentID, bool]
            Whether the episode has been truncated for each agent (max steps reached)
        infos : dict[AgentID, dict[str, Any]]
            Additional information for each agent
        """
        self.step_count += 1
        infos = infos or dict(enumerate(repeat({}, self.num_agents)))
        terminations = dict(enumerate(self.agent_states.terminated))
        rewards, infos = self._handle_actions(actions, terminations, infos)
        observations = self.gen_obs(infos)
        truncated = self.step_count >= self.max_steps
        truncations = dict(enumerate(repeat(truncated, self.num_agents)))

        # Rendering
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def gen_obs(
        self,
        infos: dict[AgentID, dict] = {},
    ) -> dict[AgentID, ObsType]:
        """
        Generate observations for each agent
        (partially observable, low-res encoding).

        Returns
        -------
        observations : dict[AgentID, ObsType]
            Mapping from agent ID to observation dict, containing:
                * 'image': partially observable view of the environment
                * 'direction': agent's direction / orientation
                * 'mission': textual mission string (instructions for the agent)
        """
        direction = self.agent_states.dir
        image = gen_obs_grid_encoding(
            self.grid.state,
            self.agent_states,
            self.agent_view_sizes,
            self.agent_see_through_walls,
        )
        observations = {}
        for i in range(self.num_agents):
            observations[i] = {
                "image": image[i],
                "direction": direction[i],
                # "mission": self.agents[i].mission,
            }
        return observations

    def place_obj(
        self,
        obj: WorldObj | None,
        top: tuple[int, int] = None,
        size: tuple[int, int] = None,
        reject_fn: Callable[[HyperGrid, tuple[int, int]], bool] | None = None,
        max_tries=50,
    ) -> tuple[int, int]:
        """
        Place an object at an empty position in the grid.

        Parameters
        ----------
        obj: WorldObj
            Object to place in the grid
        top: tuple[int, int]
            Top-left position of the rectangular area where to place the object
        size: tuple[int, int]
            Width and height of the rectangular area where to place the object
        reject_fn: Callable(env, pos) -> bool
            Function to filter out potential positions
        max_tries: int
            Maximum number of attempts to place the object
        """
        if top is None:
            top = (0,) * self.n_dims
        else:
            top = np.array(list(max(i, 0) for i in top))

        if size is None:
            size = self.dims

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError(
                    "Rejection sampling failed in place_obj, likely "
                    + "no valid placement candidate in provided range."
                )
            num_tries += 1
            pos = [
                np.random.randint(top[i], min(top[i] + size[i], d))
                for i, d in enumerate(self.dims)
            ]
            # Don't place the object on top of another object
            if self.grid.get(pos) is not None:
                continue
            # Don't place the object where agents are
            if np.bitwise_and.reduce(
                self.agent_states.pos == pos, axis=1
            ).any():
                continue
            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue
            break

        self.grid.set(pos, obj)
        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos
        return pos

    def place_agent(
        self,
        agent: Agent,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=50,
    ) -> tuple[int, int]:
        """
        Set agent starting point at an empty position in the grid.
        """
        agent.state.pos = (-1,) * self.n_dims
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        agent.state.pos = pos
        if rand_dir:
            _dir = np.zeros(self.n_dims)
            i = self._rand_int(0, self.n_dims)
            _dir[i] = 1 if self._rand_bool else -1
            agent.state.dir = _dir
        return pos

    def put_obj(self, obj: WorldObj, index: Sequence[int]):
        """
        Put an object at a specific position in the grid.
        """
        self.grid.set(index, obj)
        obj.init_pos = index
        obj.cur_pos = index

    def is_done(self) -> bool:
        """
        Return whether the current episode is finished (for all agents).
        """
        truncated = self.step_count >= self.max_steps
        return truncated or all(self.agent_states.terminated)

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success.
        """
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def close(self):
        """
        Close the rendering window.
        """
        if self.window:
            pygame.quit()

    # -----  Common Methods for extended environments ----- #

    def _gen_grid(self, dims: Sequence[int]):
        """
        :meta public:

        Generate the grid for a new episode.

        This method should:

        * Set ``self.grid`` and populate it with :class:`.WorldObj` instances
        * Set the positions and directions of each agent

        Default behavior is to populate a mostly empty space randomly.
        The space will have a wall enclosing the space,
        a single goal will be placed in the corner associate with the max
        value for each dim.

        Parameters
        ----------
        dims : Sequence[int]
            List of domains for each dimension of the environment
        """
        self.grid = NDSpace(dims)
        # Generate the surrounding walls
        self.grid.make_boundary()
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), [dn - 2 for dn in self.dims])
        # Place the agent
        for agent in self.agents:
            if (
                self.agent_start_pos is not None
                and self.agent_start_dir is not None
            ):
                agent.state.pos = self.agent_start_pos
                agent.state.dir = self.agent_start_dir
            else:
                self.place_agent(agent)

    def _on_success(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
        infos: dict[AgentID, dict] = None,
    ):
        """
        Callback for when an agent completes its mission.

        Default gives a reward (shared or otherwise),
        then terminates the agent(s) as applicable.

        Parameters
        ----------
        agent : Agent
            Agent that completed its mission
        rewards : dict[AgentID, SupportsFloat]
            Reward dictionary to be updated
        terminations : dict[AgentID, bool]
            Termination dictionary to be updated
        """
        if self.success_termination_mode == "any":
            self.agent_states.terminated = True  # terminate all agents
            for i in range(self.num_agents):
                terminations[i] = True
        else:
            agent.state.terminated = True  # terminate this agent only
            terminations[agent.index] = True

        if self.joint_reward:
            for i in range(self.num_agents):
                rewards[i] = self._reward()  # reward all agents
        else:
            rewards[agent.index] = self._reward()  # reward this agent only
        return terminations

    def _on_failure(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool] = {},
        infos: dict[AgentID, dict] = None,
    ):
        """
        Callback for when an agent fails its mission prematurely.

        Default terminates agent(s) for failure.

        Parameters
        ----------
        agent : Agent
            Agent that failed its mission
        rewards : dict[AgentID, SupportsFloat]
            Reward dictionary to be updated
        terminations : dict[AgentID, bool]
            Termination dictionary to be updated
        """
        if self.failure_termination_mode == "any":
            self.agent_states.terminated = True  # terminate all agents
            for i in range(self.num_agents):
                terminations[i] = True
        else:
            agent.state.terminated = True  # terminate this agent only
            terminations[agent.index] = True
        return terminations

    def _update_orientation(
        self,
        agent: Agent,
        orientation: Sequence[int],
        infos: dict[AgentID, dict] = None,
    ):
        """
        :meta public:

        Update the agent's orientation.

        Override this method to add constraints or checks as desired.

        Parameters
        ----------
        agent : Agent
            Agent that failed its mission
        orientation : sequence[int]
            The desired new orientation for the agent
        """
        agent.state.dir = orientation

    def _move_agent(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        infos: dict[AgentID, dict] = None,
    ):
        """
        :meta public:

        Attempt to move the agent forward by one cell.

        Parameters
        ----------
        agent : Agent
            Agent that failed its mission
        rewards : dict[AgentID, SupportsFloat]
            Reward dictionary to be updated
        """
        dest_coords = agent.state.pos + agent.state.dir
        dest_obj = self.grid.get(dest_coords)
        # Check for environmental collision
        if dest_obj is not None and not dest_obj.can_overlap():
            rewards[agent.index] -= agent.cost_collision_env
            return
        # Check for agent collision
        if not self.allow_agent_overlap:
            agent_present = np.bitwise_and.reduce(
                self.agent_states.pos == dest_coords, axis=1
            ).any()
            if agent_present:
                rewards[agent.index] -= agent.cost_collision_agent
                return
        # Move Agent
        agent.state.pos = dest_coords
        return

    def _occupation_effects(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool] = {},
        infos: dict[AgentID, dict] = None,
    ):
        """
        :meta public:

        Callback for effects related to the cells that the agent occupy.

        Default is to call failure after entering a lava cell

        Parameters
        ----------
        agent : Agent
            Agent that failed its mission
        rewards : dict[AgentID, SupportsFloat]
            Reward dictionary to be updated
        """
        loc = agent.pos
        if isinstance(self.grid.get(loc), Lava):
            self._on_failure(agent, rewards, terminations, infos)
        if isinstance(self.grid.get(loc), Goal):
            self._on_success(agent, rewards, terminations, infos)

    def _agent_interaction(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        infos: dict[AgentID, dict] = None,
    ):
        """
        :meta public:

        Callback for handling the agent interaction flag.

        Default subtracts the interaction cost, then calls
        a success if the agent if facing the goal.

        Parameters
        ----------
        agent : Agent
            Agent that failed its mission
        rewards : dict[AgentID, SupportsFloat]
            Reward dictionary to be updated
        """
        # Cost of action
        rewards[agent.index] -= agent.cost_interaction
        # Perform interaction
        fwd_pos = agent.state.pos + agent.state.dir
        fwd_obj = self.grid.get(fwd_pos)
        if type(fwd_obj) is Type.goal:
            self.on_success(agent, rewards)

    def _cooperative_interactions(
        self,
        actions: dict[AgentID, Sequence[int]],
        rewards: dict[AgentID, SupportsFloat],
        infos: dict[AgentID, dict] = None,
    ):
        """
        :meta public:

        A callback to handle interactions for all of the agents together.

        This is intended for environments in which multiple agent must
        interact with an objective for success.

        Default does does nothing, super(). will return a filtered dict
        of just the interaction values.

        Parameters
        ----------
        actions : dict[AgentID, Sequence[int]]
        rewards : dict[AgentID, SupportsFloat]
        """
        actors = {}
        for i, action in actions.items():
            if action["interact"]:
                agent = self.agents[i]
                actors[i] = agent
        return actors

    def _handle_actions(
        self,
        actions: dict[AgentID, Sequence[int]],
        terminations: dict[AgentID, bool] = {},
        infos: dict[AgentID, dict] = None,
    ) -> dict[AgentID, SupportsFloat]:
        """
        :meta public:

        How to handle actions taken by agents in this env.

        This method should:

        * Instantiate a rewards variable
        * Determine multi-agent action order
        * Then for each agent:
            * Use action_spec.to_dict to change from array to dict
            * Execute the aspects of the agent's action in some order:
                * Update to orientation
                * Move the agent
                * Check for effects in new cell
                * Perform agent's interaction effect
        * If cooperation is a feature, use it here.
            May be before the per/agent parts if desired.

        Default behavior:
        Does the above requirements, calling default corresponding
        private methods.

        Parameters
        ----------
        actions : dict[AgentID, Sequence[int]]
            Action for each agent acting at this timestep

        Returns
        -------
        rewards : dict[AgentID, SupportsFloat]
            Reward for each agent
        """
        rewards = {agent_index: 0 for agent_index in range(self.num_agents)}

        # Standardize Action
        actions = {
            i: self.agents[i].action_spec.to_dict(a) for i, a in actions.items()
        }

        # Randomize agent action order
        if self.num_agents == 1:
            order = (0,)
        else:
            order = self.np_random.random(size=self.num_agents).argsort()

        # Update agent states, grid states, and reward from actions
        for i in order:
            if i not in actions:
                continue
            agent, action = self.agents[i], actions[i]
            if agent.state.terminated:
                continue

            # # Standardize Action
            # action = agent.action_spec.to_dict(action)

            # Update Orientation
            self._update_orientation(agent, action["orient"], infos)

            # Move
            for _ in range(action["move"]):
                # attempt to move through each spot in dir one at a time
                self._move_agent(agent, rewards, infos)
                self._occupation_effects(agent, rewards, terminations, infos)

            # Interact
            if action["interact"]:
                self._agent_interaction(agent, rewards, infos)

        # Evaluate Group Interactions
        if self.cooperative_task:
            self._cooperative_interactions(actions, rewards, infos)

        return rewards, infos

    # -------------------------- Render Functions: -------------------------- #

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        # Map of object types to short string
        SYMBOLS = {
            "unseen": " ",
            "empty": "\u00b7",
            "wall": "\u2588",  # █
            "goal": "\u25c8",  # ◆
        }
        OBJECT_TO_ASCII = {
            i: SYMBOLS[o] if o in SYMBOLS.keys() else " "
            for i, o in enumerate(Type)
        }
        # "floor", "door", "key", "ball", "box", "goal", "lava",

        if self.n_dims > 2:
            # No support for higher dims yet.
            return ""

        # Support for printing 2D
        # dir_to_string = np.array([
        #     ['↖','↑','↗'],
        #     ['←','o','→'],
        #     ['↙','↓','↘']]).T
        # dir_to_uniglyph = np.array([
        #     ['◤','▲','◥'],
        #     ['◀','◬','▶'],
        #     ['◣','▼','◢']]).T
        dir_to_unicode = np.array(
            [
                ["\u25e4", "\u25b2", "\u25e5"],
                ["\u25c0", "\u25ec", "\u25b6"],
                ["\u25e3", "\u25bc", "\u25e2"],
            ]
        ).T

        # # Map agent's direction to short string
        # AGENT_DIR_TO_STR = {0: '>', 1: 'V', 2: '<', 3: '^'}

        # # Get agent locations
        # location_to_agent = {tuple(agent.pos): agent for agent in self.agents}

        # Copy grid size
        view = np.empty_like(self.grid.state.T[0], dtype=str)
        # Change objects to ASCII
        for i, v in np.ndenumerate(self.grid.state.T[0]):
            view[i] = OBJECT_TO_ASCII[v]
        # Insert Agents
        for a in self.agent_states:
            if not a.terminated:
                view.T[*a.pos] = dir_to_unicode[*([1, 1] + a.dir)]
        # Convert to string and return
        output = ""
        for row in view:
            output += str().join(row)
            output += "\n"

        return output


class HyperGrid(HyperGridEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -------------------------- Render Functions: -------------------------- #

    def get_pov_render(self, *args, **kwargs):
        """
        Render an agent's POV observation for visualization.
        """
        raise NotImplementedError(
            "POV rendering not supported for multiagent environments."
        )

    #     def get_full_render(self, highlight: bool, tile_size: int):
    #         """
    #         Render a non-partial observation for visualization.
    #         """
    #         # Compute agent visibility masks
    #         obs_shape = self.agents[0].observation_space['image'].shape[:-1]
    #         vis_masks = np.zeros((self.num_agents, *obs_shape), dtype=bool)
    #         for i, agent_obs in self.gen_obs().items():
    #             vis_masks[i] = (agent_obs['image'][..., 0] != Type.unseen.to_index())

    #         # Mask of which cells to highlight
    #         highlight_mask = np.zeros((self.width, self.height), dtype=bool)

    #         for agent in self.agents:
    #             # Compute the world coordinates of the bottom-left corner
    #             # of the agent's view area
    #             f_vec = agent.state.dir.to_vec()
    #             r_vec = np.array((-f_vec[1], f_vec[0]))
    #             top_left = (
    #                 agent.state.pos
    #                 + f_vec * (agent.view_size - 1)
    #                 - r_vec * (agent.view_size // 2)
    #             )

    #             # For each cell in the visibility mask
    #             for vis_j in range(0, agent.view_size):
    #                 for vis_i in range(0, agent.view_size):
    #                     # If this cell is not visible, don't highlight it
    #                     if not vis_masks[agent.index][vis_i, vis_j]:
    #                         continue

    #                     # Compute the world coordinates of this cell
    #                     abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

    #                     if abs_i < 0 or abs_i >= self.width:
    #                         continue
    #                     if abs_j < 0 or abs_j >= self.height:
    #                         continue

    #                     # Mark this cell to be highlighted
    #                     highlight_mask[abs_i, abs_j] = True

    #         # Render the whole grid
    #         img = self.grid.render(
    #             tile_size,
    #             agents=self.agents,
    #             highlight_mask=highlight_mask if highlight else None,
    #         )

    #         return img

    #     def get_frame(
    #         self,
    #         highlight: bool = True,
    #         tile_size: int = TILE_PIXELS,
    #         agent_pov: bool = False) -> ndarray[np.uint8]:
    #         """
    #         Returns an RGB image corresponding to the whole environment.

    #         Parameters
    #         ----------
    #         highlight: bool
    #             Whether to highlight agents' field of view (with a lighter gray color)
    #         tile_size: int
    #             How many pixels will form a tile from the NxM grid
    #         agent_pov: bool
    #             Whether to render agent's POV or the full environment

    #         Returns
    #         -------
    #         frame: ndarray of shape (H, W, 3)
    #             A frame representing RGB values for the HxW pixel image
    #         """
    #         if agent_pov:
    #             return self.get_pov_render(tile_size)
    #         else:
    #             return self.get_full_render(highlight, tile_size)

    def render(self):
        """
        Render the environment.
        """
        if len(self.n_dims) > 2:
            self.render_mode = None
            raise RuntimeWarning("Render not yet supported for >2D")
        pass


#         img = self.get_frame(self.highlight, self.tile_size)

#         if self.render_mode == 'human':
#             img = np.transpose(img, axes=(1, 0, 2))
#             screen_size = (
#                 self.screen_size * min(img.shape[0] / img.shape[1], 1.0),
#                 self.screen_size * min(img.shape[1] / img.shape[0], 1.0),
#             )
#             if self.render_size is None:
#                 self.render_size = img.shape[:2]
#             if self.window is None:
#                 pygame.init()
#                 pygame.display.init()
#                 pygame.display.set_caption(f'multigrid - {self.__class__.__name__}')
#                 self.window = pygame.display.set_mode(screen_size)
#             if self.clock is None:
#                 self.clock = pygame.time.Clock()
#             surf = pygame.surfarray.make_surface(img)

#             # Create background with mission description
#             offset = surf.get_size()[0] * 0.1
#             # offset = 32 if self.agent_pov else 64
#             bg = pygame.Surface(
#                 (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
#             )
#             bg.convert()
#             bg.fill((255, 255, 255))
#             bg.blit(surf, (offset / 2, 0))

#             bg = pygame.transform.smoothscale(bg, screen_size)

#             font_size = 22
#             text = str(self.mission)
#             font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
#             text_rect = font.get_rect(text, size=font_size)
#             text_rect.center = bg.get_rect().center
#             text_rect.y = bg.get_height() - font_size * 1.5
#             font.render_to(bg, text_rect, text, size=font_size)

#             self.window.blit(bg, (0, 0))
#             pygame.event.pump()
#             self.clock.tick(self.metadata['render_fps'])
#             pygame.display.flip()

#         elif self.render_mode == 'rgb_array':
#             return img
