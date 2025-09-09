import numpy as np
from numpy.typing import NDArray
from collections.abc import Sequence

from ..hypergrid_env import AgentID
from ..core.actions import OrthogonalActionSpec
from ..core.constants import Color
from ..utils.wrappers import OneHotObsWrapper
from .foraging import ForagingEnv
from ..core.world_object import WorldObj, Goal
from ..core.constants import Type as WO_Types
from ..core.space import NDSpace
from typing import override
from random import choice as random_choice


# class SensorSuiteEnv(HyperGridEnv):
class SensorSuiteUnwrapped(ForagingEnv):
    def __init__(
        self,
        agent_sensors: dict[AgentID, NDArray] = None,
        full_visibility: bool = False,
        remove_agents: list[AgentID] = None,
        record_visibility_on_success: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        agent_sensors : dict[AgentID, NDArray[np.bool_]]
            Each agent's boolean mask corresponding
            their ability to observe each color channel.
        """
        super().__init__(
            agent_action_spec=OrthogonalActionSpec, fixed_level=True, **kwargs
        )
        self.removed_agents = remove_agents
        self.rec_goal_vis = record_visibility_on_success
        # Override agent color cycling
        self.team_color = Color.yellow
        self.agent_states[:, self.agent_states.COLOR_IDX] = self.team_color

        # Restricted color cycle
        self.vis_channels = [c for c in Color]
        self.vis_channels.remove(Color.grey)
        self.vis_channels.remove(Color.clear)
        self.vis_channels.remove(self.team_color)

        # Ingest agent sensor param
        if full_visibility or (agent_sensors is None):
            self.full_visibility = True
        else:
            self.full_visibility = False
            # Define the sensors:
            # Provided one mask for every agent.
            if len(agent_sensors) == len(Color):
                self.agent_sensors = {
                    i: np.array(agent_sensors) for i in range(self.num_agents)
                }
            # Provided one mask per agent.
            elif len(agent_sensors) == self.num_agents:
                if isinstance(agent_sensors, dict):
                    self.agent_sensors = {
                        i: np.array(v) for i, v in agent_sensors.items()
                    }
                else:
                    self.agent_sensors = {
                        i: np.array(v) for i, v in enumerate(agent_sensors)
                    }
            # Provided partial dict of some agents.
            elif isinstance(agent_sensors, dict):
                self.agent_sensors = {
                    i: np.array(v) for i, v in agent_sensors.items()
                }
                for i in range(4):
                    if i not in agent_sensors.keys():
                        self.agent_sensors[i] = np.ones(len(Color))
            # Provided something else
            else:
                raise ValueError(
                    "Unrecognized `agent_sensors` input shape. Expecting: "
                    f"one mask of length {len(Color)} to be used all agents, "
                    f"or list of masks for {self.num_agents} agents."
                )

            # Check the sensor masks are correct length
            if any([len(v) != len(Color) for v in self.agent_sensors.values()]):
                raise ValueError(
                    "One or more supplied masks of incorrect size. "
                    f"All masks should be of length {len(Color)}"
                )
            # Check if sensors are full
            if np.all(list(self.agent_sensors.values())):
                self.full_visibility = True

    def _next_color(self):
        return random_choice(self.vis_channels)

    # Cycle color of goals when instantiated
    @override
    def _gen_grid(self, dims: Sequence[int]):
        self.grid = NDSpace(dims)
        self.grid.make_boundary()

        # Place a goals randomly
        for i in range(self.num_food):
            self.food_loc[i] = self.place_obj(Goal(color=self._next_color()))

        for agent in self.agents:
            if (
                self.agent_start_pos is not None
                and self.agent_start_dir is not None
            ):
                agent.state.pos = self.agent_start_pos
                agent.state.dir = self.agent_start_dir
            else:
                self.place_agent(agent)

        self.agent_levels = np.ones(self.num_agents)
        self.food_levels = np.ones(self.num_food)

    # Cycle color of goals when redeployed
    @override
    def _redeploy_goal(self, food_ind):
        # Redeploy goal (preventing respawn at same location)
        new_loc = self.place_obj(Goal(color=self._next_color()))
        if not new_loc:
            raise RuntimeError(f"New location error: {new_loc=}")
        self.grid.set(self.food_loc[food_ind], WorldObj.empty())
        self.food_loc[food_ind] = new_loc
        self.food_levels[food_ind] = random_choice((1, 2))

    # Track visibility of agent/goal
    @override
    def _on_success(
        self,
        food_ind: int,
        group: list[AgentID],
        infos: dict[AgentID, dict] = None,
        **kwargs,
    ):
        if self.rec_goal_vis:
            for i in group:
                food_loc = self.food_loc[food_ind]
                food_color = self.grid.get(food_loc).color
                vis_index = Color.to_index(food_color)
                infos[i]["vis_of_target"] = vis_index
        return super()._on_success(food_ind, group, infos=infos, **kwargs)

    # --- Eval support funcs --- #

    def reset(self, seed=None, **kwargs):
        outs = super().reset(seed, **kwargs)
        if self.removed_agents:
            for agent in self.removed_agents:
                self.remove_agent(agent)
        return outs

    def remove_agent(self, agent_idx):
        """Soft removes agent from environment"""
        self.agent_states[agent_idx].terminated = True
        self.agent_states[agent_idx].pos = [-1, -1]


class SensorSuiteEnv(OneHotObsWrapper):
    COLOR_SLICE = slice(len(WO_Types), len(WO_Types) + len(Color))

    def __init__(self, **kwargs):
        super().__init__(SensorSuiteUnwrapped(**kwargs))

    # Catch and mask observations
    @override
    def observation(self, obs):
        obs = super().observation(obs)
        if self.env.full_visibility:
            return obs

        for agent_id in obs:
            vis_channels = self.env.agent_sensors[agent_id]
            mask = np.sum(
                obs[agent_id]["image"][..., self.COLOR_SLICE] * vis_channels,
                axis=-1,
            )
            obs[agent_id]["image"] = obs[agent_id]["image"] * mask[..., None]

        return obs
