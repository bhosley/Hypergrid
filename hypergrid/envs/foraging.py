from __future__ import annotations

from typing import override, SupportsFloat
from collections.abc import Sequence

from ..core.agent import Agent
from ..core.space import NDSpace
from ..core.world_object import WorldObj, Goal
from ..hypergrid_env import HyperGridEnv, AgentID


class ForagingEnv(HyperGridEnv):
    def __init__(self, num_food: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_food = num_food
        self.goal_loc = []

    def _gen_grid(self, dims: Sequence[int]):
        self.grid = NDSpace(dims)
        self.grid.make_boundary()

        # Place a goals randomly
        for _ in range(self.num_food):
            self.place_obj(Goal())

        for agent in self.agents:
            if (
                self.agent_start_pos is not None
                and self.agent_start_dir is not None
            ):
                agent.state.pos = self.agent_start_pos
                agent.state.dir = self.agent_start_dir
            else:
                self.place_agent(agent)

        self.agent_states.carrying = 0

    @override
    def _on_success(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool] = {},
    ):
        if self.joint_reward:
            for i in range(self.num_agents):
                rewards[i] = self._reward()  # reward all agents
        else:
            rewards[agent.index] = self._reward()  # reward this agent only

        # Get goal current level
        lvl = self.grid.get(self.goal_loc)
        # remove current goal
        self.grid.set(self.goal_loc, WorldObj.empty())
        # respawn to new location
        self.goal_loc = self.place_obj(Goal())
        # increment the goal's level
        self.grid.get(self.goal_loc).state = lvl + 1

        # Level the agent
        self.agent_states.carrying[0] += 1

    # TODO: Level requirement checker
