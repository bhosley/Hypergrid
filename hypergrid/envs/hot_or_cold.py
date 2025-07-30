from __future__ import annotations

from collections.abc import Sequence
from typing import override, SupportsFloat
import numpy as np

from ..core.agent import Agent
from ..core.space import NDSpace
from ..core.world_object import Goal, Lava
from ..hypergrid_env import HyperGridEnv, AgentID


class HotOrColdEnv(HyperGridEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.goal_location = np.array([dn - 2 for dn in self.dims])
        # Warm reward is capped at the decay of the
        # default time-decay of completion reward
        self._warmth_reward = 1 / self.max_steps

    @override
    def _gen_grid(self, dims: Sequence[int]):
        self.grid = NDSpace(dims)
        # Generate the surrounding walls
        self.grid.make_boundary()
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.goal_location)
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

    @override
    def _reward(self) -> float:
        """
        Override the scale of the reward.
        """
        return 1 - 0.9 * (self.step_count / self.max_steps)

    @override
    def _occupation_effects(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool] = {},
    ):
        """
        Provide a small reward based on how close to the goal
        each agent is, just as a treat.
        """
        loc = agent.pos
        if isinstance(self.grid.get(loc), Lava):
            self._on_failure(agent, rewards, terminations)
        if isinstance(self.grid.get(loc), Goal):
            self._on_success(agent, rewards, terminations)

        dist = np.abs(np.subtract(self.goal_location, loc))
        feedback = np.sum(self._warmth_reward / (dist + 1))
        rewards[agent.index] = feedback
