from __future__ import annotations

import numpy as np
from typing import override, SupportsFloat
from collections.abc import Sequence

from ..core.agent import Agent
from ..core.space import NDSpace
from ..core.world_object import WorldObj, Goal
from ..hypergrid_env import HyperGridEnv, AgentID


class ForagingEnv(HyperGridEnv):
    def __init__(
        self,
        num_food: int = 1,
        level_based: bool = False,
        fixed_level: bool = False,
        warmth_reward: float = None,
        goal_shape: bool = False,
        ally_shape: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cooperative_task = True
        self.num_food = num_food
        self.food_loc = np.empty((num_food, self.n_dims), dtype=np.int32)
        # Level attributes
        self.level_based = level_based
        self.fixed_level = fixed_level
        self.check_level = level_based or fixed_level
        self.agent_levels = np.ones(self.num_agents)
        self.food_levels = np.ones(self.num_food)
        # Behavioral shaping
        self._warmth_reward = warmth_reward or 1 / self.max_steps
        self._goal_shape = goal_shape
        self._ally_shape = ally_shape

    @override
    def _gen_grid(self, dims: Sequence[int]):
        self.grid = NDSpace(dims)
        self.grid.make_boundary()

        # Place a goals randomly
        for i in range(self.num_food):
            self.food_loc[i] = self.place_obj(Goal())

        for agent in self.agents:
            if (
                self.agent_start_pos is not None
                and self.agent_start_dir is not None
            ):
                agent.state.pos = self.agent_start_pos
                agent.state.dir = self.agent_start_dir
            else:
                self.place_agent(agent)

        if self.level_based:
            self.agent_levels = np.zeros(self.num_agents)
            self.food_levels = np.zeros(self.num_food)

    @override
    def _reward(
        self,
        group=[
            1,
        ],
    ):
        # return 1 - 0.9 * (self.step_count / self.max_steps)
        return len(group)

    @override
    def _on_success(
        self,
        food_ind: int,
        group: list[AgentID],
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool] = {},
        infos: dict[AgentID, dict] = None,
    ):
        # Assign rewards
        if self.joint_reward:
            # reward all agents
            for i in range(self.num_agents):
                rewards[i] = self._reward(group)
        else:
            # reward this agent only
            for i in group:
                rewards[i] = self._reward(group)

        # If level based, increment levels
        if self.level_based:
            self.food_levels[food_ind] += 1
            self.agent_levels[[group]] += 1

        self._redeploy_goal(food_ind)

    def _redeploy_goal(self, food_ind):
        # Redeploy goal (preventing respawn at same location)
        new_loc = self.place_obj(Goal())
        if not new_loc:
            raise RuntimeError(f"New location error: {new_loc=}")
        self.grid.set(self.food_loc[food_ind], WorldObj.empty())
        self.food_loc[food_ind] = new_loc

    @override
    def _cooperative_interactions(
        self,
        actions: dict[AgentID, Sequence[int]],
        rewards: dict[AgentID, SupportsFloat],
        infos: dict[AgentID, dict] = None,
    ):
        """
        Steps:
        1. Filter for agents that are interacting
            1.1 Assign cost for interaction
        2. Group agents
            2.1 Groups without valid target don't do anything
            2.2 It is (probably) faster to group based on valid target
        3. For groups
            3.1 Verify that group meets level requirement
            3.2 Evoke `self._on_success`
        """
        # Collect acting agents
        action_targets = {}
        for i, action in actions.items():
            if action["interact"]:
                agent = self.agents[i]
                # Assign cost
                rewards[i] -= agent.cost_interaction
                action_targets[i] = agent.pos + agent.dir

        # For each goal, check for group
        for f in range(self.num_food):
            food_group = [
                i
                for i, target in action_targets.items()
                if np.array_equal(target, self.food_loc[f])
            ]

            # For goals with groups, if necessary, check group level
            if food_group and (
                not self.check_level
                or self.food_levels[f] <= self.agent_levels[[food_group]].sum()
            ):
                self._on_success(
                    food_ind=f, group=food_group, rewards=rewards, infos=infos
                )

    @override
    def _agent_interaction(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        infos: dict[AgentID, dict] = None,
    ):
        """
        Overriding the individual agent interactions prevents
        the first agent (in the random resolution order) from
        taking the reward, preventing cooperation of agents
        on a task that isn't higher than every contributor.
        """
        if not self.cooperative_task:
            super()._agent_interaction(self, agent, rewards, infos)
        else:
            pass

    @override
    def _occupation_effects(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool] = {},
        infos: dict[AgentID, dict] = None,
    ):
        """
        Overriding the occupation check is necessary to prevent solo
        occupation from triggering on success effects.
        """
        if self._goal_shape:
            distances = np.abs(self.food_loc - agent.pos)
            chebyshev = np.max(distances, axis=-1)
            infos[agent.index]["distance_to_goals"] = chebyshev
            min_cheby = np.min(chebyshev)
            if min_cheby > 0:
                feedback = self._warmth_reward / min_cheby
                rewards[agent.index] += feedback

        if self._ally_shape:
            distances = np.abs(self.agent_states.pos - agent.pos)
            chebyshev = np.max(distances, axis=-1)
            infos[agent.index]["distance_to_allies"] = chebyshev
            sum_cheby = np.sum(chebyshev)
            if sum_cheby > 0:
                feedback = self._warmth_reward / sum_cheby
                rewards[agent.index] += feedback
