import sys

sys.dont_write_bytecode = True

import pytest
import numpy as np

from hypergrid.envs.foraging import ForagingEnv as ENVCLASS


@pytest.mark.parametrize("num_agents", [1, 2, 3])
@pytest.mark.parametrize("dims", [[7, 7], [8, 8]])
@pytest.mark.parametrize("num_food", [1, 2, 3])
def test_forage_basic_loop(num_agents, dims, num_food):
    env = ENVCLASS(agents=num_agents, dims=dims, num_food=num_food)
    env.reset()
    for _ in range(4):
        action = env.action_space.sample()
        env.step(action)


def go_to_point(env, idx):
    # Calculate Routes
    routes = {}
    max_route = 0
    for i in range(env.num_agents):
        dist = env.agent_states.pos[i] - idx
        steps = []
        for d, le in enumerate(dist):
            ori = d * 2 + (0 if le < 0 else 1)
            steps.extend(abs(le) * [[1, ori, 0]])

        if len(steps) > 0:
            max_route = max(max_route, len(steps))
            steps.pop()
        routes[i] = steps

    # Line-up agents
    action = env.action_space.sample()
    for _ in range(max_route):
        for i, route in routes.items():
            if route:
                action[i] = route.pop()
            else:
                action[i][0] = 0
                action[i][-1] = 0
        env.step(action)

    return action


def go_to_food(env, food_idx=0):
    food_loc = env.food_loc[food_idx]
    action = go_to_point(env, food_loc)
    # Interact in unison
    for i in action.keys():
        dist = env.agent_states.pos[i] - food_loc
        ori = sum(
            [
                abs(le) * (2 * d + (0 if le < 0 else 1))
                for d, le in enumerate(dist)
            ]
        )
        action[i] = [0, ori, 1]
    outs = env.step(action)
    return outs


# Test Food Respawn
def test_forage_food_respawn():
    env = ENVCLASS(dims=[7, 3])
    # Initial location
    init_food_loc = np.array((env.food_loc[0]))
    go_to_food(env)
    # Food should respawn in a new location
    assert not np.array_equal(init_food_loc, env.food_loc)


# Test Level Requirement
@pytest.mark.parametrize("test_level", [2, 8])
def test_forage_food_level_requirement(test_level):
    env = ENVCLASS(agents=2, num_food=2, level_based=True)  # , dims=[7,7]
    env.reset()
    # Record starting position of food
    init_food_loc = np.array((env.food_loc))
    # Set a food too high for agents
    env.food_levels[1] = test_level
    go_to_food(env, food_idx=1)
    # Food should remain after failing to be harvested
    assert np.array_equal(init_food_loc, env.food_loc)
    # Level once
    go_to_food(env, food_idx=0)
    # Verify that this one respawns
    assert not np.array_equal(init_food_loc[0], env.food_loc[0])

    # Grind a lower level food until ready
    while env.agent_levels.sum() < test_level:
        go_to_food(env, food_idx=0)

    # Now agent team should be ready
    assert np.array_equal(init_food_loc[1], env.food_loc[1])
    go_to_food(env, food_idx=1)
    assert not np.array_equal(init_food_loc[1], env.food_loc[1])


@pytest.mark.parametrize("num_agents", [1, 2])
@pytest.mark.parametrize("goal_shape", [False, True])
@pytest.mark.parametrize("ally_shape", [False, True])
def test_forage_behavior_shaping(num_agents, goal_shape, ally_shape):
    env = ENVCLASS(
        agents=num_agents,
        goal_shape=goal_shape,
        ally_shape=ally_shape,
    )
    outs = go_to_food(env)
    for i in range(num_agents):
        assert outs[1][i] > 0


def test_forage_agents_level_up():
    env = ENVCLASS(agents=2, level_based=True)
    env.reset()
    init_lvl = env.agent_levels.copy()
    go_to_food(env)
    assert np.all(init_lvl < env.agent_levels)


def test_forage_coop_level():
    # Should not succeed
    env = ENVCLASS(agents=2, coop_level=3)
    env.reset()
    init_loc_f = env.food_loc.copy()
    go_to_food(env)
    assert np.all(init_loc_f == env.food_loc)
    # Should succeed
    env = ENVCLASS(agents=3, coop_level=3)
    env.reset()
    init_loc_f = env.food_loc.copy()
    go_to_food(env)
    assert np.any(init_loc_f != env.food_loc)
