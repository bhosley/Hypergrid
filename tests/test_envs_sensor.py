import sys

sys.dont_write_bytecode = True

import pytest
import numpy as np

from hypergrid.envs.sensor_suite import SensorSuiteUnwrapped as unwrapped
from hypergrid.envs.sensor_suite import SensorSuiteEnv as wrapped
from hypergrid.core.constants import Color

from .test_envs_forage import go_to_food


# Ruff formatter doesn't like the param fixtures below.
# fmt: off
@pytest.mark.parametrize(
    "agents, agent_sensors",
    [
        (2, 4),
        (2, [1,]*(len(Color)-1)), # One mask with missing channel
        (2, ([1,]*(len(Color)-1),)*2), # Correct number masks, bad masks
    ])
def test_env_sensor_suite_invalid_input_raises_error(agents, agent_sensors):
        with pytest.raises((TypeError, ValueError)):
            unwrapped(agents=agents, agent_sensors=agent_sensors)
            wrapped(agents=agents, agent_sensors=agent_sensors)


@pytest.mark.parametrize(
    "agents, agent_sensors",
    [
        (2, None), # Default is full vis
        (2, [1,]*len(Color)), # Send single mask of full vis
        (2, ([1,]*(len(Color)),)*2), # Send two mask of full vis
        (2, {i:np.array([1,]*len(Color)) for i in (0,1)}), # Send masks as dict
        (2, {1:np.array([1,]*len(Color))}), # Sends one mask
    ])
def test_env_sensor_suite_sensor_input_to_default(agents, agent_sensors):
        env = unwrapped(agents=agents, agent_sensors=agent_sensors)
        assert env.full_visibility
        env2 = wrapped(agents=agents, agent_sensors=agent_sensors)
        assert env2.env.full_visibility


@pytest.mark.parametrize(
    "agents, agent_sensors",
    [
        (2, ([0,]+[1,]*(len(Color)-1))), # Send single mask
        (2, ([0,]+[1,]*(len(Color)-1),)*2), # Send two masks
        (2, {i:np.array([0,]+[1,]*(len(Color)-1)) for i in (0,1)}), # Same dicts
        (2, {0:np.array([1,]*len(Color)),
             1:np.array([0,]+[1,]*(len(Color)-1))}), # Only one missing channel
        (2, {1:np.array([0,]+[1,]*(len(Color)-1))}), # as above, but implicit
    ])
def test_env_sensor_suite_sensor_input_robustness(agents, agent_sensors):
    env = unwrapped(agents=agents, agent_sensors=agent_sensors)
    assert not env.full_visibility
    env2 = wrapped(agents=agents, agent_sensors=agent_sensors)
    assert not env2.env.full_visibility

@pytest.mark.parametrize("vis", [None, True])
@pytest.mark.parametrize("agents", [1, 2, 4])
def test_sensor_suite_loop(vis, agents):
    env = unwrapped(record_visibility_on_success=vis, agents=agents)
    env.reset()
    _,_,_,_,infos = go_to_food(env)
    if vis:
        assert infos is not None, f"{infos=}"
