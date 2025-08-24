"""
************
Environments
************

This package contains implementations of several MultiGrid environments.

**************
Configurations
**************

* `Empty <./hypergrid.envs.empty>`_
    * ``HyperGrid-Empty-v0``
    * ``HyperGrid-Empty-8x8-v0``
"""

from .empty import EmptyEnv
from .hot_or_cold import HotOrColdEnv
from .foraging import ForagingEnv
from .sensor_suite import SensorSuiteEnv


CONFIGURATIONS = {
    "HyperGrid-Empty-v0": (EmptyEnv, {}),
    "HyperGrid-Empty-8x8-v0": (EmptyEnv, {"dims": [8, 8]}),
    "HyperGrid-HotCold-v0": (HotOrColdEnv, {}),
    "Foraging": (ForagingEnv, {}),
    "SensorSuite": (SensorSuiteEnv, {}),
}

# Register environments with gymnasium
from gymnasium.envs.registration import register

for name, (env_cls, config) in CONFIGURATIONS.items():
    register(id=name, entry_point=env_cls, kwargs=config)
