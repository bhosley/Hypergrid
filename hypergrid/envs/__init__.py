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

# from .blockedunlockpickup import BlockedUnlockPickupEnv
from .empty import EmptyEnv
# from .locked_hallway import LockedHallwayEnv
# from .playground import PlaygroundEnv
# from .redbluedoors import RedBlueDoorsEnv

CONFIGURATIONS = {
    "HyperGrid-Empty-v0": (EmptyEnv, {}),
    "HyperGrid-Empty-8x8-v0": (EmptyEnv, {"dims": [8, 8]}),
    #     'HyperGrid-BlockedUnlockPickup-v0': (BlockedUnlockPickupEnv, {}),
    #     'HyperGrid-Empty-5x5-v0': (EmptyEnv, {'size': 5}),
    #     'HyperGrid-Empty-Random-5x5-v0': (EmptyEnv, {'size': 5, 'agent_start_pos': None}),
    #     'HyperGrid-Empty-6x6-v0': (EmptyEnv, {'size': 6}),
    #     'HyperGrid-Empty-Random-6x6-v0': (EmptyEnv, {'size': 6, 'agent_start_pos': None}),
    #     'HyperGrid-Empty-16x16-v0': (EmptyEnv, {'size': 16}),
    #     'HyperGrid-LockedHallway-2Rooms-v0': (LockedHallwayEnv, {'num_rooms': 2}),
    #     'HyperGrid-LockedHallway-4Rooms-v0': (LockedHallwayEnv, {'num_rooms': 4}),
    #     'HyperGrid-LockedHallway-6Rooms-v0': (LockedHallwayEnv, {'num_rooms': 6}),
    #     'HyperGrid-Playground-v0': (PlaygroundEnv, {}),
    #     'HyperGrid-RedBlueDoors-6x6-v0': (RedBlueDoorsEnv, {'size': 6}),
    #     'HyperGrid-RedBlueDoors-8x8-v0': (RedBlueDoorsEnv, {'size': 8}),
}

# Register environments with gymnasium
from gymnasium.envs.registration import register

for name, (env_cls, config) in CONFIGURATIONS.items():
    register(id=name, entry_point=env_cls, kwargs=config)
