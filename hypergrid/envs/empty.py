from __future__ import annotations

from hypergrid.hypergrid_env import HyperGridEnv


class EmptyEnv(HyperGridEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
