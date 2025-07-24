from ..hypergrid_env import HyperGridBase
from ..core.actions import OrthogonalActionSpec


class OrthoSensorSuite(HyperGridBase):
    def __init__(self, **kwargs):
        super().__init__(agent_action_spec=OrthogonalActionSpec, **kwargs)

    def gen_obs(self):
        return super().gen_obs()
        # Observability check here.
