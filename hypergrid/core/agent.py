from __future__ import annotations

import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray as ndarray

from .actions import ActionSpec
from .constants import Color, Type
from .mission import Mission, MissionSpace
from .world_object import WorldObj
from ..utils.misc import PropertyAlias


class Agent:
    """
    Class representing an agent in an N-dimensional grid environment.
    """

    def __init__(
        self,
        index: int,
        spatial_ndim: int = 2,
        mission_space: MissionSpace = MissionSpace.from_string(
            "maximize reward"
        ),
        view_size: int = 7,
        see_through_walls: bool = False,
        action_spec: ActionSpec = None,
        cost_collision_agent: float | None = 0.0,
        cost_collision_env: float | None = 0.0,
        cost_interaction: float | None = 0.0,
    ):
        """
        Parameters
        ----------
        index : int
            Agent index in the environment.
        spatial_ndim : int
            Number of spatial dimensions (≥1).
        mission_space : MissionSpace
            The agent's objective specification
        view_size : int
            Size of the hypercube view (must be odd and ≥3).
        see_through_walls : bool
            Whether or not the agent can see through sight blocking obstacles
        action_spec : ActionSpec
            Agent action specifications
        """
        if spatial_ndim <= 0:
            raise ValueError("Invalid dimension, must be > 0.")
        if index is not None and index < 0:
            raise ValueError("Invalid index, must be >= 0.")
        if view_size < 3:
            raise ValueError("View too small")
        if not view_size % 2 == 1:
            raise ValueError("View width must be odd")

        self.index = index
        self.spatial_ndim = spatial_ndim
        self.view_size = view_size
        self.see_through_walls = see_through_walls
        self.mission: Mission = None

        # initialize state vector (vectorized over agents is handled at env-level)
        self.state: AgentState = AgentState(spatial_ndim=spatial_ndim)

        # Observation space: an array of shape (view_size,)*spatial_ndim + (WorldObj.dim,)
        obs_shape = (*[view_size] * spatial_ndim, WorldObj.dim)

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=obs_shape,
                    dtype=np.uint8,
                    # low=0, high=255, shape=obs_shape, #dtype=np.float32
                ),
                # "direction": spaces.Discrete(spatial_ndim),
                "direction": spaces.MultiDiscrete([3] * spatial_ndim),
                # "mission": mission_space,
            }
        )

        # # Actions remain discrete (must import Action enum if used)
        # self.action_space = spaces.Discrete(len(Type))  # adjust as appropriate
        # Actions: an orientation vector in {-1,0,+1} for each axis
        # 0->-1, 1->0, 2->+1
        # self.action_space = MultiDiscrete([3] * spatial_ndim)
        if action_spec and issubclass(action_spec, ActionSpec):
            self.action_spec = action_spec(n_dim=self.spatial_ndim)
        else:
            self.action_spec = ActionSpec(n_dim=self.spatial_ndim)
        # if not action_spec:
        # self.action_spec = ActionSpec(n_dim=self.spatial_ndim)
        self.action_space = self.action_spec.to_space()

        # Initialize orientation vector in state (index-level)
        self.state.orientation = (
            1,
        ) * spatial_ndim  # default to no movement (1 maps to 0)

        # Agent related costs
        self.cost_collision_agent = (
            float(cost_collision_agent) if cost_collision_agent else 0.0
        )
        self.cost_collision_env = (
            float(cost_collision_env) if cost_collision_env else 0.0
        )
        self.cost_interaction = (
            float(cost_interaction) if cost_interaction else 0.0
        )

    color = PropertyAlias("state", "color", doc="Alias for AgentState.color")
    dir = PropertyAlias("state", "dir", doc="Alias for AgentState.dir")
    orientation = PropertyAlias(
        "state", "orientation", doc="Alias for AgentState.orientation"
    )
    pos = PropertyAlias("state", "pos", doc="Alias for AgentState.pos")
    terminated = PropertyAlias(
        "state", "terminated", doc="Alias for AgentState.terminated"
    )
    carrying = PropertyAlias(
        "state", "carrying", doc="Alias for AgentState.carrying"
    )

    @property
    def front_pos(self) -> tuple[int, ...]:
        """
        Return the coordinates of the cell in front of the agent,
        the position plus the agent's orientation vector.
        """
        # TODO: np.add instead?
        delta = self.state.orientation
        current = self.state.pos
        return tuple(c + d for c, d in zip(current, delta))

    def reset(self, mission: Mission = Mission("maximize reward")):
        self.mission = mission
        # initialize direction, position, and flags
        # self.state.dir = -1
        self.state.orientation = (0,) * self.spatial_ndim
        self.state.pos = (1,) * self.spatial_ndim
        self.state.terminated = False
        self.state.carrying = None

    def encode(self) -> tuple[int, int, tuple[int, ...]]:
        return (
            Type.agent.to_index(),
            self.state.color.to_index(),
            self.state.orientation,
        )

    def render(self, img: ndarray[np.uint8]):
        """
        Render the agent onto an N-dimensional projection image.
        For 2D, img should be shape (W,H,3). For higher D, you may slice to 2D first.
        """
        # Example: always render into the last two dims
        # user must supply img appropriately projected
        # tri_fn = ...  # your rendering logic here


class AgentState(np.ndarray):
    """
    N-dimensional vectorized state for agents.
    Supports stacking multiple agents by passing batch_dims to __new__.
    """

    TYPE_IDX = 0
    COLOR_IDX = 1
    #
    ORIENT_START = COLOR_IDX + 1
    ORIENT_END = ORIENT_START
    POS_START = ORIENT_END
    POS_END = POS_START
    TERM_IDX = POS_END
    #
    CARRY_START = TERM_IDX + 1
    CARRY_END = CARRY_START + WorldObj.dim
    total_dim = CARRY_END

    # Testing pre-provision:
    ENCODING = slice(0, ORIENT_END)
    _slice_pos = slice(POS_START, POS_END)
    _slice_orient = slice(ORIENT_START, ORIENT_END)
    _slice_carry = slice(CARRY_START, CARRY_END)

    def __new__(cls, *batch_dims: int, spatial_ndim: int):
        if spatial_ndim <= 0:
            raise ValueError("Invalid dimension, must be > 0.")

        # Recompute slice indices dynamically
        AgentState.ORIENT_END = cls.ORIENT_START + spatial_ndim
        AgentState.POS_START = cls.ORIENT_END
        AgentState.POS_END = cls.POS_START + spatial_ndim
        AgentState.TERM_IDX = cls.POS_END
        AgentState.CARRY_START = cls.TERM_IDX + 1
        AgentState.CARRY_END = cls.CARRY_START + WorldObj.dim
        AgentState.total_dim = cls.CARRY_END
        AgentState.ENCODING = slice(0, cls.ORIENT_END)

        # stash indices for property methods
        cls._idx_type = AgentState.TYPE_IDX
        cls._idx_color = AgentState.COLOR_IDX
        cls._slice_pos = slice(AgentState.POS_START, AgentState.POS_END)
        cls._slice_orient = slice(
            AgentState.ORIENT_START, AgentState.ORIENT_END
        )
        cls._idx_term = AgentState.TERM_IDX
        cls._slice_carry = slice(AgentState.CARRY_START, AgentState.CARRY_END)
        cls._spatial_ndim = spatial_ndim

        # create base array
        base = np.zeros(batch_dims + (cls.total_dim,), dtype=int).view(cls)
        # default init
        # base[..., TYPE_IDX] = Type.agent
        # base[..., COLOR_IDX].flat = Color.cycle(np.prod(batch_dims or (1,)))
        base[..., AgentState.TYPE_IDX] = Type.agent
        base[..., AgentState.COLOR_IDX].flat = Color.cycle(
            np.prod(batch_dims or (1,))
        )
        # base[..., DIR_IDX] = -1
        # initialize orientation indices (1 maps to zero movement)
        # base[..., ORIENT_START:ORIENT_END] = 1
        # base[..., POS_START:POS_END] = (1,) * spatial_ndim
        base[..., AgentState._slice_orient] = 1
        # base[..., AgentState.ORIENT_START:AgentState.ORIENT_END] = 1
        base[..., AgentState._slice_pos] = (1,) * spatial_ndim
        # base[..., AgentState.POS_START:AgentState.POS_END] = (1,) * spatial_ndim

        # store auxiliary arrays
        base._terminated = np.zeros(batch_dims, dtype=bool)
        base._carried_obj = np.full(batch_dims, None, dtype=object)
        # # N-dimensional orientation vector per agent in {-1,0,+1}, map to indices 0,1,2
        # base._orientation = np.full(batch_dims + (spatial_ndim,), 1, dtype=int)
        base._view = base.view(np.ndarray)

        # # stash indices for property methods
        # base._idx_type = TYPE_IDX
        # base._idx_color = COLOR_IDX
        # # base._idx_dir = DIR_IDX
        # base._slice_pos = slice(POS_START, POS_END)
        # base._slice_orient = slice(ORIENT_START, ORIENT_END)
        # base._idx_term = TERM_IDX
        # base._slice_carry = slice(CARRY_START, CARRY_END)
        # base._spatial_ndim = spatial_ndim
        return base

    def __repr__(self):
        shape = str(self.shape[:-1]).replace(",)", ")")
        return f"{self.__class__.__name__}{shape}"

    # def __array_finalize__(self, obj):
    #     """
    #     Ensure that attributes from the original array are carried over to views and slices.
    #     """
    #     if obj is None:
    #         return
    #     # List of metadata attributes to propagate
    #     for attr in (
    #         '_terminated', '_carried_obj', '_view',
    #         # '_idx_type', '_idx_color', '_idx_dir',
    #         # '_slice_pos', '_idx_term', '_slice_carry',
    #         '_idx_type', '_idx_color',
    #         '_slice_pos', '_slice_orient', '_idx_term', '_slice_carry',
    #         '_spatial_ndim'
    #     ):
    #         # Only set if the source object has the attribute
    #         if hasattr(obj, attr):
    #             setattr(self, attr, getattr(obj, attr))

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        # if out.shape and out.shape[-1] == self.dim:
        if out.shape and out.shape[-1] == self.total_dim:
            out._view = self._view[idx, ...]
            out._carried_obj = self._carried_obj[
                idx, ...
            ]  # set carried object reference
            out._terminated = self._terminated[idx, ...]  # set terminated cache

        return out

    # def __getitem__(self, idx):
    #     out = super().__getitem__(idx)
    #     if isinstance(out, AgentState):
    #         # # carry over metadata
    #         # for attr in ('_terminated','_carried_obj','_view',
    #         #              '_idx_type','_idx_color','_idx_dir',
    #         #              '_slice_pos','_idx_term','_slice_carry','_spatial_ndim'):
    #         #     # setattr(out, attr, getattr(self, attr) if attr in ('_terminated','_carried_obj') else getattr(self, attr))
    #         #     setattr(out, attr, getattr(self, attr)[idx, ...])
    #                     # Per-agent arrays: extract this index
    #         out._terminated = self._terminated[idx]
    #         out._carried_obj = self._carried_obj[idx]
    #         # Recompute view for the sub-array
    #         out._view = out.view(np.ndarray)
    #         # Static metadata: copy directly
    #         out._idx_type = self._idx_type
    #         out._idx_color = self._idx_color
    #         out._idx_dir = self._idx_dir
    #         out._slice_pos = self._slice_pos
    #         out._idx_term = self._idx_term
    #         out._slice_carry = self._slice_carry
    #         out._spatial_ndim = self._spatial_ndim
    #     return out

    @property
    def color(self):
        """Return the agent color."""
        return Color.from_index(self._view[..., AgentState.COLOR_IDX])

    @color.setter
    def color(self, value: str | ArrayLike[str]):
        """Set the agent color."""
        self[..., AgentState.COLOR_IDX] = np.vectorize(
            lambda c: Color(c).to_index()
        )(value)

    @property
    def pos(self):
        # out = self._view[..., self._slice_pos]
        out = self._view[..., slice(AgentState.POS_START, AgentState.POS_END)]
        if out.ndim == 1:
            return tuple(out)
        return out

    @pos.setter
    def pos(self, value):
        """
        Set the agent position. Expects a sequence of length spatial_ndim.
        """
        # arr = np.array(value, dtype=int)
        # if arr.shape != (self._spatial_ndim,):
        #     raise ValueError(f"Position must be length {self._spatial_ndim}, got {arr.shape}")
        # Write into the shared state vector
        # self._view[..., self._slice_pos] = arr
        if (
            len(np.shape(value)) != 1
            or np.shape(value)[0] != AgentState._spatial_ndim
        ):
            raise ValueError(
                f"Update shape {np.shape(value)} does not match"
                + f"Position {AgentState._spatial_ndim}"
            )
        # if np.shape(value) != len(slice(AgentState.POS_START,AgentState.POS_END)):
        #     raise ValueError(f"Position must be length {0}, got {np.shape(value)}")
        self[..., slice(AgentState.POS_START, AgentState.POS_END)] = value

    @property
    def terminated(self):
        out = self._terminated
        return bool(out) if out.ndim == 0 else out

    @terminated.setter
    def terminated(self, value):
        """
        Set termination flag. Accepts bool or array of bool.
        """
        self._terminated[...] = value

    @property
    def carrying(self):
        out = self._carried_obj
        return out.item() if out.ndim == 0 else out

    @carrying.setter
    def carrying(self, obj):
        """
        Set carrying object. Accepts any object or array of objects.
        """
        self._carried_obj[...] = obj

    @property
    def orientation(self):
        """Return per-axis orientation vector in {-1,0,+1} as tuple or array."""
        # out = self._view[..., self._slice_orient]
        # out = self._view[..., slice(AgentState.ORIENT_START,AgentState.ORIENT_END)]
        out = self._view[..., AgentState._slice_orient]
        # mapped = out - 1
        # if mapped.ndim == 1:
        #     # return tuple(mapped.tolist())
        #     return tuple(int(x) for x in mapped)
        # return mapped
        return out

    @property
    def dir(self):
        """Alias for `orientation`"""
        return self.orientation

    @orientation.setter
    def orientation(self, value):
        # expect a sequence or tuple of length spatial_ndim with values in {-1,0,+1}
        # arr = np.array(value, dtype=int) + 1
        # if arr.shape != (self._spatial_ndim,):
        # arr = np.array(value, dtype=int) + 1
        # if np.shape(value) != AgentState._spatial_ndim:
        if (
            len(np.shape(value)) != 1
            or np.shape(value)[0] != AgentState._spatial_ndim
        ):
            raise ValueError(
                f"Update shape {np.shape(value)} does not match"
                + f"Orientation {AgentState._spatial_ndim}"
            )
        # self._orientation[...] = arr
        self[..., AgentState._slice_orient] = value
        # self[..., slice(AgentState.POS_START,AgentState.POS_END)] = value

    @dir.setter
    def dir(self, value):
        """Alias for `orientation`"""
        self.orientation = value
        # arr = np.array(value, dtype=int) + 1
        # if arr.shape != (self._spatial_ndim,):
        #     raise ValueError(f"Orientation must be length {self._spatial_ndim}")
        # # self._orientation[...] = arr
        # self._view[..., self._slice_orient] = arr

    @property
    def encode(self):
        """Return the agent encoding"""
        types = self._view[..., AgentState.TYPE_IDX]
        colors = self._view[..., AgentState.COLOR_IDX]
        dirs = self._view[..., AgentState._slice_orient]
        out = [
            [t, c, int("".join(str(i + 1) for i in d))]
            for t, c, d in zip(types, colors, dirs)
        ]
        # TODO: Write note about encoding orientation as a one-shifted string
        # if out.ndim == 1:
        #     return tuple(out)
        return out
