from __future__ import annotations

import numpy as np

from collections import defaultdict
from functools import cached_property
from itertools import product
from numpy.typing import NDArray as ndarray
from typing import Any, Callable, Iterable, Sequence

# from .agent import Agent
from .constants import Type, TILE_PIXELS
from .world_object import Wall, WorldObj

# from ..utils.rendering import (
#     downsample,
#     fill_coords,
#     highlight_img,
#     point_in_rect,
# )


class NDSpace:
    """
    Class representing a hypergrid of :class:`.WorldObj` objects.

    Attributes
    ----------
    dims : tuple[int]
        Tuple of n-Dimensions and their domains
    world_objects : dict[tuple[int, int], WorldObj]
        Dictionary of world objects in the grid, indexed by (x, y) location
    state : ndarray[int] of shape (width, height, WorldObj.dim)
        Grid state, where each (x, y) entry is a world object encoding
    """

    # Static cache of pre-renderer tiles
    _tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(self, dims: Sequence[int]):
        """
        Parameters
        ----------
        dims : tuple[int]
            Tuple of n-Dimensions and their domains
        """
        # Validate dimensions
        if not len(dims) > 0:
            raise ValueError("Invalid dimensionality, must be > 0.")
        if not all((isinstance(dn, int) and dn > 0 for dn in dims)):
            raise ValueError("One or more dimensions are invalid.")

        self.world_objects: dict[tuple[int, ...], WorldObj] = (
            {}
        )  # indexed by location
        # self.state: ndarray = np.zeros((*dims, WorldObj.dim), dtype=int)
        # self.state[...] = WorldObj.empty()
        self.dims = [*dims]
        self.state: ndarray = np.full((*dims, WorldObj.dim), WorldObj.empty())

    @property
    def shape(self) -> tuple[int, ...]:
        """Alias for `dims`."""
        return self.state.shape

    @property
    def grid(self) -> list[WorldObj | None]:
        """
        Return a list of all world objects in the grid.
        """
        # return [self.get(i) for i,_ in np.ndenumerate(np.zeros(self.dims))]
        return [self.get(idx) for idx in np.ndindex(*self.dims)]
        # return [self.get(i) for i,_ in np.ndenumerate(self.state)]
        # return [self.get(i, j) for i in range(self.width) for j in range(self.height)]

    def set(self, index: Sequence[int], obj: WorldObj | None):
        # def set(self, x: int, y: int, obj: WorldObj | None):
        """
        Set a world object at the given coordinates.

        Parameters
        ----------
        index : Sequence[int]
            Set of n coordinates
        obj : WorldObj or None
            Object to place
        """
        # Update world object dictionary
        self.world_objects[*index] = obj

        # Update grid state
        if isinstance(obj, WorldObj):
            self.state[*index] = obj
        elif obj is None:
            self.state[*index] = WorldObj.empty()
        else:
            raise TypeError(f"cannot set grid value to {type(obj)}")

    def get(self, index: Sequence[int]) -> WorldObj | None:
        """
        Get the world object at the given coordinates.

        Parameters
        ----------
        index : Sequence[int]
            Set of n coordinates
        """
        # Create WorldObj instance if none exists
        if tuple(index) not in self.world_objects:
            try:
                self.world_objects[*index] = WorldObj.from_array(
                    self.state[*index]
                )
            except IndexError:
                return WorldObj.empty()

        return self.world_objects[*index]

    def update(self, index: Sequence[int]):
        """
        Update the grid state from the world object at the given coordinates.

        Parameters
        ----------
        index : Sequence[int]
            Set of n coordinates
        """
        if index in self.world_objects:
            self.state[*index] = self.world_objects[*index]

    def hyperwall(
        self,
        start: Sequence[int],
        end: Sequence[int],
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        """Create a hyperspatial wall between two vectors"""
        # Check vectors
        if not len(start) == len(self.dims):
            raise ValueError("Start vector not in vector space")
        if not len(end) == len(self.dims):
            raise ValueError("End vector not in vector space")

        if not any(list(s == e for s, e in zip(start, end))):
            raise ValueError("Invalid start/stop pair, no shared hyperplanes")

        ranges = (
            range(
                s,
                e + (1 if e >= s else -1),  # End inclusive
                1 if e >= s else -1,
            )  # Reverse order if start < end
            for s, e in zip(start, end)
        )
        points = list(product(*ranges))

        if self.world_objects:
            for p in points:
                self.set(p, obj_type())
        else:
            for p in points:
                self.state[p] = obj_type()

    def make_boundary(self):
        """Build hyperwalls along each boundary."""
        origin = [0] * len(self.dims)  # all zeros
        up_lim = [d - 1 for d in self.dims]  # each upper‐limit is d−1

        for i, d in enumerate(up_lim):
            c = list(up_lim)
            c[i] = 0
            self.hyperwall(origin, c)
            c = list(origin)
            c[i] = d
            self.hyperwall(up_lim, c)

    # def wall_rect(self, x: int, y: int, w: int, h: int):
    #     """
    #     Create a walled rectangle.

    #     Parameters
    #     ----------
    #     x : int
    #         X-coordinate of top-left corner
    #     y : int
    #         Y-coordinate of top-left corner
    #     w : int
    #         Width of rectangle
    #     h : int
    #         Height of rectangle
    #     """
    #     self.horz_wall(x, y, w)
    #     self.horz_wall(x, y + h - 1, w)
    #     self.vert_wall(x, y, h)
    #     self.vert_wall(x + w - 1, y, h)

    # @classmethod
    # def render_tile(
    #     cls,
    #     obj: WorldObj | None = None,
    #     agent: Agent | None = None,
    #     highlight: bool = False,
    #     tile_size: int = TILE_PIXELS,
    #     subdivs: int = 3) -> ndarray[np.uint8]:
    #     """
    #     Render a tile and cache the result.

    #     Parameters
    #     ----------
    #     obj : WorldObj or None
    #         Object to render
    #     agent : Agent or None
    #         Agent to render
    #     highlight : bool
    #         Whether to highlight the tile
    #     tile_size : int
    #         Tile size (in pixels)
    #     subdivs : int
    #         Downsampling factor for supersampling / anti-aliasing
    #     """
    #     # Hash map lookup key for the cache
    #     key: tuple[Any, ...] = (highlight, tile_size)
    #     if agent:
    #         key += (agent.state.color, agent.state.dir)
    #     else:
    #         key += (None, None)
    #     key = obj.encode() + key if obj else key

    #     if key in cls._tile_cache:
    #         return cls._tile_cache[key]

    #     img = np.zeros(
    #         shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    #     # Draw the grid lines (top and left edges)
    #     fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    #     fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    #     # Draw the object
    #     if obj is not None:
    #         obj.render(img)

    #     # Draw the agent
    #     if agent is not None and not agent.state.terminated:
    #         agent.render(img)

    #     # Highlight the cell if needed
    #     if highlight:
    #         highlight_img(img)

    #     # Downsample the image to perform supersampling/anti-aliasing
    #     img = downsample(img, subdivs)

    #     # Cache the rendered tile
    #     cls._tile_cache[key] = img

    #     return img

    # def render(
    #     self,
    #     tile_size: int,
    #     agents: Iterable[Agent] = (),
    #     highlight_mask: ndarray[np.bool] | None = None) -> ndarray[np.uint8]:
    #     """
    #     Render this grid at a given scale.

    #     Parameters
    #     ----------
    #     tile_size: int
    #         Tile size (in pixels)
    #     agents: Iterable[Agent]
    #         Agents to render
    #     highlight_mask: ndarray
    #         Boolean mask indicating which grid locations to highlight
    #     """
    #     if highlight_mask is None:
    #         highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

    #     # Get agent locations
    #     # For overlapping agents, non-terminated agents get priority
    #     location_to_agent = defaultdict(type(None))
    #     for agent in sorted(agents, key=lambda a: not a.terminated):
    #         location_to_agent[tuple(agent.pos)] = agent

    #     # Initialize pixel array
    #     width_px = self.width * tile_size
    #     height_px = self.height * tile_size
    #     img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    #     # Render the grid
    #     for j in range(0, self.height):
    #         for i in range(0, self.width):
    #             assert highlight_mask is not None
    #             cell = self.get(i, j)
    #             tile_img = Grid.render_tile(
    #                 cell,
    #                 agent=location_to_agent[i, j],
    #                 highlight=highlight_mask[i, j],
    #                 tile_size=tile_size,
    #             )

    #             ymin = j * tile_size
    #             ymax = (j + 1) * tile_size
    #             xmin = i * tile_size
    #             xmax = (i + 1) * tile_size
    #             img[ymin:ymax, xmin:xmax, :] = tile_img

    #     return img

    # def encode(self, vis_mask: ndarray[np.bool] | None = None) -> ndarray[np.int]:
    #     """
    #     Produce a compact numpy encoding of the grid.

    #     Parameters
    #     ----------
    #     vis_mask : ndarray[bool] of shape (width, height)
    #         Visibility mask
    #     """
    #     if vis_mask is None:
    #         vis_mask = np.ones((self.width, self.height), dtype=bool)

    #     encoding = self.state.copy()
    #     encoding[~vis_mask][..., WorldObj.TYPE] = Type.unseen.to_index()
    #     return encoding

    # @staticmethod
    # def decode(array: ndarray[np.int]) -> tuple['Grid', ndarray[np.bool]]:
    #     """
    #     Decode an array grid encoding back into a `Grid` instance.

    #     Parameters
    #     ----------
    #     array : ndarray[int] of shape (width, height, dim)
    #         Grid encoding

    #     Returns
    #     -------
    #     grid : Grid
    #         Decoded `Grid` instance
    #     vis_mask : ndarray[bool] of shape (width, height)
    #         Visibility mask
    #     """
    #     width, height, dim = array.shape
    #     assert dim == WorldObj.dim

    #     vis_mask = (array[..., WorldObj.TYPE] != Type.unseen.to_index())
    #     grid = Grid(width, height)
    #     grid.state[vis_mask] = array[vis_mask]
    #     return grid, vis_mask
