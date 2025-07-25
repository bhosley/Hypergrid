import sys

sys.dont_write_bytecode = True

import pytest
import numpy as np

from hypergrid.core.space import NDSpace
from hypergrid.core.world_object import WorldObj, Wall


VALID_VALS = (
    "valid_dims, valid_idx",
    [
        ((2, 2), (1, 1)),
        ((2, 3), (1, 1)),
        (
            (
                3,
                2,
            ),
            (1, 1),
        ),
        ((10, 10), (5, 5)),
        ((3, 3, 3), (1, 1, 1)),
        ((6, 6, 6), (3, 3, 3)),
    ],
)


@pytest.mark.parametrize(
    "invalid_dims", [(), (0,), (-1, 2), (1.5, 2), (2, 2, None), None]
)
def test_ndspace_invalid_dims_raise_value_error(invalid_dims):
    """Invalid dims tuples should raise ValueError on construction."""
    with pytest.raises((ValueError, TypeError)):
        NDSpace(invalid_dims)


@pytest.mark.parametrize(
    "first_dims, second_dims",
    [
        ((2, 3), [2, 3]),
        ([2, 3, 4], (2, 3, 4)),
        ((2, 3, 4), [2, 3, 4]),
    ],
)
def test_ndspace_accepts_list_dims(first_dims, second_dims):
    """Passing dims as a list should work identically to a tuple."""
    space1 = NDSpace(first_dims)
    space2 = NDSpace(second_dims)
    space_funcs = (
        "__class__",
        "shape",
        "dims",
    )
    state_funcs = (
        "__class__",
        "shape",
    )

    for func in space_funcs:
        assert getattr(space1, func) == getattr(space2, func)
    for func in state_funcs:
        assert getattr(space1.state, func) == getattr(space2.state, func)


@pytest.mark.parametrize(*VALID_VALS)
def test_ndspace_initial_state_empty(valid_dims, valid_idx):
    """All cells in state should initially be set to the empty encoding."""
    space = NDSpace(valid_dims)
    for idx in np.ndindex(space.state.shape[:-1]):
        assert not WorldObj.from_array(space.state[*idx])  # NoneType is Falsey


@pytest.mark.parametrize(*VALID_VALS)
def test_ndspace_get_creates_and_caches_worldobj(valid_dims, valid_idx):
    """space.get should return a WorldObj and cache it in world_objects."""
    space = NDSpace(valid_dims)
    assert valid_idx not in space.world_objects  # Initially not in cache
    obj1 = space.get(valid_idx)
    assert not obj1  # Empty cell returns none
    assert valid_idx in space.world_objects  # Idx is now cached


@pytest.mark.parametrize(*VALID_VALS)
def test_ndspace_grid_list_length_and_contents(valid_dims, valid_idx):
    """grid property returns a list of WorldObj for each cell."""
    space = NDSpace(valid_dims)
    grid_list = space.grid
    # length should equal product of dims
    assert len(grid_list) == np.prod(valid_dims)
    # each element should be a WorldObj.empty() == NoneType by default
    assert all(not o for o in grid_list)


# --------------------------- Test Objects in Space  ---------------------------


@pytest.mark.parametrize(*VALID_VALS)
def test_ndspace_manipulation(valid_dims, valid_idx):
    valid_objects = list(WorldObj._TYPE_IDX_TO_CLASS.values())
    space = NDSpace(valid_dims)
    for OBJ in valid_objects:
        space.set(valid_idx, OBJ())
        assert isinstance(space.get(valid_idx), OBJ)


# Invalid Walls
@pytest.mark.parametrize(
    "dims, inval_start, inval_end",
    [
        # Type problems:
        ((10, 10), None, (1, 1)),
        ((10, 10), (1, 1), None),
        ((10, 10), 1, (1, 1)),
        ((10, 10), (1, 1), 1),
        ((10, 10), "1", (1, 1)),
        ((10, 10), (1, 1), "1"),
        # Value Problems:
        ((10, 10), (0, 0), (1, 1)),  # Points must share a hyperplane
        ((10, 10, 10), (0, 0), (1, 0, 0)),  # Start is not in Vector space
        ((10, 10, 10), (0, 0, 0), (1, 0)),  # Stop is not in Vector space
    ],
)
def test_ndspace_hyperwall_invalid(dims, inval_start, inval_end):
    space = NDSpace(dims)
    with pytest.raises((ValueError, TypeError)):
        space.hyperwall(inval_start, inval_end)


@pytest.mark.parametrize(
    "dims, start, end, test_point_pos, test_point_neg",
    [
        # Type problems:
        ((10, 10), (0, 0), (0, 4), (0, 2), (2, 0)),
        ((10, 10, 10), (0, 0, 0), (0, 4, 0), (0, 2, 0), (2, 0, 0)),
    ],
)
def test_ndspace_hyperwall(dims, start, end, test_point_pos, test_point_neg):
    space = NDSpace(dims)
    space.hyperwall(start, end)
    assert isinstance(space.get(test_point_pos), Wall)
    assert not isinstance(space.get(test_point_neg), Wall)


@pytest.mark.parametrize(*VALID_VALS)
def test_ndspace_boundary(valid_dims, valid_idx):
    from itertools import product

    def split_points(dims):
        """
        dims: tuple of ints, e.g. (3,3,4)
        returns: (interior, exterior) as lists of tuples
        """
        # build a sequence of ranges: 0..d-1 for each dim
        axes = [range(d) for d in dims]

        interior = []
        exterior = []
        for pt in product(*axes):
            # check if every coord is strictly inside (1..d-2)
            if all(0 < x < d - 1 for x, d in zip(pt, dims)):
                interior.append(pt)
            else:
                exterior.append(pt)
        return interior, exterior

    space = NDSpace(valid_dims)
    space.make_boundary()
    ip, ep = split_points(valid_dims)
    assert all((isinstance(space.get(p), Wall) for p in ep))
    assert all((not isinstance(space.get(p), Wall) for p in ip))
