from dataclasses import dataclass
from collections import defaultdict
from gymnasium import spaces


@dataclass
class ActionSpec:
    min_speed: int = 0
    max_speed: int = 1
    n_dim: int = 2
    indices: dict = defaultdict

    def __post_init__(self):
        self.indices: dict = {
            "move": 0,
            "orient": range(1, self.n_dim + 1),
            "interact": self.n_dim + 1,
        }

    def to_space(self) -> spaces.MultiDiscrete:
        """
        Represent actions as a single MultiDiscrete vector:
        - idx 0:            move in [min_speed .. max_speed]
        - idx 1:n_dim+1:    orient in [-1 .. 1]
        - idx n_dim+1:      interact in [0, 1]
        """
        # Lower bounds for each sub-action
        n_low = [
            self.min_speed,  # move
            *(0,) * self.n_dim,  # orient index
            0,  # interact
        ]
        # Range sizes for each sub-action
        n_vec = [
            self.max_speed - self.min_speed + 1,  # move choices
            *(3,) * self.n_dim,  # orient choices
            2,  # interact choices (0,1)
        ]

        return spaces.MultiDiscrete(start=n_low, nvec=n_vec)

    def to_dict(self, action):
        return {k: action[i] for k, i in self.indices.items()}


class OrthogonalActionSpec(ActionSpec):
    """
    ActionSpec subtype with ortholinear (axis-aligned) constrained orientation.
    Orientation is chosen as one of 2 * n_dim discrete directions.
    """

    def __post_init__(self):
        # Initialize base indices, then override for a single orient index
        super().__post_init__()
        self.indices = {"move": 0, "orient": 1, "interact": 2}

    def to_space(self) -> spaces.MultiDiscrete:
        """
        Represent actions as a single MultiDiscrete vector:
        - idx 0: move in [min_speed .. max_speed]
        - idx 1: orient in [0 .. 2*n_dim-1]
        - idx 2: interact in [0, 1]
        """
        # Lower bounds for each sub-action
        n_low = [self.min_speed, 0, 0]  # move  # orient index  # interact
        # Range sizes for each sub-action
        n_vec = [
            self.max_speed - self.min_speed + 1,  # move choices
            2 * self.n_dim,  # orient choices
            2,  # interact choices (0,1)
        ]
        return spaces.MultiDiscrete(start=n_low, nvec=n_vec)

    def to_dict(self, action):
        """
        Decode a Dict action into the same format as ActionSpec.to_dict:
        - move: actual speed (action['move'] + min_speed)
        - orient: a length-n_dim list with one nonzero entry
        - interact: 0 or 1
        """
        # Decode move
        move_val = action[self.indices["move"]]

        # Decode orient index into axis and sign
        idx = action[self.indices["orient"]]
        axis = idx // 2  # which axis
        sign = 1 if (idx % 2) == 0 else -1
        orient_vec = [0] * self.n_dim
        orient_vec[axis] = sign

        # Decode interact
        interact_val = action[self.indices["interact"]]

        # Return in same dict form as base
        return {
            "move": move_val,
            "orient": orient_vec,
            "interact": interact_val,
        }
