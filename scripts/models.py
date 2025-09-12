from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import (
    TorchMultiCategorical,
    TorchCategorical,
)


def get_policy_mapping_fn(
    checkpoint_dir: Path | str | None,
    num_agents: int,
) -> Callable:
    try:
        policies = sorted(
            [
                path
                for path in (checkpoint_dir / "policies").iterdir()
                if path.is_dir()
            ]
        )

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return policies[agent_id % len(policies)].name

        print("Loading policies from:", checkpoint_dir)
        for agent_id in range(num_agents):
            print(
                "Agent ID:", agent_id, "Policy ID:", policy_mapping_fn(agent_id)
            )
        return policy_mapping_fn

    except Exception:
        return lambda agent_id, *args, **kwargs: f"policy_{agent_id}"


def get_TorchMultiDiscrete(action_space):
    from ray.rllib.utils.annotations import override

    class TorchMultDiscrete(TorchMultiCategorical):
        input_lengths = list(action_space.nvec)

        @override(TorchMultiCategorical)
        def __init__(self, categoricals: list[TorchCategorical], **kwargs):
            super().__init__(categoricals)

        @classmethod
        @override(TorchMultiCategorical)
        def from_logits(cls, logits: "torch.Tensor", **kwargs):
            # If RLlib already supplied input_lens, don't override it.
            kwargs.setdefault("input_lens", cls.input_lengths)
            # Properly delegate to the parent implementation (no manual 'cls' arg).
            return super(TorchMultDiscrete, cls).from_logits(logits, **kwargs)

    return TorchMultDiscrete


class TwoHeadModule(TorchRLModule, ValueFunctionAPI):
    # self._tune_module_cfg.get("",)

    def __init__(self, *args, **kwargs):
        # Extract an optional module_config; RLlib passes this through RLModuleSpec.
        self._tune_module_cfg = (
            kwargs.pop("model_config", None)
            or kwargs.pop("module_config", {})
            or {}
        )
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        self.action_dist_cls = get_TorchMultiDiscrete(self.action_space)
        # Manually build sub-nets for each part of the Dict:
        input_img = self.observation_space["image"].shape[-1]
        input_dir = self.observation_space["direction"].shape[0]

        # Encoders
        self.encoder_dims = self._tune_module_cfg.get("encoder_dims", [128, 16])
        # Image Encoder
        # img_layers = self._tune_module_cfg.get("img_layers", [64, 64])
        # img_kernel = self._tune_module_cfg.get("img_kernel", 2)
        # img_act_nm = self._tune_module_cfg.get("img_activation", "relu")
        # match img_act_nm:
        #     case "tanh":
        #         img_act_fn = nn.Tanh()
        #     case "gelu":
        #         img_act_fn = nn.GELU()
        #     case "elu":
        #         img_act_fn = nn.ELU()
        #     case _:
        #         img_act_fn = nn.ReLU()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(input_img, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, self.encoder_dims[0]),
        )
        self.ori_encoder = nn.Sequential(
            nn.Linear(input_dir, self.encoder_dims[1]),
            nn.ReLU(),
        )
        # Define a fusion trunk and heads:
        self.trunk = nn.Sequential(
            nn.Linear(sum(self.encoder_dims), 128),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Tanh(),
        )
        # Unpack the action space shape as an int
        # self.pi_head = nn.Linear(64, *self.action_space.shape)
        self.pi_head = nn.Linear(64, int(self.action_space.nvec.sum()))
        self.value_head = nn.Linear(64, 1)

    def _forward(self, batch: dict, **kwargs):
        embeddings = self.heads_and_body(batch)
        logits = self.pi_head(embeddings)
        assert logits.shape[-1] == int(
            self.action_space.nvec.sum()
        ), f"logits dim {logits.shape[-1]} != sum(nvec) {
            int(self.action_space.nvec.sum())
        }"
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    def value_function(self, **kwargs):
        # RLlib's ValueFunctionAPI expects this to return a 1D tensor of V(s)
        # matching the last forward() call.
        return self._value_out

    def compute_values(self, batch, embeddings=None):
        """
        Compute V(s) for a batch.
        If `embeddings` (i.e., trunk output) is provided, reuse it; otherwise,
        recompute the embedding from the obs dict using this module's encoders.
        Returns a 1D tensor of shape [B].
        """
        if embeddings is None:
            embeddings = self.heads_and_body(batch)
        values = self.value_head(embeddings).squeeze(1)
        return values

    def heads_and_body(self, batch):
        img = batch["obs"]["image"].float()
        ori = batch["obs"]["direction"].float()
        # Permute channels from [-1] to [1] - (N D1 D2 C -> N C D1 D2)
        img = img.permute(0, -1, *(range(1, img.ndim - 1)))
        z_img = self.img_encoder(img)
        z_ori = self.ori_encoder(ori)
        z = torch.cat([z_img, z_ori], dim=1)
        embedding = self.trunk(z)
        return embedding
