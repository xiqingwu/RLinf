# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from hydra.utils import instantiate
from tqdm import tqdm
import numpy as np
import torch
from tianshou.data import Batch
from omegaconf import OmegaConf, DictConfig
import os
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from groot.vla.model.dreamzero.base_vla import VLAConfig, VLA
from transformers.configuration_utils import PretrainedConfig
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DreamZeroConfig(VLAConfig):
    model_type = "dreamzero"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    env_action_dim: int = field(default=None, metadata={"help": "Environment action dimension."})
    num_action_chunks: int = field(default=16, metadata={"help": "Number of action chunks."})

    relative_action: bool = field(default=False, metadata={"help": "Relative action."})
    relative_action_per_horizon: bool = field(default=False, metadata={"help": "Relative action per horizon."})
    relative_action_keys: list = field(default_factory=list, metadata={"help": "Relative action keys."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DreamZeroPolicy(VLA, BasePolicy):
    """DreamZero action model: IdentityBackbone + WANPolicyHead.

    - predict_action_batch: for eval (inference)
    - forward(ForwardType.SFT): for SFT training (returns loss)
    - forward(ForwardType.DEFAULT): default forward
    """

    # Module names that should not be split across FSDP shards.
    _no_split_modules = [
        "WanAttentionBlock",
        "GanAttentionBlock",
        "CausalWanAttentionBlock",
        "DiTBlock",
    ]

    def __init__(
        self,
        config: DreamZeroConfig,
        _transforms: Any = None,
    ):
        super().__init__(config)
        self.config = config
        self._transforms = _transforms

        # FSDP requires all parameters to have ndim >= 1;
        # some Wan components register scalar (0-D) parameters.
        self._fix_scalar_parameters()

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Normalize inputs"""
        if self._transforms is None:
            raise RuntimeError("DreamZero eval transform is unavailable for this checkpoint.")
        obs = batch.obs
        normalized_input = self._transforms(obs)
        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: dict = None, **kwargs):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        if self._transforms is None:
            raise RuntimeError("DreamZero eval transform is unavailable for this checkpoint.")
        unnormalized_action = self._transforms.unapply(
            dict(action=batch.normalized_action.cpu())
        )

        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.config.relative_action
        relative_action_per_horizon = self.config.relative_action_per_horizon
        relative_action_keys = self.config.relative_action_keys
        if (relative_action or relative_action_per_horizon) and relative_action_keys and obs is not None:
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"

                if action_key not in unnormalized_action:
                    continue

                last_state = None

                if state_key in obs:
                    last_state = obs[state_key]
                else:
                    for obs_key in obs.keys():
                        if 'state' in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break

                    if last_state is None and 'state' in obs:
                        state_data = obs['state']
                        action_dim = unnormalized_action[action_key].shape[-1]
                        if torch.is_tensor(state_data):
                            state_dim = state_data.shape[-1]
                        elif isinstance(state_data, np.ndarray):
                            state_dim = state_data.shape[-1]
                        else:
                            state_dim = None

                        if state_dim == action_dim:
                            last_state = state_data

                if last_state is None:
                    continue

                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()

                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]

                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(last_state, axis=-2)

                unnormalized_action[action_key] = unnormalized_action[action_key] + last_state

        batch.act = unnormalized_action
        return batch

    def _process_batch(self, batch: Batch) -> Batch:
        """Process batch."""
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        target_dtype = next(self.parameters()).dtype
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and target_dtype != torch.float32:
                normalized_input[k] = v.to(dtype=target_dtype)
        return normalized_input

    def _observation_convert(self, env_obs: dict) -> dict:
        """Convert environment observation to model input for end-effector control"""
        main = env_obs["main_images"]
        wrist = env_obs.get("wrist_images", None)
        extra_view = env_obs.get("extra_view_images", None)
        states = env_obs.get("states", None)
        prompts = env_obs.get("task_descriptions", None)
        if torch.is_tensor(main):
            main = main.detach().cpu().numpy()
        else:
            main = np.asarray(main)
        B = main.shape[0]
        if wrist is not None:
            if torch.is_tensor(wrist):
                wrist = wrist.detach().cpu().numpy()
            else:
                wrist = np.asarray(wrist)
        import cv2

        def _resize_bt_hwc_uint8(x, h=256, w=256):
            B = x.shape[0]
            out = np.empty((B, h, w, 3), dtype=np.uint8)
            for b in range(B):
                frame = x[b]
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                out[b] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            return out

        main = _resize_bt_hwc_uint8(main)
        if wrist is not None:
            wrist = _resize_bt_hwc_uint8(wrist)
        if main.ndim == 4:
            main = main[:, None, ...]
        if wrist is not None and wrist.ndim == 4:
            wrist = wrist[:, None, ...]
        if states is not None:
            if torch.is_tensor(states):
                s_np = states.detach().cpu().numpy()
            else:
                s_np = np.asarray(states)
        else:
            s_np = np.zeros((B, 8), dtype=np.float32)
        if s_np.ndim == 1:
            s_np = s_np[None, :]
        elif s_np.ndim > 2:
            s_np = s_np.reshape(B, -1)
        s_np = s_np.astype(np.float32)
        state_bt = s_np[:, None, :]
        prompts = prompts if prompts is not None else [""] * B
        if isinstance(prompts, str):
            prompts = [prompts] * B
        converted_obs = {
            "video.image": main,
            "video.wrist_image": wrist,
            "state.state": state_bt,
            "annotation.language.action_text": list(prompts),
        }
        return converted_obs

    def predict_action_batch(self, env_obs, mode, **kwargs) -> np.ndarray:
        """
        input:
        env_obs:
            - main_images: [B,H,W,C] uint8
            - extra_view_images: [B,H,W,C]
            - states: [B,D]
            - task_descriptions: list[str] or None
        output:
        actions: np.ndarray [B, num_action_chunks, action_dim]
        result: dict  # compatible with rollout interface
        """
        B = env_obs["main_images"].shape[0]
        converted_obs = self._observation_convert(env_obs)
        batch = Batch(obs=converted_obs)
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()

        unnormalized_action = self._transforms.unapply(
            dict(action=normalized_action.cpu())
        )
        batch.act = unnormalized_action

        actions = batch.act["action.actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(actions.dtype)

        assert actions.shape[-1] == self.config.env_action_dim, (
            f"Action shape mismatch: {actions.shape} != {self.config.env_action_dim}"
        )

        flat = torch.as_tensor(actions, dtype=torch.float32).reshape(actions.shape[0], -1).cpu()
        forward_inputs = {"action": flat}
        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"forward_type {forward_type} is not supported.")

    def sft_forward(self, data, **kwargs):
        """Forward pass for SFT training, returning {"loss", "dynamics_loss", "action_loss"}."""
        inputs = data
        if not isinstance(inputs, dict):
            raise TypeError("DreamZero sft_forward expects a dict input batch.")
        outputs = VLA.forward(self, inputs)
        if "loss" not in outputs:
            raise KeyError("DreamZero model output does not contain `loss`.")
        out = {"loss": outputs["loss"]}
        if "dynamics_loss" in outputs:
            out["dynamics_loss"] = outputs["dynamics_loss"]
        if "action_loss" in outputs:
            out["action_loss"] = outputs["action_loss"]
        return out

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor] = None,
        data=None,
        **kwargs,
    ) -> dict[str, Any]:
        """Default forward pass."""
        if data is not None:
            return self.sft_forward(data=data)
        raise NotImplementedError("DreamZero default_forward requires `data`.")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        from torch.utils.checkpoint import checkpoint

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
            if hasattr(module, "_gradient_checkpointing_func"):
                module._gradient_checkpointing_func = checkpoint

    def gradient_checkpointing_disable(self):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False

    def _fix_scalar_parameters(self) -> None:
        """Reshape 0-D parameters to 1-D for FSDP compatibility."""
        for name, param in list(self.named_parameters()):
            if param.dim() == 0:
                parts = name.split(".")
                parent = self
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                new_param = torch.nn.Parameter(
                    param.data.unsqueeze(0), requires_grad=param.requires_grad
                )
                setattr(parent, parts[-1], new_param)
                logger.info("Reshaped scalar param %s to 1-D for FSDP", name)
