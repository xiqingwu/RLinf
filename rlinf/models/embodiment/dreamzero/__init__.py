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

"""DreamZero model entry point for RLinf.

Called by rlinf/models/__init__.py when model_type == "dreamzero".
Supports both eval (with transforms) and SFT training (with cpu_init for FSDP).
"""

import json
import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from rlinf.models.embodiment.dreamzero.dreamzero_policy import (
    DreamZeroConfig,
    DreamZeroPolicy,
)

logger = logging.getLogger(__name__)


def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint.

    Args:
        cfg: actor.model config block with fields:
            model_path       - checkpoint dir containing config.json and safetensors
            tokenizer_path   - local path or HF hub ID for umt5-xxl tokenizer
            action_dim       - environment action dimension (default 7)
            embodiment_tag   - tag to select transforms from metadata (default "libero_sim")
            cpu_init         - if True, build on CPU; FSDP shards to GPUs later (avoids OOM)
            is_lora          - if False, set train_architecture="full"
        torch_dtype: dtype to cast model to (ignored when cpu_init=True).
    """
    model_path = Path(cfg.get("model_path"))
    if not model_path.exists():
        raise FileNotFoundError(f"DreamZero model_path does not exist: {model_path}")

    tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")
    action_dim = cfg.get("action_dim", 7)
    cpu_init = cfg.get("cpu_init", False)

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    head_cfg = config_dict.get("action_head_cfg", {}).get("config", {})
    if isinstance(head_cfg, dict):
        head_cfg["train_architecture"] = "lora" if cfg.get("is_lora", False) else "full"

    dreamzero_config = DreamZeroConfig(**config_dict)
    # Disable defer_lora_injection for immediate loading
    if "config" in dreamzero_config.action_head_cfg and isinstance(
        dreamzero_config.action_head_cfg["config"], dict
    ):
        dreamzero_config.action_head_cfg["config"]["defer_lora_injection"] = False
        dreamzero_config.action_head_cfg["config"]["skip_component_loading"] = True

    dreamzero_config.env_action_dim = action_dim

    # Set train_architecture for full fine-tuning (non-LoRA)
    if not cfg.get("is_lora", False):
        head_cfg = config_dict.get("action_head_cfg", {}).get("config", {})
        if isinstance(head_cfg, dict):
            head_cfg["train_architecture"] = "full"
            # Re-create config with updated head_cfg
            dreamzero_config = DreamZeroConfig(**config_dict)
            if "config" in dreamzero_config.action_head_cfg and isinstance(
                dreamzero_config.action_head_cfg["config"], dict
            ):
                dreamzero_config.action_head_cfg["config"]["defer_lora_injection"] = False
                dreamzero_config.action_head_cfg["config"]["skip_component_loading"] = True
            dreamzero_config.env_action_dim = action_dim

    # Build transforms from experiment config (if available)
    _transforms = None
    exp_cfg_dir = model_path / "experiment_cfg"
    train_cfg_path = exp_cfg_dir / "conf.yaml"
    metadata_path = exp_cfg_dir / "metadata.json"
    if train_cfg_path.exists() and metadata_path.exists():
        from groot.vla.data.schema import DatasetMetadata
        from groot.vla.data.transform import ComposedModalityTransform

        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        embodiment_tag = cfg.get("embodiment_tag", "libero_sim")
        if embodiment_tag in metadatas:
            metadata = DatasetMetadata.model_validate(metadatas[embodiment_tag])

            train_cfg = OmegaConf.load(train_cfg_path)
            if embodiment_tag in train_cfg.get("transforms", {}):
                train_cfg.transforms[embodiment_tag].transforms[-1].tokenizer_path = tokenizer_path
                _transforms = instantiate(train_cfg.transforms[embodiment_tag])
                assert isinstance(_transforms, ComposedModalityTransform), f"{_transforms=}"
                _transforms.set_metadata(metadata)
                _transforms.eval()

                dreamzero_config.relative_action = train_cfg.get("relative_action", False)
                dreamzero_config.relative_action_per_horizon = train_cfg.get(
                    "relative_action_per_horizon", False
                )
                dreamzero_config.relative_action_keys = train_cfg.get("relative_action_keys", [])
        else:
            logger.warning(
                "embodiment_tag '%s' not found in metadata.json; transforms disabled.",
                embodiment_tag,
            )
    else:
        logger.info(
            "DreamZero checkpoint has no experiment_cfg; eval transforms disabled. "
            "SFT training path is still supported."
        )

    model = DreamZeroPolicy(
        config=dreamzero_config,
        _transforms=_transforms,
    )

    # Load safetensors weights (support index shards)
    state_dict = {}
    st = model_path / "model.safetensors"
    st_index = model_path / "model.safetensors.index.json"
    if st_index.exists():
        with open(st_index, "r") as f:
            index = json.load(f)
        for shard_file in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(str(model_path / shard_file)))
    elif st.exists():
        state_dict.update(load_file(str(st)))
    else:
        logger.warning("No safetensors weights found under %s; initialized from config only.", model_path)

    if state_dict:
        if any(".base_layer." in k for k in state_dict):
            state_dict = {k.replace(".base_layer.", "."): v for k, v in state_dict.items()}
        for k, v in state_dict.items():
            if v.dim() == 0:
                state_dict[k] = v.unsqueeze(0)
        model.load_state_dict(state_dict, strict=False)

    if not cpu_init and hasattr(model, "post_initialize"):
        try:
            model.post_initialize()
        except Exception as e:
            logger.warning("post_initialize skipped: %s", e)

    # Always cast to the requested dtype (bf16) to halve parameter memory.
    # When cpu_init is True, stay on CPU — FSDP will shard to GPUs later.
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    return model
