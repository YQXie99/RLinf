# Copyright 2025 The RLinf Authors.
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

import numpy as np
import torch

from rlinf.config import SupportedModel
from rlinf.envs import SupportedEnvType
from rlinf.models.embodiment.prismatic.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NormalizationType,
)


def prepare_actions_for_maniskill(
    raw_chunk_actions,
    num_action_chunks,
    action_dim,
    action_scale,
    policy,
) -> torch.Tensor:
    if "panda" in policy:
        return raw_chunk_actions
    # TODO only suitable for action_dim = 7
    reshaped_actions = raw_chunk_actions.reshape(-1, action_dim)
    batch_size = reshaped_actions.shape[0]
    raw_actions = {
        "world_vector": np.array(reshaped_actions[:, :3]),
        "rotation_delta": np.array(reshaped_actions[:, 3:6]),
        "open_gripper": np.array(
            reshaped_actions[:, 6:7]
        ),  # range [0, 1]; 1 = open; 0 = close
    }

    # process raw_action to obtain the action to be sent to the maniskill2 environment
    actions = {}
    actions["world_vector"] = raw_actions["world_vector"] * action_scale  # [B, 3]
    actions["rot_axangle"] = raw_actions["rotation_delta"] * action_scale  # [B, 3]

    if policy == "google_robot":
        raise NotImplementedError
    elif policy == "widowx_bridge":
        actions["gripper"] = 2.0 * (raw_actions["open_gripper"] > 0.5) - 1.0  # [B, 1]
    elif policy == "panda_wristcam":
        actions["gripper"] = 2.0 * (raw_actions["open_gripper"] > 0.5) - 1.0  # [B, 1]

    actions["terminate_episode"] = np.array([0.0] * batch_size).reshape(-1, 1)  # [B, 1]

    actions = {k: torch.tensor(v, dtype=torch.float32) for k, v in actions.items()}
    actions = torch.cat(
        [actions["world_vector"], actions["rot_axangle"], actions["gripper"]], dim=1
    ).cuda()

    chunk_actions = actions.reshape(-1, num_action_chunks, action_dim)

    return chunk_actions


def prepare_actions_for_libero(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = np.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_isaaclab(
    raw_chunk_actions,
    model_type,
) -> torch.Tensor:
    """
    Here reture a general 7 dof action. If the action is modified, please change the output of the model
    For example, in `RLinf/rlinf/models/embodiment/gr00t/simulation_io.py`
    """
    chunk_actions = (
        torch.from_numpy(raw_chunk_actions)
        if isinstance(raw_chunk_actions, np.ndarray)
        else raw_chunk_actions
    )
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = torch.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_calvin(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) == SupportedModel.OPENPI:
        chunk_actions[..., -1] = np.sign(chunk_actions[..., -1])
    else:
        chunk_actions[..., -1] = np.where(chunk_actions[..., -1] > 0, 1, -1)
    return chunk_actions


def prepare_actions_for_metaworld(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        # the action dimesion of metaworld is 4-dim (x, y, z, gripper)
        # we need to extract the first 3-dim and the last dim in a 7-dim action
        if chunk_actions.shape[-1] == 7:
            chunk_actions = np.concatenate(
                [chunk_actions[..., :3], chunk_actions[..., -1:]], axis=-1
            )
    return chunk_actions


def prepare_actions_for_robocasa(
    raw_chunk_actions,
    action_dim,
    action_space,
) -> np.ndarray:
    """
    Prepare actions for robocasa environment.
    Model outputs 32D actions per chunk, and model got first N valid actions defined by action_space, but robocasa expects 12D.
    So extract the first N dimensions, fit to corresponding ids, and pad the rest to get12 dimensions (3D pos + 3D ori + 1D gripper + 5D base).
    """

    # raw_chunk_actions shape: [num_chunks, 32]
    # Extract first action_dim (<=12) dimensions as valid action chunks
    # Then pad them to default actions to get (..., 12)-shaped action chunks for RobocasaEnv.step()
    from rlinf.envs.robocasa.utils import (
        ROBOCASA_ALL_ACTION_DIM,
        ROBOCASA_DEFAULT_ACTION,
        get_action_ids,
        get_action_space,
    )

    assert action_dim <= ROBOCASA_ALL_ACTION_DIM, (
        f"Requested action_dim ({action_dim}) exceeds max dimension ({ROBOCASA_ALL_ACTION_DIM})."
    )

    valid_chunk_actions = raw_chunk_actions[..., :action_dim]

    chunk_actions = np.full(
        shape=valid_chunk_actions.shape[:-1] + (ROBOCASA_ALL_ACTION_DIM,),
        fill_value=ROBOCASA_DEFAULT_ACTION,
        dtype=valid_chunk_actions.dtype,
    )

    all_action_ids = get_action_ids(get_action_space(action_space))
    assert len(all_action_ids) == action_dim, (
        f"Mismatch between action_space ids length ({len(all_action_ids)}) and provided action_dim ({action_dim})."
    )
    chunk_actions[..., all_action_ids] = valid_chunk_actions

    return chunk_actions


def prepare_actions_for_mujoco(raw_chunk_actions, model_type):
    if raw_chunk_actions.shape[-1] >= 7:
        chunk_actions = np.concatenate(
            [raw_chunk_actions[..., :3], raw_chunk_actions[..., 6:7]], axis=-1
        )
    else:
        chunk_actions = raw_chunk_actions[..., :4]
    if SupportedModel(model_type) == SupportedModel.OPENPI:
        chunk_actions[..., -1] = np.clip(chunk_actions[..., -1], -1.0, 1.0)
    return chunk_actions


def unnormalize_actions(normalized_actions, action_norm_stats):
    """Unnormalize actions using dataset statistics"""

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
        )
        # Ensure mask is a numpy boolean array (may come as list from json)
        mask = np.array(mask, dtype=bool)
        action_high, action_low = (
            np.array(action_norm_stats["max"]),
            np.array(action_norm_stats["min"]),
        )
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = (
            np.array(action_norm_stats["q99"]),
            np.array(action_norm_stats["q01"]),
        )
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    if isinstance(normalized_actions, torch.Tensor):
        normalized_actions = normalized_actions.cpu().numpy()

    action_dim = normalized_actions.shape[-1]

    action_high = action_high[:action_dim]
    action_low = action_low[:action_dim]
    mask = mask[:action_dim]

    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_actions,
    )

    return actions


def prepare_actions(
    raw_chunk_actions,
    env_type: str,
    model_type: str,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
    wm_env_type=None,
    action_norm_stats=None,
    use_openpi_unnormalize: bool = None,
    openpi_use_quantiles: bool = False,
) -> torch.Tensor | np.ndarray:
    raw_chunk_actions = (
        raw_chunk_actions.cpu().numpy()
        if isinstance(raw_chunk_actions, torch.Tensor)
        else raw_chunk_actions
    )

    env_type = SupportedEnvType(env_type)
    if env_type == SupportedEnvType.LIBERO:
        chunk_actions = prepare_actions_for_libero(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.OPENSORAWM or env_type == SupportedEnvType.WANWM:
        # TODO: Implement prepare_actions_for_opensora_wm
        if wm_env_type == "libero":
            chunk_actions = prepare_actions_for_libero(
                raw_chunk_actions=raw_chunk_actions,
                model_type=model_type,
            )
        else:
            raise NotImplementedError(f"Env type {wm_env_type} not implemented")
    elif env_type == SupportedEnvType.MANISKILL:
        chunk_actions = prepare_actions_for_maniskill(
            raw_chunk_actions=raw_chunk_actions,
            num_action_chunks=num_action_chunks,
            action_dim=action_dim,
            action_scale=action_scale,
            policy=policy,
        )
    elif env_type == SupportedEnvType.ROBOTWIN:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.METAWORLD:
        chunk_actions = prepare_actions_for_metaworld(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.CALVIN:
        chunk_actions = prepare_actions_for_calvin(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.BEHAVIOR:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.ISAACLAB:
        chunk_actions = prepare_actions_for_isaaclab(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.ROBOCASA:
        chunk_actions = prepare_actions_for_robocasa(
            raw_chunk_actions=raw_chunk_actions,
            action_dim=action_dim,
            action_space=policy,
        )
    elif env_type == SupportedEnvType.REALWORLD:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.FRANKASIM:
        chunk_actions = prepare_actions_for_mujoco(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    else:
        raise NotImplementedError

    # Auto-detect OpenPI model if use_openpi_unnormalize is not explicitly set
    if use_openpi_unnormalize is None:
        try:
            use_openpi_unnormalize = SupportedModel(model_type) == SupportedModel.OPENPI
        except (ValueError, TypeError):
            use_openpi_unnormalize = False

    # Use OpenPI-specific unnormalization if specified
    if action_norm_stats is not None:
        if use_openpi_unnormalize:
            # Use OpenPI normalization function (defined in this file)
            chunk_actions = unnormalize_openpi_actions(
                chunk_actions,
                action_norm_stats,
            )
        else:
            chunk_actions = unnormalize_actions(chunk_actions, action_norm_stats)

    return chunk_actions


def normalize_openpi_state(state, state_norm_stats, use_quantiles=True):
    """
    Normalize state for OpenPI model using dataset statistics.

    This function normalizes state observations to the range [-1, 1] for OpenPI model input.
    It supports both bounds-based and quantile-based normalization.

    Args:
        state: Raw state observations from environment (can be torch.Tensor or np.ndarray)
        state_norm_stats: Dictionary containing normalization statistics. Should contain:
            - For bounds: "min" and "max" keys
            - For quantiles: "q01" and "q99" keys (or "q10" and "q90")
            - Optional: "mask" key for masking certain state dimensions
        use_quantiles: If True, use quantile-based normalization (q01/q99),
                      otherwise use bounds-based normalization (min/max)

    Returns:
        Normalized state in the range [-1, 1]
    """
    if state_norm_stats is None:
        return state

    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()

    state_high = np.array(state_norm_stats["q99"])
    state_low = np.array(state_norm_stats["q01"])

    # Check if stats are empty before proceeding
    if state_high.shape[0] == 0 or state_low.shape[0] == 0:
        raise ValueError("State stats are empty, skipping normalization")

    mask = state_norm_stats.get(
        "mask", np.ones_like(state_norm_stats["q01"], dtype=bool)
    )

    state_dim = state.shape[-1]

    state_high = state_high[:state_dim]
    state_low = state_low[:state_dim]
    mask = mask[:state_dim]

    # Normalize: convert from [state_low, state_high] to [-1, 1]
    # Formula: normalized_state = 2 * (state - low) / (high - low) - 1
    normalized_state = np.where(
        mask,
        2.0 * (state - state_low) / (state_high - state_low + 1e-8) - 1.0,
        state,
    )

    return normalized_state


def unnormalize_openpi_actions(
    normalized_actions, action_norm_stats, use_quantiles=True
):
    """
    Unnormalize actions for OpenPI model using dataset statistics.

    This function is designed specifically for OpenPI models.
    It supports both bounds-based and quantile-based normalization.

    Args:
        normalized_actions: Normalized actions from OpenPI model (expected to be in [-1, 1] range)
        action_norm_stats: Dictionary containing normalization statistics. Should contain:
            - For bounds: "min" and "max" keys
            - For quantiles: "q01" and "q99" keys (or "q10" and "q90")
            - Optional: "mask" key for masking certain action dimensions
        use_quantiles: If True, use quantile-based normalization (q01/q99),
                      otherwise use bounds-based normalization (min/max)

    Returns:
        Unnormalized actions in the original action space
    """
    if action_norm_stats is None:
        return normalized_actions

    if isinstance(normalized_actions, torch.Tensor):
        normalized_actions = normalized_actions.cpu().numpy()

    mask = action_norm_stats.get(
        "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
    )
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])

    # If stats are empty, skip unnormalization
    if action_high.shape[0] == 0 or action_low.shape[0] == 0:
        raise ValueError("Action stats are empty, skipping unnormalization")

    # Ensure mask is a numpy boolean array (may come as list from json)
    mask = np.array(mask, dtype=bool)

    action_dim = normalized_actions.shape[-1]
    repeat_factor = action_dim // action_high.shape[0]
    # If dimensions are incompatible, skip unnormalization
    if repeat_factor == 0:
        return normalized_actions
    action_high = action_high.repeat(repeat_factor)
    action_low = action_low.repeat(repeat_factor)
    mask = mask.repeat(repeat_factor) if mask.ndim > 0 else mask

    # Unnormalize: convert from [-1, 1] to [action_low, action_high]
    # Formula: action = 0.5 * (normalized_action + 1) * (high - low) + low
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_actions,
    )

    return actions
