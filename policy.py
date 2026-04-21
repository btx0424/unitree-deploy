from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from observation import (
    CommandObservation,
    JointPositionObservation,
    JointVelocityObservation,
    ObservationContext,
    ObservationGroup,
    PreviousActionObservation,
    ProjectedGravityObservation,
)

class Policy:
    def __init__(
        self,
        policy_yaml_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.policy_yaml_path = Path(policy_yaml_path).expanduser().resolve()
        with self.policy_yaml_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._load_action_config()
        self._load_policy_config()
        self._load_observation_config()

        self.model_path = self._resolve_model_path()
        session_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=session_providers)

    def _load_action_config(self) -> None:
        self.robot_joint_names = list(self.config["robot_joint_names"])
        self.robot_joint_index = {name: idx for idx, name in enumerate(self.robot_joint_names)}
        self.default_joint_pos_full = np.asarray(
            self.config["default_qpos_full"],
            dtype=np.float32,
        )
        self.control_step_dt = float(self.config["step_dt"])
        self.physics_dt = float(self.config["physics_dt"])
        self.decimation = int(self.config["decimation"])

        self.action_dim = int(self.config["action_dim"])
        self.action_clip = float(self.config["action_clip"])
        self.policy_action_joint_names = list(self.config["action_joint_names"])
        self.controlled_joint_names = list(self.config["controlled_joint_names"])
        self.policy_to_controlled_reorder = np.asarray(
            self.config["policy_to_controlled_reorder"],
            dtype=np.int64,
        )

        self.controlled_joint_indices = np.asarray(
            [self.robot_joint_index[name] for name in self.controlled_joint_names],
            dtype=np.int64,
        )

        default_joint_pos_cfg = {
            name: value for name, value in zip(self.policy_action_joint_names, self.config["default_qpos_action"])
        }
        self.default_joint_pos_controlled = np.asarray(
            [default_joint_pos_cfg[name] for name in self.controlled_joint_names],
            dtype=np.float32,
        )

        action_scaling_cfg = {
            name: value for name, value in zip(self.policy_action_joint_names, self.config["action_scale"])
        }
        self.action_scaling_controlled = np.asarray(
            [action_scaling_cfg[name] for name in self.controlled_joint_names],
            dtype=np.float32,
        )

    def _load_policy_config(self) -> None:
        self.input_name = self.config["policy_input_name"]
        self.policy_input_dim = int(self.config["policy_input_dim"])
        self.action_output_name = self.config["policy_output_name"]

    def _load_observation_config(self) -> None:
        command_observation = CommandObservation(
            history_len=int(self.config["command_history_len"]),
            height_command=float(self.config["height_command"]),
            command_range=self.config["command_range"],
        )
        gravity_observation = ProjectedGravityObservation(
            history_len=int(self.config["projected_gravity_history_len"])
        )
        joint_pos_observation = JointPositionObservation(
            controlled_joint_indices=self.controlled_joint_indices,
            history_len=int(self.config["joint_pos_history_len"]),
        )
        joint_vel_observation = JointVelocityObservation(
            controlled_joint_indices=self.controlled_joint_indices,
            history_len=int(self.config["joint_vel_history_len"]),
        )
        previous_action_observation = PreviousActionObservation(
            action_dim=self.action_dim,
            history_len=int(self.config["prev_action_history_len"]),
        )

        self.previous_action_observations = [previous_action_observation]
        self.observation = ObservationGroup(
            [
                command_observation,
                gravity_observation,
                joint_pos_observation,
                joint_vel_observation,
                previous_action_observation,
            ]
        )

    def _resolve_model_path(self) -> Path:
        return (self.policy_yaml_path.parent / self.config["policy_path"]).resolve()

    def reset(self) -> None:
        self.observation.reset()

    def compute_target_q(self, context: ObservationContext) -> np.ndarray:
        self.observation.update(context)
        obs_vector = self.observation.compute().astype(np.float32, copy=False)
        outputs = self.session.run(
            [self.action_output_name],
            {self.input_name: obs_vector[None, :]},
        )
        policy_action = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        controlled_action = policy_action[self.policy_to_controlled_reorder]
        for previous_action_observation in self.previous_action_observations:
            previous_action_observation.record_action(controlled_action)
        clipped_action = np.clip(controlled_action, -self.action_clip, self.action_clip)

        target_q = self.default_joint_pos_full.astype(np.float64, copy=True)
        target_q[self.controlled_joint_indices] = (
            self.default_joint_pos_controlled.astype(np.float64, copy=False)
            + self.action_scaling_controlled.astype(np.float64, copy=False) * clipped_action
        )
        return target_q
