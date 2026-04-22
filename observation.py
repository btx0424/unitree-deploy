from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationContext:
    q: np.ndarray
    dq: np.ndarray
    quat: np.ndarray
    gyro: np.ndarray
    lin_acc: np.ndarray
    command: np.ndarray

def _normalize_quaternion(quat) -> np.ndarray:
    quat_array = np.asarray(quat, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(quat_array))
    return quat_array / norm


def _quat_to_body_gravity(quat) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(quat)
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return rot.T @ gravity_world


class ObservationBase:
    def __init__(self, *, base_dim: int, history_len: int, dtype=np.float32) -> None:
        self.base_dim = int(base_dim)
        self.history_len = int(history_len)
        self.dtype = np.dtype(dtype)
        self.buffer = np.zeros((self.history_len, self.base_dim), dtype=self.dtype)

    @property
    def size(self) -> int:
        return self.base_dim * self.history_len

    def reset(self) -> None:
        self.buffer.fill(0.0)

    def update(self, context: ObservationContext) -> None:
        current = np.asarray(self._compute_current(context), dtype=self.dtype).reshape(-1)
        if self.history_len > 1:
            self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = current

    def compute(self) -> np.ndarray:
        return self.buffer.reshape(-1).copy()

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        raise NotImplementedError


class CommandObservation(ObservationBase):
    def __init__(
        self,
        *,
        history_len: int,
        height_command: float,
        command_range,
        dtype=np.float32,
    ) -> None:
        super().__init__(base_dim=4, history_len=history_len, dtype=dtype)
        self.height_command = float(height_command)
        command_range_array = np.asarray(command_range, dtype=self.dtype)
        self.command_min = command_range_array[:, 0]
        self.command_max = command_range_array[:, 1]

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        joystick = np.asarray(context.command, dtype=self.dtype).reshape(-1)
        joystick = np.clip(joystick, -1.0, 1.0)
        command = 0.5 * (joystick + 1.0) * (self.command_max - self.command_min) + self.command_min
        return np.concatenate(
            (command, np.array([self.height_command], dtype=self.dtype))
        ).astype(self.dtype, copy=False)


class ProjectedGravityObservation(ObservationBase):
    def __init__(self, *, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=3, history_len=history_len, dtype=dtype)

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        return _quat_to_body_gravity(context.quat).astype(self.dtype, copy=False)


class RootAngularVelocityObservation(ObservationBase):
    def __init__(self, *, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=3, history_len=history_len, dtype=dtype)

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        return np.asarray(context.gyro, dtype=self.dtype).reshape(-1)


class JointPositionObservation(ObservationBase):
    def __init__(
        self,
        *,
        controlled_joint_indices: np.ndarray,
        history_len: int,
        dtype=np.float32,
    ) -> None:
        controlled_joint_indices = np.asarray(controlled_joint_indices, dtype=np.int64).reshape(-1)
        super().__init__(base_dim=controlled_joint_indices.size, history_len=history_len, dtype=dtype)
        self.controlled_joint_indices = controlled_joint_indices

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        q = np.asarray(context.q, dtype=self.dtype).reshape(-1)
        return q[self.controlled_joint_indices]


class JointVelocityObservation(ObservationBase):
    def __init__(
        self,
        *,
        controlled_joint_indices: np.ndarray,
        history_len: int,
        use_position_difference: bool = False,
        control_dt: float | None = None,
        dtype=np.float32,
    ) -> None:
        controlled_joint_indices = np.asarray(controlled_joint_indices, dtype=np.int64).reshape(-1)
        super().__init__(base_dim=controlled_joint_indices.size, history_len=history_len, dtype=dtype)
        self.controlled_joint_indices = controlled_joint_indices
        self.use_position_difference = bool(use_position_difference)
        self.control_dt = None if control_dt is None else float(control_dt)
        self._previous_q = None

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        if self.use_position_difference:
            if self.control_dt is None or self.control_dt <= 0.0:
                raise ValueError("control_dt must be positive when use_position_difference is enabled.")
            q = np.asarray(context.q, dtype=self.dtype).reshape(-1)[self.controlled_joint_indices]
            if self._previous_q is None:
                joint_vel = np.zeros_like(q)
            else:
                joint_vel = (q - self._previous_q) / self.control_dt
            self._previous_q = q.copy()
            return joint_vel

        dq = np.asarray(context.dq, dtype=self.dtype).reshape(-1)
        return dq[self.controlled_joint_indices]

    def reset(self) -> None:
        super().reset()
        self._previous_q = None


class PreviousActionObservation(ObservationBase):
    def __init__(self, *, action_dim: int, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=action_dim, history_len=history_len, dtype=dtype)

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        return np.zeros(self.base_dim, dtype=self.dtype)

    def update(self, context: ObservationContext) -> None:
        del context

    def record_action(self, action) -> None:
        action_array = np.asarray(action, dtype=self.dtype).reshape(-1)
        if self.history_len > 1:
            self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = action_array


class ObservationGroup:
    def __init__(self, observations: Sequence[ObservationBase], *, dtype=np.float32) -> None:
        self.observations = list(observations)
        self.dtype = np.dtype(dtype)

    @property
    def size(self) -> int:
        return int(sum(observation.size for observation in self.observations))

    def reset(self) -> None:
        for observation in self.observations:
            observation.reset()

    def update(self, context: ObservationContext) -> None:
        for observation in self.observations:
            observation.update(context)

    def compute(self) -> np.ndarray:
        return np.concatenate(
            [observation.compute().astype(self.dtype, copy=False) for observation in self.observations]
        ).astype(self.dtype, copy=False)
