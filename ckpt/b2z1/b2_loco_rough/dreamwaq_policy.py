from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from unitree_deploy.policy.base_policy import BasePolicy, ObservationRegistry


class DreamWaQPolicy(BasePolicy):
    """B2 DreamWaQ policy with a recurrent adaptation state."""

    def __init__(
        self,
        policy_yaml_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        observation_types: ObservationRegistry | None = None,
    ) -> None:
        super().__init__(
            policy_yaml_path,
            providers=providers,
            observation_types=observation_types,
        )
        self.init_input_name = str(self.config["policy_init_input_name"])
        self.state_input_name = str(self.config["policy_state_input_name"])
        self.state_output_name = str(self.config["policy_state_output_name"])
        self.state_dim = int(self.config["policy_state_dim"])
        self.is_init = np.ones((1, 1), dtype=np.bool_)
        self.adapt_hx = np.zeros((1, self.state_dim), dtype=np.float32)
        self._validate_onnx_contract()

    def reset(self) -> None:
        super().reset()
        self.is_init.fill(True)
        self.adapt_hx.fill(0.0)

    def _infer_action(self, obs_vector: np.ndarray) -> np.ndarray:
        action, next_adapt_hx = self.session.run(
            [self.action_output_name, self.state_output_name],
            {
                self.input_name: obs_vector.reshape(1, -1),
                self.init_input_name: self.is_init,
                self.state_input_name: self.adapt_hx,
            },
        )
        next_adapt_hx = np.asarray(next_adapt_hx, dtype=np.float32)
        if next_adapt_hx.shape != self.adapt_hx.shape:
            raise ValueError(
                f"ONNX recurrent output has shape {next_adapt_hx.shape}, "
                f"expected {self.adapt_hx.shape}"
            )
        self.adapt_hx[:] = next_adapt_hx
        self.is_init.fill(False)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _validate_onnx_contract(self) -> None:
        inputs = {info.name: info for info in self.session.get_inputs()}
        outputs = {info.name: info for info in self.session.get_outputs()}

        required_inputs = [self.input_name, self.init_input_name, self.state_input_name]
        missing_inputs = [name for name in required_inputs if name not in inputs]
        if missing_inputs:
            raise ValueError(f"ONNX model missing inputs: {missing_inputs}")

        required_outputs = [self.action_output_name, self.state_output_name]
        missing_outputs = [name for name in required_outputs if name not in outputs]
        if missing_outputs:
            raise ValueError(f"ONNX model missing outputs: {missing_outputs}")

        expected_dims = {
            self.input_name: self.observation.size,
            self.init_input_name: 1,
            self.state_input_name: self.state_dim,
            self.action_output_name: self.action_dim,
            self.state_output_name: self.state_dim,
        }
        for name, expected_dim in expected_dims.items():
            info = inputs.get(name, outputs.get(name))
            shape = info.shape
            if len(shape) != 2 or (isinstance(shape[1], int) and shape[1] != expected_dim):
                raise ValueError(
                    f"ONNX tensor {name!r} has shape {shape}, expected [batch, {expected_dim}]"
                )
