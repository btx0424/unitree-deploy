import argparse
import signal
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


NUM_JOINTS = 29
DEFAULT_LOWCMD_TOPIC = "rt/lowcmd"
DEFAULT_LOWSTATE_TOPIC = "rt/lowstate"
DEFAULT_MODE = "sim"
DEFAULT_NET = "lo"
CONTROL_DT = 0.002
MOVE_TO_DEFAULT_DURATION = 2.0
ACTION_SCALE = 0.25
ACTION_CLIP = 1.0

DEFAULT_QPOS = np.zeros(NUM_JOINTS, dtype=np.float64)
DEFAULT_KP = np.array(
    [
        60.0, 60.0, 60.0, 100.0, 40.0, 40.0,
        60.0, 60.0, 60.0, 100.0, 40.0, 40.0,
        60.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    ],
    dtype=np.float64,
)
DEFAULT_KD = np.array(
    [
        1.0, 1.0, 1.0, 2.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 2.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ],
    dtype=np.float64,
)
ZERO_TORQUE_KD = np.full(NUM_JOINTS, 1.0, dtype=np.float64)

STATE_ZERO_TORQUE = "zero_torque_state"
STATE_MOVE_TO_DEFAULT = "move_to_default_qpos"
STATE_DEFAULT_QPOS = "default_qpos_state"
STATE_RUN = "run"


class LoopTimer:
    def __init__(self, dt: float):
        self.dt = float(dt)
        self.next_t = time.perf_counter() + self.dt

    def sleep(self) -> None:
        now = time.perf_counter()
        sleep_t = self.next_t - now
        if sleep_t > 0.0:
            time.sleep(sleep_t)
            self.next_t += self.dt
            return
        self.next_t = now + self.dt


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    net: str | None


class RemoteCommand:
    def __init__(self) -> None:
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.L1 = False
        self.L2 = False
        self.R1 = False
        self.R2 = False
        self.A = False
        self.B = False
        self.X = False
        self.Y = False
        self.Up = False
        self.Down = False
        self.Left = False
        self.Right = False
        self.Start = False
        self.Select = False
        self.F1 = False
        self.F3 = False
        self._pressed_edges: set[str] = set()

    def _update_button(self, name: str, value: bool) -> None:
        previous = getattr(self, name)
        current = bool(value)
        if current and not previous:
            self._pressed_edges.add(name)
        setattr(self, name, current)

    def _set_axes_from_wireless_remote(self, wireless_remote) -> None:
        payload = bytes(wireless_remote)
        self.lx = struct.unpack("<f", payload[4:8])[0]
        self.rx = struct.unpack("<f", payload[8:12])[0]
        self.ry = struct.unpack("<f", payload[12:16])[0]
        self.ly = struct.unpack("<f", payload[20:24])[0]

    def set(self, wireless_remote) -> None:
        self._set_axes_from_wireless_remote(wireless_remote)
        data1 = int(wireless_remote[2])
        data2 = int(wireless_remote[3])

        self._update_button("R1", (data1 >> 0) & 1)
        self._update_button("L1", (data1 >> 1) & 1)
        self._update_button("Start", (data1 >> 2) & 1)
        self._update_button("Select", (data1 >> 3) & 1)
        self._update_button("R2", (data1 >> 4) & 1)
        self._update_button("L2", (data1 >> 5) & 1)
        self._update_button("F1", (data1 >> 6) & 1)
        self._update_button("F3", (data1 >> 7) & 1)
        self._update_button("A", (data2 >> 0) & 1)
        self._update_button("B", (data2 >> 1) & 1)
        self._update_button("X", (data2 >> 2) & 1)
        self._update_button("Y", (data2 >> 3) & 1)
        self._update_button("Up", (data2 >> 4) & 1)
        self._update_button("Right", (data2 >> 5) & 1)
        self._update_button("Down", (data2 >> 6) & 1)
        self._update_button("Left", (data2 >> 7) & 1)

    def consume_pressed(self, name: str) -> bool:
        if name in self._pressed_edges:
            self._pressed_edges.remove(name)
            return True
        return False


class Controller:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.lowcmd_topic = DEFAULT_LOWCMD_TOPIC
        self.lowstate_topic = DEFAULT_LOWSTATE_TOPIC
        self.control_dt = CONTROL_DT
        self.is_alive = True

        self.lowstate_subscriber = ChannelSubscriber(self.lowstate_topic, LowState_)
        self.lowstate_lock = threading.Lock()
        self.lowstate_subscriber.Init(self._consume_low_state, 1)
        self.lowcmd_publisher = ChannelPublisher(self.lowcmd_topic, LowCmd_)
        self.lowcmd_publisher.Init()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.policy = None

        self.q = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.dq = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.tau = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.gyro = np.zeros(3, dtype=np.float64)
        self.lin_acc = np.zeros(3, dtype=np.float64)
        self.command = np.zeros(3, dtype=np.float64)
        self.obs = np.zeros(2 * NUM_JOINTS + 10, dtype=np.float64)
        self.remote = RemoteCommand()

        self.has_low_state = False
        self.mode_machine = 0
        self.mode_pr = 0
        self.cleanup_done = False

        self.state = STATE_ZERO_TORQUE
        self.state_enter_t = time.perf_counter()
        self.move_start_q = np.zeros(NUM_JOINTS, dtype=np.float64)

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

    def set_policy(self, policy) -> None:
        self.policy = policy

    def _transition_to(self, state: str) -> None:
        if self.state == state:
            return
        self.state = state
        self.state_enter_t = time.perf_counter()
        if state == STATE_MOVE_TO_DEFAULT:
            with self.lowstate_lock:
                self.move_start_q = self.q.copy()
        print(f"[deploy] state -> {self.state}")

    def _consume_low_state(self, msg: LowState_) -> None:
        with self.lowstate_lock:
            self.mode_machine = int(getattr(msg, "mode_machine", self.mode_machine))
            self.mode_pr = int(getattr(msg, "mode_pr", self.mode_pr))

            for i in range(NUM_JOINTS):
                motor_state = msg.motor_state[i]
                self.q[i] = float(motor_state.q)
                self.dq[i] = float(motor_state.dq)
                self.tau[i] = float(motor_state.tau_est)

            self.quat[:] = np.asarray(msg.imu_state.quaternion[:4], dtype=np.float64)
            self.gyro[:] = np.asarray(msg.imu_state.gyroscope[:3], dtype=np.float64)
            self.lin_acc[:] = np.asarray(msg.imu_state.accelerometer[:3], dtype=np.float64)

            wireless_remote = getattr(msg, "wireless_remote", [0] * 40)
            self.remote.set(wireless_remote)

            self.command[:] = np.array(
                [self.remote.ly, self.remote.lx, self.remote.rx],
                dtype=np.float64,
            )
            self.has_low_state = True

    def _has_low_state(self) -> bool:
        with self.lowstate_lock:
            return self.has_low_state

    def _consume_button(self, name: str) -> bool:
        with self.lowstate_lock:
            return self.remote.consume_pressed(name)

    def _fill_low_cmd(
        self,
        target_q: np.ndarray,
        target_dq: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        tau_ff: np.ndarray,
        *,
        enable: bool,
    ) -> None:
        with self.lowstate_lock:
            mode_pr = int(self.mode_pr)
            mode_machine = int(self.mode_machine)

        self.low_cmd.mode_pr = mode_pr
        self.low_cmd.mode_machine = mode_machine

        for motor_cmd in self.low_cmd.motor_cmd:
            motor_cmd.mode = 0
            motor_cmd.q = 0.0
            motor_cmd.dq = 0.0
            motor_cmd.tau = 0.0
            motor_cmd.kp = 0.0
            motor_cmd.kd = 0.0

        for i in range(NUM_JOINTS):
            motor_cmd = self.low_cmd.motor_cmd[i]
            motor_cmd.mode = 1 if enable else 0
            motor_cmd.q = float(target_q[i])
            motor_cmd.dq = float(target_dq[i])
            motor_cmd.tau = float(tau_ff[i])
            motor_cmd.kp = float(kp[i])
            motor_cmd.kd = float(kd[i])

    def send_cmd(self) -> None:
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def update_obs(self) -> np.ndarray:
        with self.lowstate_lock:
            q = self.q.copy()
            dq = self.dq.copy()
            quat = self.quat.copy()
            gyro = self.gyro.copy()
            command = self.command.copy()

        self.obs = np.concatenate((q - DEFAULT_QPOS, dq, quat, gyro, command)).astype(
            np.float64, copy=False
        )
        return self.obs

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        if self.policy is None:
            return np.zeros(NUM_JOINTS, dtype=np.float64)

        action = self.policy(obs)
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] < NUM_JOINTS:
            padded = np.zeros(NUM_JOINTS, dtype=np.float64)
            padded[: action.shape[0]] = action
            action = padded
        elif action.shape[0] > NUM_JOINTS:
            action = action[:NUM_JOINTS]
        return np.clip(action, -ACTION_CLIP, ACTION_CLIP)

    def zero_torque_state(self) -> None:
        if self._consume_button("A"):
            self._transition_to(STATE_MOVE_TO_DEFAULT)
            return

        with self.lowstate_lock:
            current_q = self.q.copy()

        self._fill_low_cmd(
            target_q=current_q,
            target_dq=np.zeros(NUM_JOINTS, dtype=np.float64),
            kp=np.zeros(NUM_JOINTS, dtype=np.float64),
            kd=ZERO_TORQUE_KD,
            tau_ff=np.zeros(NUM_JOINTS, dtype=np.float64),
            enable=True,
        )
        self.send_cmd()

    def move_to_default_qpos(self) -> None:
        if self._consume_button("X"):
            self._transition_to(STATE_ZERO_TORQUE)
            return

        elapsed = time.perf_counter() - self.state_enter_t
        ratio = float(np.clip(elapsed / MOVE_TO_DEFAULT_DURATION, 0.0, 1.0))
        target_q = (1.0 - ratio) * self.move_start_q + ratio * DEFAULT_QPOS

        self._fill_low_cmd(
            target_q=target_q,
            target_dq=np.zeros(NUM_JOINTS, dtype=np.float64),
            kp=DEFAULT_KP,
            kd=DEFAULT_KD,
            tau_ff=np.zeros(NUM_JOINTS, dtype=np.float64),
            enable=True,
        )
        self.send_cmd()

        if ratio >= 1.0:
            self._transition_to(STATE_DEFAULT_QPOS)

    def default_qpos_state(self) -> None:
        if self._consume_button("X"):
            self._transition_to(STATE_ZERO_TORQUE)
            return
        if self._consume_button("Start"):
            self._transition_to(STATE_RUN)
            return

        self._fill_low_cmd(
            target_q=DEFAULT_QPOS,
            target_dq=np.zeros(NUM_JOINTS, dtype=np.float64),
            kp=DEFAULT_KP,
            kd=DEFAULT_KD,
            tau_ff=np.zeros(NUM_JOINTS, dtype=np.float64),
            enable=True,
        )
        self.send_cmd()

    def run(self) -> None:
        if self._consume_button("X"):
            self._transition_to(STATE_ZERO_TORQUE)
            return

        obs = self.update_obs()
        action = self.compute_action(obs)
        target_q = DEFAULT_QPOS + ACTION_SCALE * action

        self._fill_low_cmd(
            target_q=target_q,
            target_dq=np.zeros(NUM_JOINTS, dtype=np.float64),
            kp=DEFAULT_KP,
            kd=DEFAULT_KD,
            tau_ff=np.zeros(NUM_JOINTS, dtype=np.float64),
            enable=True,
        )
        self.send_cmd()

    def spin(self) -> None:
        print(
            f"[deploy] mode={self.config.mode} "
            f"topics: lowstate={self.lowstate_topic}, lowcmd={self.lowcmd_topic}"
        )
        if self.config.mode == "sim":
            print("[deploy] sim keymap: b->A, m->Start, r->X")
        print("[deploy] A: zero torque -> default pose, Start: default pose -> run, X: back to zero torque")
        print("[deploy] waiting for lowstate...")

        timer = LoopTimer(self.control_dt)
        last_log_t = time.perf_counter()

        while self.is_alive:
            if not self._has_low_state():
                timer.sleep()
                continue

            if self.state == STATE_ZERO_TORQUE:
                self.zero_torque_state()
            elif self.state == STATE_MOVE_TO_DEFAULT:
                self.move_to_default_qpos()
            elif self.state == STATE_DEFAULT_QPOS:
                self.default_qpos_state()
            elif self.state == STATE_RUN:
                self.run()

            now = time.perf_counter()
            if now - last_log_t >= 1.0:
                with self.lowstate_lock:
                    command = self.command.copy()
                print(
                    f"[deploy] state={self.state} "
                    f"cmd=({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f})"
                )
                last_log_t = now

            timer.sleep()

    def close(self, *_args) -> None:
        if not self.is_alive:
            return
        self.is_alive = False

    def cleanup(self) -> None:
        if self.cleanup_done:
            return
        self.cleanup_done = True

        self.is_alive = False

        if self.lowstate_subscriber is not None and hasattr(self.lowstate_subscriber, "Close"):
            try:
                self.lowstate_subscriber.Close()
            except Exception:
                print("[deploy] failed to close lowstate subscriber")

        if self.lowcmd_publisher is not None and hasattr(self.lowcmd_publisher, "Close"):
            try:
                self.lowcmd_publisher.Close()
            except Exception:
                print("[deploy] failed to close lowcmd publisher")


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Reusable G1 deploy controller for sim or real robot.")
    parser.add_argument(
        "--mode",
        choices=("real", "sim"),
        default=DEFAULT_MODE,
        help="Decode wireless remote as sim2sim keyboard input or real controller input.",
    )
    parser.add_argument(
        "--net",
        default=DEFAULT_NET,
        help="DDS network interface. Use 'lo' for local sim by default.",
    )
    args = parser.parse_args()
    return RuntimeConfig(mode=args.mode, net=args.net)


if __name__ == "__main__":
    config = parse_args()

    if config.mode == "real":
        print("WARNING: Please ensure there are no obstacles around the robot while running deploy.py.")
        input("Press Enter to continue...")

    if config.net:
        ChannelFactoryInitialize(0, config.net)
    else:
        ChannelFactoryInitialize(0)

    controller = Controller(config)
    try:
        controller.spin()
    finally:
        controller.cleanup()
