import math
import signal
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


MODEL_PATH = Path(__file__).resolve().parent / "robot_model" / "g1.xml"
DEFAULT_LOWCMD_TOPIC = "rt/lowcmd"
DEFAULT_LOWSTATE_TOPIC = "rt/lowstate"
DEFAULT_NET = "lo"
DEFAULT_SIM_FREQ = 500
DEFAULT_STATE_FREQ = 200
DEFAULT_RENDER_FREQ = 30
DEFAULT_BASE_HEIGHT = 1.0
DEFAULT_PIN_BASE = True
DEFAULT_HEADLESS = False
DEFAULT_BASE_QUAT = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)
FALLBACK_KP = np.full(29, 40.0, dtype=np.float64)
FALLBACK_KD = np.full(29, 1.0, dtype=np.float64)
FALLBACK_QPOS = np.zeros(29, dtype=np.float64)


class LoopTimer:
    def __init__(self, dt: float):
        self.dt = float(dt)
        self.next_t = time.perf_counter() + self.dt

    def sleep(self):
        now = time.perf_counter()
        sleep_t = self.next_t - now
        if sleep_t > 0.0:
            time.sleep(sleep_t)
            self.next_t += self.dt
            return

        # Resync after overruns to avoid accumulating drift.
        self.next_t = now + self.dt

def quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


class SimBridge:
    def __init__(self):
        self.lowcmd_topic = DEFAULT_LOWCMD_TOPIC
        self.lowstate_topic = DEFAULT_LOWSTATE_TOPIC
        self.sim_freq = DEFAULT_SIM_FREQ
        self.state_freq = DEFAULT_STATE_FREQ
        self.render_freq = DEFAULT_RENDER_FREQ
        self.pin_base = DEFAULT_PIN_BASE
        self.headless = DEFAULT_HEADLESS
        self.is_alive = True
        self.command_received = False
        self.state_tick = 1
        self.mode_machine = 0
        self.mode_pr = 0

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.model.opt.timestep = 1.0 / self.sim_freq
        self.data = mujoco.MjData(self.model)
        self.num_motor = int(self.model.nu)

        self.ctrl_lower = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_upper = self.model.actuator_ctrlrange[:, 1].copy()

        self.base_qpos = np.array(
            [0.0, 0.0, DEFAULT_BASE_HEIGHT, *DEFAULT_BASE_QUAT], dtype=np.float64
        )
        initial_qpos = FALLBACK_QPOS[: self.num_motor].copy()
        if initial_qpos.shape[0] != self.num_motor:
            initial_qpos = np.zeros(self.num_motor, dtype=np.float64)

        self.data.qpos[:7] = self.base_qpos
        self.data.qpos[7 : 7 + self.num_motor] = initial_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.target_q = self.data.qpos[7 : 7 + self.num_motor].copy()
        self.target_dq = np.zeros(self.num_motor, dtype=np.float64)
        self.kp = FALLBACK_KP[: self.num_motor].copy()
        self.kd = FALLBACK_KD[: self.num_motor].copy()
        if self.kp.shape[0] != self.num_motor:
            self.kp = FALLBACK_KP[: self.num_motor].copy()
        if self.kd.shape[0] != self.num_motor:
            self.kd = FALLBACK_KD[: self.num_motor].copy()
        self.tau_ff = np.zeros(self.num_motor, dtype=np.float64)
        self.motor_enable = np.ones(self.num_motor, dtype=bool)

        self.state_lock = threading.Lock()
        self.cmd_lock = threading.Lock()

        self.viewer = None
        self.viewer_tick = 0
        self.viewer_decim = max(1, self.sim_freq // max(1, self.render_freq))

        self.imu_ang_vel_adr, self.imu_ang_vel_dim = self._resolve_sensor_slice(
            "imu_ang_vel"
        )
        self.imu_lin_acc_adr, self.imu_lin_acc_dim = self._resolve_sensor_slice(
            "imu_lin_acc"
        )
        if self.imu_lin_acc_adr is None or self.imu_lin_acc_dim < 3:
            raise ValueError(
                "Missing required MuJoCo sensor 'imu_lin_acc' (dim >= 3) in XML."
            )

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()

        self.lowstate_publisher = ChannelPublisher(self.lowstate_topic, LowState_)
        self.lowstate_publisher.Init()
        self.lowstate_publisher_thread = threading.Thread(
            target=self.lowstate_handler,
            name="lowstate-publisher",
            daemon=False,
        )

        self.lowcmd_subscriber = ChannelSubscriber(self.lowcmd_topic, LowCmd_)
        self.lowcmd_subscriber.Init(self.lowcmd_subscriber_callback)

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

    def _resolve_sensor_slice(self, sensor_name: str):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sid < 0:
            return None, 0
        return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])

    def _apply_base_constraint_locked(self):
        if not self.pin_base:
            return
        self.data.qpos[:7] = self.base_qpos
        self.data.qvel[:6] = 0.0

    def _viewer_sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            self.is_alive = False
            return False

        self.viewer_tick += 1
        if self.viewer_tick % self.viewer_decim == 0:
            self.viewer.sync()
        return True

    def lowcmd_subscriber_callback(self, msg: LowCmd_):
        if msg is None:
            return

        with self.cmd_lock:
            self.low_cmd = msg
            self.mode_machine = int(getattr(msg, "mode_machine", self.mode_machine))
            self.mode_pr = int(getattr(msg, "mode_pr", self.mode_pr))

            motor_count = min(self.num_motor, len(msg.motor_cmd))
            for i in range(motor_count):
                motor_cmd = msg.motor_cmd[i]
                self.target_q[i] = float(motor_cmd.q)
                self.target_dq[i] = float(motor_cmd.dq)
                self.kp[i] = float(motor_cmd.kp)
                self.kd[i] = float(motor_cmd.kd)
                self.tau_ff[i] = float(motor_cmd.tau)
                self.motor_enable[i] = int(getattr(motor_cmd, "mode", 1)) != 0

            self.command_received = True

    def build_low_state(self) -> unitree_hg_msg_dds__LowState_:
        with self.state_lock:
            joint_qpos = self.data.qpos[7 : 7 + self.num_motor].copy()
            joint_qvel = self.data.qvel[6 : 6 + self.num_motor].copy()
            joint_torque = self.data.ctrl[: self.num_motor].copy()
            base_quat = self.data.qpos[3:7].copy()
            if self.imu_ang_vel_adr is not None and self.imu_ang_vel_dim >= 3:
                imu_gyro = self.data.sensordata[
                    self.imu_ang_vel_adr : self.imu_ang_vel_adr + 3
                ].copy()
            else:
                imu_gyro = self.data.qvel[3:6].copy()
            imu_acc = self.data.sensordata[
                self.imu_lin_acc_adr : self.imu_lin_acc_adr + 3
            ].copy()

        msg = unitree_hg_msg_dds__LowState_()
        msg.mode_machine = int(self.mode_machine)
        msg.tick = int(self.state_tick)

        for i in range(self.num_motor):
            msg.motor_state[i].q = float(joint_qpos[i])
            msg.motor_state[i].dq = float(joint_qvel[i])
            msg.motor_state[i].tau_est = float(joint_torque[i])

        msg.imu_state.quaternion = base_quat.tolist()
        msg.imu_state.gyroscope = imu_gyro.tolist()
        msg.imu_state.accelerometer = imu_acc.tolist()
        if hasattr(msg.imu_state, "rpy"):
            msg.imu_state.rpy = quat_wxyz_to_rpy(base_quat).tolist()

        msg.crc = CRC().Crc(msg)
        self.state_tick += 1
        self.low_state = msg
        return msg

    def lowstate_handler(self):
        timer = LoopTimer(1.0 / self.state_freq)
        while self.is_alive:
            msg = self.build_low_state()
            self.lowstate_publisher.Write(msg)
            timer.sleep()

    def _compute_ctrl(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        with self.cmd_lock:
            target_q = self.target_q.copy()
            target_dq = self.target_dq.copy()
            kp = self.kp.copy()
            kd = self.kd.copy()
            tau_ff = self.tau_ff.copy()
            motor_enable = self.motor_enable.copy()

        ctrl = kp * (target_q - qpos) + kd * (target_dq - qvel) + tau_ff
        ctrl = np.where(motor_enable, ctrl, 0.0)
        return np.clip(ctrl, self.ctrl_lower, self.ctrl_upper)

    def simulate(self):
        timer = LoopTimer(1.0 / self.sim_freq)
        last_log_t = time.perf_counter()
        steps = 0

        while self.is_alive:
            with self.state_lock:
                self._apply_base_constraint_locked()

                joint_qpos = self.data.qpos[7 : 7 + self.num_motor]
                joint_qvel = self.data.qvel[6 : 6 + self.num_motor]
                self.data.ctrl[:] = self._compute_ctrl(joint_qpos, joint_qvel)

                mujoco.mj_step(self.model, self.data)

                if self.pin_base:
                    self._apply_base_constraint_locked()
                    mujoco.mj_forward(self.model, self.data)

            if not self._viewer_sync():
                break

            steps += 1
            now = time.perf_counter()
            if now - last_log_t >= 1.0:
                print(
                    f"[sim_bridge] t={steps / self.sim_freq:6.2f}s "
                    f"height={self.data.qpos[2]:.3f} "
                    f"cmd={'yes' if self.command_received else 'no'}"
                )
                last_log_t = now
            timer.sleep()

    def run(self):
        print(
            f"[sim_bridge] topics: lowcmd={self.lowcmd_topic}, "
            f"lowstate={self.lowstate_topic}"
        )
        print(
            f"[sim_bridge] sim={self.sim_freq}Hz state_pub={self.state_freq}Hz "
            f"pin_base={self.pin_base}"
        )

        self.lowstate_publisher_thread.start()

        if self.headless:
            self.simulate()
            return

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            self.viewer = viewer
            try:
                self.simulate()
            finally:
                self.viewer = None

    def close(self, *_args):
        if not self.is_alive:
            return

        print("[sim_bridge] shutting down...")
        self.is_alive = False

        if (
            self.lowstate_publisher_thread.is_alive()
            and threading.current_thread() is not self.lowstate_publisher_thread
        ):
            self.lowstate_publisher_thread.join(timeout=1.0)

        if self.lowcmd_subscriber is not None and hasattr(self.lowcmd_subscriber, "Close"):
            try:
                self.lowcmd_subscriber.Close()
            except Exception:
                pass

        if self.lowstate_publisher is not None and hasattr(self.lowstate_publisher, "Close"):
            try:
                self.lowstate_publisher.Close()
            except Exception:
                pass
if __name__ == "__main__":
    ChannelFactoryInitialize(0, DEFAULT_NET)

    bridge = SimBridge()
    try:
        bridge.run()
    finally:
        bridge.close()
