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

from sshkeyboard import listen_keyboard, stop_listening


MODEL_PATH = Path(__file__).resolve().parent / "robot_model" / "g1.xml"
DEFAULT_LOWCMD_TOPIC = "rt/lowcmd"
DEFAULT_LOWSTATE_TOPIC = "rt/lowstate"
DEFAULT_NET = "lo"
DEFAULT_SIM_FREQ = 500
DEFAULT_STATE_FREQ = 200
DEFAULT_RENDER_FREQ = 30
DEFAULT_BASE_HEIGHT = 1.0
DEFAULT_HEADLESS = False
DEFAULT_BAND_ENABLED = True
DEFAULT_BAND_SITE_NAMES = ("left_gantry_attach_point", "right_gantry_attach_point")
DEFAULT_BAND_CLEARANCE = 0.35
DEFAULT_BAND_STIFFNESS = 550.0
DEFAULT_BAND_DAMPING = 45.0
DEFAULT_BAND_HEIGHT_STEP = 0.1
DEFAULT_BAND_MIN_HEIGHT = 0.8
DEFAULT_BAND_MAX_HEIGHT = 2.2
DEFAULT_BAND_MAX_FORCE = 400.0
DEFAULT_BASE_QUAT = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)
FALLBACK_KP = np.full(29, 40.0, dtype=np.float64)
FALLBACK_KD = np.full(29, 1.0, dtype=np.float64)
FALLBACK_QPOS = np.zeros(29, dtype=np.float64)
ZERO_TORQUE_KD = np.full(29, 1.0, dtype=np.float64)
CONTROL_MODE_ZERO_TORQUE = "zero_torque"
CONTROL_MODE_DEFAULT_POSE = "default_pose"
CONTROL_MODE_EXTERNAL_CMD = "external_cmd"


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
        self.headless = DEFAULT_HEADLESS
        self.band_enabled = DEFAULT_BAND_ENABLED
        self.band_clearance = DEFAULT_BAND_CLEARANCE
        self.band_stiffness = DEFAULT_BAND_STIFFNESS
        self.band_damping = DEFAULT_BAND_DAMPING
        self.band_height_step = DEFAULT_BAND_HEIGHT_STEP
        self.band_min_height = DEFAULT_BAND_MIN_HEIGHT
        self.band_max_height = DEFAULT_BAND_MAX_HEIGHT
        self.band_max_force = DEFAULT_BAND_MAX_FORCE
        self.is_alive = True
        self.command_received = False
        self.state_tick = 1
        self.mode_machine = 0
        self.mode_pr = 0
        self.control_mode = CONTROL_MODE_ZERO_TORQUE

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
        self.motor_enable = np.zeros(self.num_motor, dtype=bool)

        self.state_lock = threading.Lock()
        self.cmd_lock = threading.Lock()
        self.band_lock = threading.Lock()
        self.keyboard_state_lock = threading.Lock()

        self.viewer = None
        self.viewer_tick = 0
        self.viewer_decim = max(1, self.sim_freq // max(1, self.render_freq))
        self.keyboard_thread = None
        self.band_site_ids: tuple[int, ...] = ()
        self.band_anchor_positions: np.ndarray | None = None
        self.band_target_height = 0.0
        self.pressed_keys: set[str] = set()

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

        if self.band_enabled:
            self.band_site_ids = tuple(
                self._resolve_site_id(site_name) for site_name in DEFAULT_BAND_SITE_NAMES
            )
            anchor_positions = []
            for site_id in self.band_site_ids:
                anchor_position = np.asarray(self.data.site_xpos[site_id], dtype=np.float64).copy()
                anchor_position[2] += self.band_clearance
                anchor_positions.append(anchor_position)
            self.band_anchor_positions = np.asarray(anchor_positions, dtype=np.float64)
            self.band_target_height = float(np.mean(self.band_anchor_positions[:, 2]))

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

        self._set_zero_torque_mode(log=False)

    def _resolve_sensor_slice(self, sensor_name: str):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sid < 0:
            return None, 0
        return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])

    def _resolve_site_id(self, site_name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Missing required MuJoCo site '{site_name}' in XML.")
        return int(site_id)

    def _adjust_band_height(self, delta_height: float) -> None:
        if not self.band_enabled or self.band_anchor_positions is None:
            return

        with self.band_lock:
            self.band_target_height = float(
                np.clip(
                    self.band_target_height + delta_height,
                    self.band_min_height,
                    self.band_max_height,
                )
            )
            self.band_anchor_positions[:, 2] = self.band_target_height
            target_height = self.band_target_height

        print(f"[sim_bridge] band height -> {target_height:.3f} m")

    def _set_zero_torque_mode(self, *, log: bool = True) -> None:
        zero_torque_kd = ZERO_TORQUE_KD[: self.num_motor]
        with self.cmd_lock:
            self.control_mode = CONTROL_MODE_ZERO_TORQUE
            self.target_q.fill(0.0)
            self.target_dq.fill(0.0)
            self.kp.fill(0.0)
            self.kd[:] = zero_torque_kd
            self.tau_ff.fill(0.0)
            self.motor_enable.fill(True)
            self.command_received = False
    
        if log:
            print("[sim_bridge] control mode -> zero torque with damping")

    def _set_default_pose_mode(self) -> None:
        default_qpos = FALLBACK_QPOS[: self.num_motor]
        default_kp = FALLBACK_KP[: self.num_motor]
        default_kd = FALLBACK_KD[: self.num_motor]

        with self.cmd_lock:
            self.control_mode = CONTROL_MODE_DEFAULT_POSE
            self.target_q[:] = default_qpos
            self.target_dq.fill(0.0)
            self.kp[:] = default_kp
            self.kd[:] = default_kd
            self.tau_ff.fill(0.0)
            self.motor_enable.fill(True)
            self.command_received = False

        print("[sim_bridge] control mode -> default pose damping")

    def _set_external_command_mode(self) -> None:
        with self.cmd_lock:
            self.control_mode = CONTROL_MODE_EXTERNAL_CMD
            self.target_q.fill(0.0)
            self.target_dq.fill(0.0)
            self.kp.fill(0.0)
            self.kd.fill(0.0)
            self.tau_ff.fill(0.0)
            self.motor_enable.fill(False)
            self.command_received = False

        print("[sim_bridge] control mode -> external lowcmd")

    def _release_band(self) -> None:
        with self.band_lock:
            self.band_enabled = False

        print("[sim_bridge] suspension bands released")

    def _restore_band_zero_torque(self) -> None:
        with self.band_lock:
            has_band = self.band_anchor_positions is not None
            if has_band:
                self.band_enabled = True
                band_height = float(np.mean(self.band_anchor_positions[:, 2]))
            else:
                band_height = None

        self._set_zero_torque_mode(log=False)

        if band_height is None:
            print("[sim_bridge] control mode -> zero torque (band unavailable)")
            return

        print(f"[sim_bridge] restored suspension at z={band_height:.3f} m, control mode -> zero torque")

    def _on_keyboard_press(self, key: str) -> None:
        normalized_key = key.lower()
        with self.keyboard_state_lock:
            if normalized_key in self.pressed_keys:
                return
            self.pressed_keys.add(normalized_key)

        if normalized_key == "up":
            self._adjust_band_height(self.band_height_step)
        elif normalized_key == "down":
            self._adjust_band_height(-self.band_height_step)
        elif normalized_key == "b":
            self._set_default_pose_mode()
        elif normalized_key == "m":
            self._set_external_command_mode()
        elif normalized_key == "n":
            self._release_band()
        elif normalized_key == "r":
            self._restore_band_zero_torque()
        elif normalized_key == "q":
            self.close()

    def _on_keyboard_release(self, key: str) -> None:
        normalized_key = key.lower()
        with self.keyboard_state_lock:
            self.pressed_keys.discard(normalized_key)

    def _keyboard_control_loop(self) -> None:
        print(
            "[sim_bridge] keyboard: Up/Down move band, b=default pose damping, "
            "n=release band, r=restore band+zero torque, m=enable lowcmd, q=quit."
        )
        try:
            listen_keyboard(
                on_press=self._on_keyboard_press,
                on_release=self._on_keyboard_release,
                until=None,
                sequential=True,
            )
        except Exception as exc:
            if self.is_alive:
                print(f"[sim_bridge] keyboard control stopped: {exc}")

    def _apply_band_support_locked(self):
        if not self.band_enabled or self.band_anchor_positions is None:
            return

        with self.band_lock:
            anchor_positions = self.band_anchor_positions.copy()

        for site_id, anchor_position in zip(self.band_site_ids, anchor_positions):
            site_position = np.asarray(self.data.site_xpos[site_id], dtype=np.float64).copy()
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, None, site_id)
            site_velocity = jacp @ self.data.qvel

            force = self.band_stiffness * (anchor_position - site_position) - self.band_damping * site_velocity
            force = np.clip(force, -self.band_max_force, self.band_max_force)
            body_id = int(self.model.site_bodyid[site_id])
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force,
                np.zeros(3, dtype=np.float64),
                site_position,
                body_id,
                self.data.qfrc_applied,
            )

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
            if self.control_mode != CONTROL_MODE_EXTERNAL_CMD:
                return

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
                self.data.qfrc_applied[:] = 0.0
                self._apply_band_support_locked()

                joint_qpos = self.data.qpos[7 : 7 + self.num_motor]
                joint_qvel = self.data.qvel[6 : 6 + self.num_motor]
                self.data.ctrl[:] = self._compute_ctrl(joint_qpos, joint_qvel)

                mujoco.mj_step(self.model, self.data)

            if not self._viewer_sync():
                break

            steps += 1
            now = time.perf_counter()
            if now - last_log_t >= 1.0:
                print(
                    f"[sim_bridge] t={steps / self.sim_freq:6.2f}s "
                    f"height={self.data.qpos[2]:.3f} "
                    f"mode={self.control_mode} "
                    f"cmd={'yes' if self.command_received else 'no'} "
                    f"band={'on' if self.band_enabled else 'off'}"
                )
                last_log_t = now
            timer.sleep()

    def run(self):
        print(
            f"[sim_bridge] topics: lowcmd={self.lowcmd_topic}, "
            f"lowstate={self.lowstate_topic}"
        )
        print(
            f"[sim_bridge] sim={self.sim_freq}Hz state_pub={self.state_freq}Hz"
        )
        if self.band_enabled:
            print(f"[sim_bridge] suspension bands enabled at z={self.band_target_height:.3f} m")

        self.lowstate_publisher_thread.start()
        self.keyboard_thread = threading.Thread(
            target=self._keyboard_control_loop,
            name="keyboard-control",
            daemon=True,
        )
        self.keyboard_thread.start()

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

        if stop_listening is not None:
            try:
                stop_listening()
            except Exception:
                pass

        if (
            self.keyboard_thread is not None
            and self.keyboard_thread.is_alive()
            and threading.current_thread() is not self.keyboard_thread
        ):
            self.keyboard_thread.join(timeout=1.0)

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
