import argparse
import time
import os
from dataclasses import dataclass

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

import numpy as np

import mujoco

from standalone_viser_mesh import RealSenseCameraConfig, StandaloneMujocoScene
import viser

from pathlib import Path

MODEL = Path(__file__).parent / "robot_model" / "g1.xml"
DEFAULT_MODE = os.getenv("G1_RUN_MODE", "sim")
DEFAULT_ENABLE_CAMERA = os.getenv("ENABLE_CAMERA", "0").lower() not in {"0", "false", "no"}
DEFAULT_REALSENSE_SERIAL = os.getenv("REALSENSE_SERIAL", "140122071098")
DEFAULT_REALSENSE_CAMERA_NAME = os.getenv("REALSENSE_CAMERA_NAME", "d435_head")
DEFAULT_REALSENSE_POSE_CAMERA_NAME = os.getenv("REALSENSE_POSE_CAMERA_NAME", "d435_head")
DEFAULT_REALSENSE_WIDTH = int(os.getenv("REALSENSE_WIDTH", "640"))
DEFAULT_REALSENSE_HEIGHT = int(os.getenv("REALSENSE_HEIGHT", "480"))
DEFAULT_REALSENSE_FPS = int(os.getenv("REALSENSE_FPS", "30"))
DEFAULT_REALSENSE_ENABLE_DEPTH = os.getenv("REALSENSE_ENABLE_DEPTH", "1").lower() not in {"0", "false", "no"}

G1_NUM_MOTOR = 29

Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    40, 40, 40, 40,  40, 40, 40,  # arms
    40, 40, 40, 40,  40, 40, 40   # arms
]

Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    1, 1, 1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1, 1, 1   # arms 
]

class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    net: str | None
    enable_camera: bool


class Custom:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.time_ = 0.0
        self.control_dt_ = 0.002  # [2ms]
        self.duration_ = 3.0    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()
        self.msc = None
        self.lowcmd_publisher_ = None
        self.lowstate_subscriber = None
        self.odomstate_subscriber = None
        self.model = None
        self.data = None
        self.viser_server = None
        self.viser_scene = None
        self.lowCmdWriteThreadPtr = None
        self.timerPtr = None
        self._control_thread_started = False
        self._visualize_thread_started = False
        self._closed = False

    def Init(self):
        if self.config.mode == "real":
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

            status, result = self.msc.CheckMode()
            while result["name"]:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = np.zeros(self.model.nq)

        self.viser_server = viser.ViserServer()
        real_sense_configs = None
        if self.config.enable_camera:
            real_sense_configs = [
                RealSenseCameraConfig(
                    camera_name=DEFAULT_REALSENSE_CAMERA_NAME,
                    pose_camera_name=DEFAULT_REALSENSE_POSE_CAMERA_NAME,
                    serial_number=DEFAULT_REALSENSE_SERIAL,
                    color_width=DEFAULT_REALSENSE_WIDTH,
                    color_height=DEFAULT_REALSENSE_HEIGHT,
                    depth_width=DEFAULT_REALSENSE_WIDTH,
                    depth_height=DEFAULT_REALSENSE_HEIGHT,
                    fps=DEFAULT_REALSENSE_FPS,
                    enable_depth=DEFAULT_REALSENSE_ENABLE_DEPTH,
                )
            ]
        self.viser_scene = StandaloneMujocoScene.create(
            self.viser_server,
            self.model,
            show_camera_frustums=True,
            real_sense_configs=real_sense_configs,
        )

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        if self.config.mode == "real":
            # create subscriber for estimated state (optional, for visualization) #
            self.odomstate_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
            self.odomstate_subscriber.Init(self.OdomStateHandler, 10)

        # self.time_prev = 0.0

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.update_mode_machine_ == False:
            time.sleep(1)

        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()
            self._control_thread_started = True

        self.timerPtr = RecurrentThread(interval=0.05, target=self.Visualize, name="visualize")
        self.timerPtr.Start()
        self._visualize_thread_started = True

    def OdomStateHandler(self, msg: SportModeState_):
        self.odom_state = msg

        if self.counter_ % 100 == 0 :
            self.data.qpos[:3] = self.odom_state.position
            self.data.qpos[3:7] = self.odom_state.imu_state.quaternion

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
        
        self.counter_ +=1
        if (self.counter_ % 500 == 0) :
            self.counter_ = 0
            print(self.low_state.imu_state.rpy)

        if(self.counter_ % 100 == 0) :
            num_motor = self.model.nu
            self.data.qpos[7:7+num_motor] = [self.low_state.motor_state[i].q for i in range(num_motor)]

    def LowCmdWrite(self):
        self.time_ += self.control_dt_

        if self.time_ < self.duration_ :
            # [Stage 1]: set robot to zero posture
            for i in range(G1_NUM_MOTOR):
                ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0. 
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q 
                self.low_cmd.motor_cmd[i].dq = 0. 
                self.low_cmd.motor_cmd[i].kp = Kp[i] 
                self.low_cmd.motor_cmd[i].kd = Kd[i]

        elif self.time_ < self.duration_ * 2 :
            # [Stage 2]: swing ankle using PR mode
            max_P = np.pi * 30.0 / 180.0
            max_R = np.pi * 10.0 / 180.0
            t = self.time_ - self.duration_
            L_P_des = max_P * np.sin(2.0 * np.pi * t)
            L_R_des = max_R * np.sin(2.0 * np.pi * t)
            R_P_des = max_P * np.sin(2.0 * np.pi * t)
            R_R_des = -max_R * np.sin(2.0 * np.pi * t)

            self.low_cmd.mode_pr = Mode.PR
            self.low_cmd.mode_machine = self.mode_machine_
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = L_P_des
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleRoll].q = L_R_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnklePitch].q = R_P_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnkleRoll].q = R_R_des

        else :
            # [Stage 3]: swing ankle using AB mode
            max_A = np.pi * 30.0 / 180.0
            max_B = np.pi * 10.0 / 180.0
            t = self.time_ - self.duration_ * 2
            L_A_des = max_A * np.sin(2.0 * np.pi * t)
            L_B_des = max_B * np.sin(2.0 * np.pi * t + np.pi)
            R_A_des = -max_A * np.sin(2.0 * np.pi * t)
            R_B_des = -max_B * np.sin(2.0 * np.pi * t + np.pi)

            self.low_cmd.mode_pr = Mode.AB
            self.low_cmd.mode_machine = self.mode_machine_
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleA].q = L_A_des
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleB].q = L_B_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnkleA].q = R_A_des
            self.low_cmd.motor_cmd[G1JointIndex.RightAnkleB].q = R_B_des
            
            max_WristYaw = np.pi * 30.0 / 180.0
            L_WristYaw_des = max_WristYaw * np.sin(2.0 * np.pi * t)
            R_WristYaw_des = max_WristYaw * np.sin(2.0 * np.pi * t)
            self.low_cmd.motor_cmd[G1JointIndex.LeftWristRoll].q = L_WristYaw_des
            self.low_cmd.motor_cmd[G1JointIndex.RightWristRoll].q = R_WristYaw_des
    

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def Visualize(self):
        if self.viser_scene is None:
            return

        # if time.time() - self.time_prev > 0.05:  # visualize at 20 Hz
        self.data.qpos[7:7+G1_NUM_MOTOR] = [self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)]
        mujoco.mj_forward(self.model, self.data)
        self.viser_scene.update_from_mjdata(self.data)
        self.time_prev = time.time()

    def Close(self):
        if self._closed:
            return
        self._closed = True

        if self._visualize_thread_started and self.timerPtr is not None:
            try:
                self.timerPtr.Wait(1.0)
            except Exception as exc:
                print(f"[WARN] Failed to stop visualize thread: {exc}")

        if self._control_thread_started and self.lowCmdWriteThreadPtr is not None:
            try:
                self.lowCmdWriteThreadPtr.Wait(1.0)
            except Exception as exc:
                print(f"[WARN] Failed to stop control thread: {exc}")

        if self.lowstate_subscriber is not None:
            try:
                self.lowstate_subscriber.Close()
            except Exception as exc:
                print(f"[WARN] Failed to close lowstate subscriber: {exc}")

        if self.odomstate_subscriber is not None:
            try:
                self.odomstate_subscriber.Close()
            except Exception as exc:
                print(f"[WARN] Failed to close odomstate subscriber: {exc}")

        if self.lowcmd_publisher_ is not None:
            try:
                self.lowcmd_publisher_.Close()
            except Exception as exc:
                print(f"[WARN] Failed to close lowcmd publisher: {exc}")

        if self.viser_scene is not None:
            try:
                self.viser_scene.close()
            except Exception as exc:
                print(f"[WARN] Failed to close viser scene: {exc}")

        if self.viser_server is not None:
            try:
                self.viser_server.stop()
            except Exception as exc:
                print(f"[WARN] Failed to stop viser server: {exc}")


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="G1 low-level example for real robot or simulation.")
    parser.add_argument("--net", default='lo', help="Optional DDS network interface.")
    parser.add_argument(
        "--mode",
        choices=("real", "sim"),
        default=DEFAULT_MODE,
        help="Run against a real robot or the sim bridge.",
    )
    parser.add_argument(
        "--camera",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_CAMERA,
        help="Enable or disable camera-related visualization and capture.",
    )
    args = parser.parse_args()
    return RuntimeConfig(
        mode=args.mode,
        net=args.net,
        enable_camera=bool(args.camera),
    )

if __name__ == '__main__':
    config = parse_args()

    if config.mode == "real":
        print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
        input("Press Enter to continue...")
    else:
        print("[INFO] Running in simulation mode.")

    if config.net:
        ChannelFactoryInitialize(0, config.net)
    else:
        ChannelFactoryInitialize(0)

    custom = Custom(config)
    try:
        custom.Init()
        custom.Start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        custom.Close()
