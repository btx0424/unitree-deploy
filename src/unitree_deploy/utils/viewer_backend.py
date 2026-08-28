from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import glfw
import mujoco
import mujoco.viewer

from unitree_deploy.robot_model.robot_config import RobotModel
from unitree_deploy.utils.yaml_utils import load_yaml


class ViewerBackend(Protocol):
    """Small lifecycle adapter so SimBridge does not branch on viewer type."""

    def run(self, simulate: Callable[[], None]) -> None:
        ...

    def sync(self) -> bool:
        ...


@dataclass(frozen=True)
class ViewerCameraConfig:
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.85)
    distance: float = 2.0
    elevation: float = -15.0
    azimuth: float = 20.0
    track_body: str | None = None
    track_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def load_viewer_camera_config(robot: RobotModel) -> ViewerCameraConfig:
    path = robot.config_dir / "visualizer.yaml"
    if not path.exists():
        return ViewerCameraConfig()

    config = load_yaml(path)
    camera = config.get("viewer", {}).get("camera", {})
    if not isinstance(camera, dict):
        return ViewerCameraConfig()

    defaults = ViewerCameraConfig()
    lookat = camera.get("lookat", defaults.lookat)
    if len(lookat) != 3:
        raise ValueError(f"{path} viewer.camera.lookat must contain 3 values")

    track_offset = camera.get("track_offset", defaults.track_offset)
    if len(track_offset) != 3:
        raise ValueError(f"{path} viewer.camera.track_offset must contain 3 values")

    return ViewerCameraConfig(
        lookat=tuple(float(value) for value in lookat),
        distance=float(camera.get("distance", defaults.distance)),
        elevation=float(camera.get("elevation", defaults.elevation)),
        azimuth=float(camera.get("azimuth", defaults.azimuth)),
        track_body=camera.get("track_body"),
        track_offset=tuple(float(value) for value in track_offset),
    )


class MujocoViewerBackend:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera: ViewerCameraConfig,
        *,
        sim_hz: int,
        render_hz: int,
        log: Callable[[str], None],
    ) -> None:
        self.model = model
        self.data = data
        self.camera = camera
        self.log = log
        self.track_body_id = self.resolve_track_body_id()
        self.tracking_enabled = self.track_body_id is not None
        self.keyboard_masked = False
        self._viewer_window = None
        self._mujoco_key_callback = None
        self._masked_key_callback = glfw._GLFWkeyfun(self.on_masked_key)
        self.viewer = None
        self.viewer_tick = 0
        self.viewer_decim = max(1, sim_hz // render_hz)

    def resolve_track_body_id(self) -> int | None:
        if not self.camera.track_body:
            return None
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.camera.track_body,
        )
        if body_id < 0:
            raise ValueError(f"viewer.camera.track_body not found: {self.camera.track_body}")
        return int(body_id)

    def run(self, simulate: Callable[[], None]) -> None:
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.on_key,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            self.viewer = viewer
            viewer.cam.lookat[:] = self.camera.lookat
            viewer.cam.distance = self.camera.distance
            viewer.cam.elevation = self.camera.elevation
            viewer.cam.azimuth = self.camera.azimuth
            self.log("MuJoCo keyboard: press F12 to toggle native keyboard input")
            if self.track_body_id is not None:
                self.log("MuJoCo camera: press T to toggle robot tracking")
            simulate()
            self.viewer = None

    def on_key(self, key: int) -> None:
        if key == glfw.KEY_F12:
            self.enable_keyboard_mask()
        elif key == glfw.KEY_T:
            self.toggle_tracking()

    def on_masked_key(
        self,
        _window,
        key: int,
        _scancode: int,
        action: int,
        _mods: int,
    ) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_F12:
            self.disable_keyboard_mask()
        elif key == glfw.KEY_T:
            self.toggle_tracking()

    def enable_keyboard_mask(self) -> None:
        if self.keyboard_masked:
            return
        window = glfw.get_current_context()
        if not window:
            self.log("camera keyboard mask unavailable: no current GLFW window")
            return

        # MuJoCo's Python key callback cannot consume events. Replacing the
        # underlying GLFW callback is the only way to block its native keys.
        previous = glfw._glfw.glfwSetKeyCallback(window, self._masked_key_callback)
        if not previous:
            self.log("camera keyboard mask unavailable: MuJoCo callback not found")
            return
        self._viewer_window = window
        self._mujoco_key_callback = previous
        self.keyboard_masked = True
        self.log("MuJoCo keyboard input=masked (F12: unmask, T: camera tracking)")

    def disable_keyboard_mask(self) -> None:
        if not self.keyboard_masked:
            return
        if self._viewer_window is None or self._mujoco_key_callback is None:
            return
        glfw._glfw.glfwSetKeyCallback(self._viewer_window, self._mujoco_key_callback)
        self.keyboard_masked = False
        self.log("MuJoCo keyboard input=enabled")

    def toggle_tracking(self) -> None:
        if self.track_body_id is None:
            return
        self.tracking_enabled = not self.tracking_enabled
        self.log(f"camera tracking={'on' if self.tracking_enabled else 'off'}")

    def sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            return False
        self.viewer_tick += 1
        if self.viewer_tick % self.viewer_decim == 0:
            self.update_tracked_lookat()
            self.viewer.sync()
        return True

    def update_tracked_lookat(self) -> None:
        if self.viewer is None or self.track_body_id is None or not self.tracking_enabled:
            return
        self.viewer.cam.lookat[:] = self.data.xpos[self.track_body_id] + self.camera.track_offset


def create_viewer_backend(
    viewer: str,
    robot: RobotModel,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    sim_hz: int,
    render_hz: int,
    log: Callable[[str], None],
) -> ViewerBackend:
    camera = load_viewer_camera_config(robot)
    if viewer == "mujoco":
        return MujocoViewerBackend(
            model,
            data,
            camera,
            sim_hz=sim_hz,
            render_hz=render_hz,
            log=log,
        )
    raise ValueError(f"unsupported viewer backend: {viewer}")
