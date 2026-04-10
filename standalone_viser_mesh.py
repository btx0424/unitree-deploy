from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import trimesh.visual
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "unitree-deploy" / "robot_model" / "g1.xml"


@dataclass
class BodyMesh:
    body_id: int
    body_name: str
    mesh: trimesh.Trimesh
    pos: np.ndarray
    wxyz: np.ndarray
    fixed: bool


@dataclass
class SiteMesh:
    site_id: int
    site_name: str
    body_id: int
    body_name: str
    mesh: trimesh.Trimesh
    pos: np.ndarray
    wxyz: np.ndarray
    fixed: bool


@dataclass
class RealSenseFrame:
    camera_name: str
    serial_number: str | None
    color: np.ndarray | None
    depth: np.ndarray | None
    timestamp_sec: float | None


@dataclass
class RealSenseCameraConfig:
    camera_name: str = "realsense"
    pose_camera_name: str | None = None
    serial_number: str | None = None
    color_width: int = 640
    color_height: int = 480
    depth_width: int = 640
    depth_height: int = 480
    fps: int = 30
    enable_depth: bool = True
    align_depth_to_color: bool = True
    frustum_scale: float = 0.15
    jpeg_quality: int | None = 80


def _body_name(mj_model: mujoco.MjModel, body_id: int) -> str:
    name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
    return name or f"body_{body_id}"


def _site_name(mj_model: mujoco.MjModel, site_id: int) -> str:
    name = mj_id2name(mj_model, mjtObj.mjOBJ_SITE, site_id)
    return name or f"site_{site_id}"


def _camera_id_by_name(mj_model: mujoco.MjModel, camera_name: str | None) -> int | None:
    if not camera_name:
        return None

    camera_id = mujoco.mj_name2id(mj_model, mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        return None
    return int(camera_id)


def _opencv_pose_from_mujoco_matrix(position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # MuJoCo cameras use +Y up and look along -Z, while viser frustums follow
    # the OpenCV convention of +Y down and +Z forward.
    rot_180_x = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    adjusted_rotation = rotation @ rot_180_x
    wxyz = vtf.SO3.from_matrix(adjusted_rotation).wxyz.astype(np.float32)
    return position.astype(np.float32), wxyz


def _camera_pose(mj_data: mujoco.MjData, camera_id: int) -> tuple[np.ndarray, np.ndarray]:
    cam_pos = np.asarray(mj_data.cam_xpos[camera_id], dtype=np.float32)
    cam_mat = np.asarray(mj_data.cam_xmat[camera_id], dtype=np.float32).reshape(3, 3)
    return _opencv_pose_from_mujoco_matrix(cam_pos, cam_mat)


def _is_fixed_body(mj_model: mujoco.MjModel, body_id: int) -> bool:
    is_weld = int(mj_model.body_weldid[body_id]) == 0
    root_id = int(mj_model.body_rootid[body_id])
    root_is_mocap = int(mj_model.body_mocapid[root_id]) >= 0
    return is_weld and not root_is_mocap


def _is_collision_geom(mj_model: mujoco.MjModel, geom_id: int) -> bool:
    return (
        int(mj_model.geom_contype[geom_id]) != 0
        or int(mj_model.geom_conaffinity[geom_id]) != 0
    )


def _geom_rgba(mj_model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    mat_id = int(mj_model.geom_matid[geom_id])
    if 0 <= mat_id < int(mj_model.nmat):
        rgba = np.asarray(mj_model.mat_rgba[mat_id], dtype=np.float32)
    else:
        rgba = np.asarray(mj_model.geom_rgba[geom_id], dtype=np.float32)
    if np.allclose(rgba, 0.0):
        rgba = np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32)
    return np.clip(rgba, 0.0, 1.0)


def _paint_mesh(mesh: trimesh.Trimesh, rgba: np.ndarray) -> trimesh.Trimesh:
    colors = np.tile((rgba * 255).astype(np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    return mesh


def _site_rgba(mj_model: mujoco.MjModel, site_id: int) -> np.ndarray:
    rgba = np.asarray(mj_model.site_rgba[site_id], dtype=np.float32)
    if np.allclose(rgba, 0.0):
        rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)
    return np.clip(rgba, 0.0, 1.0)


def _primitive_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    size = np.asarray(mj_model.geom_size[geom_id], dtype=np.float32)
    geom_type = int(mj_model.geom_type[geom_id])

    if geom_type == mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    elif geom_type == mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif geom_type == mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(2.0 * size[1]))
    elif geom_type == mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(2.0 * size[1]))
    elif geom_type == mjtGeom.mjGEOM_PLANE:
        sx = float(2.0 * size[0]) if size[0] > 0 else 20.0
        sy = float(2.0 * size[1]) if size[1] > 0 else 20.0
        mesh = trimesh.creation.box(extents=(sx, sy, 0.001))
    elif geom_type == mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
        mesh.apply_scale(size)
    else:
        raise ValueError(f"Unsupported geom type: {geom_type}")

    return _paint_mesh(mesh, _geom_rgba(mj_model, geom_id))


def _mesh_geom_to_trimesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    mesh_id = int(mj_model.geom_dataid[geom_id])
    vert_adr = int(mj_model.mesh_vertadr[mesh_id])
    vert_num = int(mj_model.mesh_vertnum[mesh_id])
    face_adr = int(mj_model.mesh_faceadr[mesh_id])
    face_num = int(mj_model.mesh_facenum[mesh_id])

    vertices = np.asarray(mj_model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float32)
    faces = np.asarray(mj_model.mesh_face[face_adr : face_adr + face_num], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return _paint_mesh(mesh, _geom_rgba(mj_model, geom_id))


def geom_to_trimesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    geom_type = int(mj_model.geom_type[geom_id])
    if geom_type == mjtGeom.mjGEOM_MESH:
        mesh = _mesh_geom_to_trimesh(mj_model, geom_id)
    else:
        mesh = _primitive_mesh(mj_model, geom_id)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = vtf.SO3(np.asarray(mj_model.geom_quat[geom_id], dtype=np.float32)).as_matrix()
    transform[:3, 3] = np.asarray(mj_model.geom_pos[geom_id], dtype=np.float32)
    mesh.apply_transform(transform)
    return mesh


def site_to_trimesh(mj_model: mujoco.MjModel, site_id: int) -> trimesh.Trimesh:
    size = np.asarray(mj_model.site_size[site_id], dtype=np.float32)
    site_type = int(mj_model.site_type[site_id])

    if site_type == mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    elif site_type == mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif site_type == mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(2.0 * size[1]))
    elif site_type == mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(2.0 * size[1]))
    elif site_type == mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
        mesh.apply_scale(size)
    else:
        raise ValueError(f"Unsupported site type: {site_type}")

    return _paint_mesh(mesh, _site_rgba(mj_model, site_id))


def extract_body_meshes(
    mj_model: mujoco.MjModel,
    *,
    include_collision: bool = False,
    skip_plane_geoms: bool = False,
) -> list[BodyMesh]:
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    body_geoms: dict[int, list[int]] = {}
    for geom_id in range(int(mj_model.ngeom)):
        if not include_collision and _is_collision_geom(mj_model, geom_id):
            continue
        if skip_plane_geoms and int(mj_model.geom_type[geom_id]) == mjtGeom.mjGEOM_PLANE:
            continue
        body_id = int(mj_model.geom_bodyid[geom_id])
        body_geoms.setdefault(body_id, []).append(geom_id)

    body_meshes: list[BodyMesh] = []
    for body_id, geom_ids in body_geoms.items():
        meshes = []
        for geom_id in geom_ids:
            try:
                meshes.append(geom_to_trimesh(mj_model, geom_id))
            except ValueError:
                continue
        if not meshes:
            continue

        merged = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        xmat = np.asarray(mj_data.xmat[body_id], dtype=np.float32).reshape(3, 3)
        body_meshes.append(
            BodyMesh(
                body_id=body_id,
                body_name=_body_name(mj_model, body_id),
                mesh=merged,
                pos=np.asarray(mj_data.xpos[body_id], dtype=np.float32),
                wxyz=vtf.SO3.from_matrix(xmat).wxyz.astype(np.float32),
                fixed=_is_fixed_body(mj_model, body_id),
            )
        )
    return body_meshes


def extract_site_meshes(mj_model: mujoco.MjModel) -> list[SiteMesh]:
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    site_meshes: list[SiteMesh] = []
    for site_id in range(int(mj_model.nsite)):
        try:
            mesh = site_to_trimesh(mj_model, site_id)
        except ValueError:
            continue

        body_id = int(mj_model.site_bodyid[site_id])
        xmat = np.asarray(mj_data.site_xmat[site_id], dtype=np.float32).reshape(3, 3)
        site_meshes.append(
            SiteMesh(
                site_id=site_id,
                site_name=_site_name(mj_model, site_id),
                body_id=body_id,
                body_name=_body_name(mj_model, body_id),
                mesh=mesh,
                pos=np.asarray(mj_data.site_xpos[site_id], dtype=np.float32),
                wxyz=vtf.SO3.from_matrix(xmat).wxyz.astype(np.float32),
                fixed=_is_fixed_body(mj_model, body_id),
            )
        )
    return site_meshes


def load_model(xml: str) -> mujoco.MjModel:
    xml_path = Path(xml).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"Model file not found: {xml_path}")
    return mujoco.MjModel.from_xml_path(str(xml_path))


class RealSenseCameraStream:
    def __init__(
        self,
        config: RealSenseCameraConfig,
        *,
        server: viser.ViserServer | None = None,
        mj_model: mujoco.MjModel | None = None,
        show_frustum: bool = True,
    ):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install it before enabling RealSense camera streaming."
            )

        self.config = config
        self.server = server
        self.mj_model = mj_model
        self.show_frustum = show_frustum
        self.latest_frame: RealSenseFrame | None = None
        self._pipeline = rs.pipeline()
        self._align = None
        self._depth_scale = 1.0
        self._pose_camera_id: int | None = None
        self._frustum_handle = None
        self._gui_folder = None
        self._gui_image_handle = None

        pipeline_config = rs.config()
        if self.config.serial_number:
            pipeline_config.enable_device(self.config.serial_number)
        pipeline_config.enable_stream(
            rs.stream.color,
            self.config.color_width,
            self.config.color_height,
            rs.format.bgr8,
            self.config.fps,
        )
        if self.config.enable_depth:
            pipeline_config.enable_stream(
                rs.stream.depth,
                self.config.depth_width,
                self.config.depth_height,
                rs.format.z16,
                self.config.fps,
            )

        self._profile = self._pipeline.start(pipeline_config)
        color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intrinsics = color_profile.get_intrinsics()
        self._color_width = int(color_intrinsics.width)
        self._color_height = int(color_intrinsics.height)
        self._color_aspect = float(self._color_width) / float(self._color_height)
        self._color_fov_y_rad = float(2.0 * np.arctan2(self._color_height, 2.0 * color_intrinsics.fy))
        if self.config.enable_depth:
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
            if self.config.align_depth_to_color:
                self._align = rs.align(rs.stream.color)

        if self.mj_model is not None:
            pose_camera_name = self.config.pose_camera_name or self.config.camera_name
            self._pose_camera_id = _camera_id_by_name(self.mj_model, pose_camera_name)
            if self._pose_camera_id is None:
                self._pose_camera_id = _camera_id_by_name(self.mj_model, "d435_head")
            if self._pose_camera_id is None and int(self.mj_model.ncam) == 1:
                self._pose_camera_id = 0

        if self.show_frustum and self.server is not None and self._pose_camera_id is not None:
            self._frustum_handle = self.server.scene.add_camera_frustum(
                name=f"/realsense/{self.config.camera_name}/frustum",
                fov=self._color_fov_y_rad,
                aspect=self._color_aspect,
                position=np.zeros(3, dtype=np.float32),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                scale=self.config.frustum_scale,
                color=(80, 180, 255),
                jpeg_quality=self.config.jpeg_quality,
            )
        elif self.show_frustum and self.server is not None and self.mj_model is not None:
            pose_camera_name = self.config.pose_camera_name or self.config.camera_name
            print(
                f"[WARN] Failed to locate MuJoCo camera '{pose_camera_name}' for RealSense frustum "
                f"{self.config.camera_name}."
            )

        if self.server is not None:
            self._gui_folder = self.server.gui.add_folder(
                f"Camera: {self.config.camera_name}",
                expand_by_default=True,
            )
            placeholder = np.zeros((self._color_height, self._color_width, 3), dtype=np.uint8)
            with self._gui_folder:
                self._gui_image_handle = self.server.gui.add_image(
                    placeholder,
                    label="RGB",
                    jpeg_quality=self.config.jpeg_quality,
                )

    def poll_frame(self) -> RealSenseFrame | None:
        frames = self._pipeline.poll_for_frames()
        if not frames:
            return self.latest_frame

        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if self.config.enable_depth else None
        if not color_frame and not depth_frame:
            return self.latest_frame

        color = None
        if color_frame:
            color = np.asarray(color_frame.get_data())[:, :, ::-1].copy()

        depth = None
        if depth_frame:
            depth = np.asarray(depth_frame.get_data(), dtype=np.float32) * self._depth_scale

        frame = RealSenseFrame(
            camera_name=self.config.camera_name,
            serial_number=self.config.serial_number,
            color=color,
            depth=depth,
            timestamp_sec=float(frames.get_timestamp()) / 1000.0,
        )
        self.latest_frame = frame
        return frame

    def update_visualization(self, mj_data: mujoco.MjData) -> None:
        if self._frustum_handle is not None and self._pose_camera_id is not None:
            position, wxyz = _camera_pose(mj_data, self._pose_camera_id)
            self._frustum_handle.visible = True
            self._frustum_handle.position = position
            self._frustum_handle.wxyz = wxyz

        if self._gui_image_handle is not None and self.latest_frame is not None and self.latest_frame.color is not None:
            self._gui_image_handle.image = self.latest_frame.color

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass
        if self._gui_folder is not None:
            self._gui_folder.remove()


class StandaloneMujocoScene:
    def __init__(
        self,
        server: viser.ViserServer,
        mj_model: mujoco.MjModel,
        *,
        include_collision: bool = False,
        show_sites: bool = True,
        add_ground: bool = True,
        show_camera_frustums: bool = True,
        real_sense_configs: Sequence[RealSenseCameraConfig] | None = None,
    ):
        self.server = server
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.include_collision = include_collision
        self.show_sites = show_sites
        self.add_ground = add_ground
        self.show_camera_frustums = show_camera_frustums
        self.real_sense_configs = list(real_sense_configs or [])

        self.fixed_bodies_frame = None
        self.body_meshes = extract_body_meshes(
            mj_model,
            include_collision=include_collision,
            skip_plane_geoms=add_ground,
        )
        self.site_meshes = extract_site_meshes(mj_model) if show_sites else []
        self.body_handles: dict[int, object] = {}
        self.site_handles: dict[int, object] = {}
        self.real_sense_cameras: list[RealSenseCameraStream] = []

    @classmethod
    def create(
        cls,
        server: viser.ViserServer,
        mj_model: mujoco.MjModel,
        *,
        include_collision: bool = False,
        show_sites: bool = True,
        add_ground: bool = True,
        show_camera_frustums: bool = True,
        real_sense_configs: Sequence[RealSenseCameraConfig] | None = None,
    ) -> "StandaloneMujocoScene":
        scene = cls(
            server,
            mj_model,
            include_collision=include_collision,
            show_sites=show_sites,
            add_ground=add_ground,
            show_camera_frustums=show_camera_frustums,
            real_sense_configs=real_sense_configs,
        )
        scene._setup()
        return scene

    def _setup(self) -> None:
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.server.scene.configure_environment_map(environment_intensity=0.8)
        self.fixed_bodies_frame = self.server.scene.add_frame("/fixed_bodies", show_axes=False)

        if self.add_ground:
            self._add_ground()
        self._add_fixed_meshes()
        self._create_mesh_handles()
        self._add_fixed_sites()
        self._create_site_handles()
        self._create_real_sense_cameras()
        self.update_from_mjdata(self.mj_data)

    def _add_ground(self) -> None:
        plane_found = False
        for geom_id in range(int(self.mj_model.ngeom)):
            if int(self.mj_model.geom_type[geom_id]) != mjtGeom.mjGEOM_PLANE:
                continue
            body_id = int(self.mj_model.geom_bodyid[geom_id])
            if not _is_fixed_body(self.mj_model, body_id):
                continue
            geom_name = mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            self.server.scene.add_grid(
                f"/fixed_bodies/{_body_name(self.mj_model, body_id)}/{geom_name}",
                infinite_grid=True,
                fade_distance=50.0,
                shadow_opacity=0.2,
                plane_opacity=0.4,
                position=np.asarray(self.mj_model.geom_pos[geom_id], dtype=np.float32),
                wxyz=np.asarray(self.mj_model.geom_quat[geom_id], dtype=np.float32),
            )
            plane_found = True

        if not plane_found:
            self.server.scene.add_grid(
                "/fixed_bodies/ground",
                infinite_grid=True,
                fade_distance=50.0,
                shadow_opacity=0.2,
                plane_opacity=0.4,
            )

    def _add_fixed_meshes(self) -> None:
        for body in self.body_meshes:
            if not body.fixed:
                continue
            self.server.scene.add_mesh_trimesh(
                f"/fixed_bodies/{body.body_name}",
                body.mesh,
                position=body.pos,
                wxyz=body.wxyz,
                cast_shadow=False,
                receive_shadow=0.2,
            )

    def _create_mesh_handles(self) -> None:
        for body in self.body_meshes:
            if body.fixed:
                continue
            handle = self.server.scene.add_mesh_trimesh(
                f"/bodies/{body.body_name}",
                body.mesh,
                position=body.pos,
                wxyz=body.wxyz,
                receive_shadow=0.2,
            )
            self.body_handles[body.body_id] = handle

    def _add_fixed_sites(self) -> None:
        for site in self.site_meshes:
            if not site.fixed:
                continue
            self.server.scene.add_mesh_trimesh(
                f"/fixed_bodies/{site.body_name}/sites/{site.site_name}",
                site.mesh,
                position=site.pos,
                wxyz=site.wxyz,
                cast_shadow=False,
                receive_shadow=0.0,
            )

    def _create_site_handles(self) -> None:
        for site in self.site_meshes:
            if site.fixed:
                continue
            handle = self.server.scene.add_mesh_trimesh(
                f"/sites/{site.body_name}/{site.site_name}",
                site.mesh,
                position=site.pos,
                wxyz=site.wxyz,
                cast_shadow=False,
                receive_shadow=0.0,
            )
            self.site_handles[site.site_id] = handle

    def _create_real_sense_cameras(self) -> None:
        for config in self.real_sense_configs:
            try:
                camera = RealSenseCameraStream(
                    config,
                    server=self.server,
                    mj_model=self.mj_model,
                    show_frustum=self.show_camera_frustums,
                )
            except Exception as exc:
                print(f"[WARN] Failed to create RealSense camera {config.camera_name}: {exc}")
                continue
            self.real_sense_cameras.append(camera)

    def update_from_mjdata(self, mj_data: mujoco.MjData) -> None:
        with self.server.atomic():
            if self.body_handles:
                body_xquat = vtf.SO3.from_matrix(
                    np.asarray(mj_data.xmat, dtype=np.float32).reshape(int(self.mj_model.nbody), 3, 3)
                ).wxyz.astype(np.float32)
                for body_id, handle in self.body_handles.items():
                    handle.position = np.asarray(mj_data.xpos[body_id], dtype=np.float32)
                    handle.wxyz = body_xquat[body_id]

            if self.site_handles:
                site_xquat = vtf.SO3.from_matrix(
                    np.asarray(mj_data.site_xmat, dtype=np.float32).reshape(int(self.mj_model.nsite), 3, 3)
                ).wxyz.astype(np.float32)
                for site_id, handle in self.site_handles.items():
                    handle.position = np.asarray(mj_data.site_xpos[site_id], dtype=np.float32)
                    handle.wxyz = site_xquat[site_id]

            for camera in self.real_sense_cameras:
                camera.poll_frame()
                camera.update_visualization(mj_data)

            self.server.flush()

    def get_latest_real_sense_frame(self, camera_name: str) -> RealSenseFrame | None:
        for camera in self.real_sense_cameras:
            if camera.config.camera_name == camera_name:
                return camera.latest_frame
        return None

    def close(self) -> None:
        for camera in self.real_sense_cameras:
            camera.close()


def show_body_meshes(
    mj_model: mujoco.MjModel,
    *,
    include_collision: bool = False,
    show_sites: bool = True,
    add_ground: bool = True,
    show_camera_frustums: bool = True,
    real_sense_configs: Sequence[RealSenseCameraConfig] | None = None,
) -> None:
    server = viser.ViserServer(label="standalone-viser")
    scene = StandaloneMujocoScene.create(
        server,
        mj_model,
        include_collision=include_collision,
        show_sites=show_sites,
        add_ground=add_ground,
        show_camera_frustums=show_camera_frustums,
        real_sense_configs=real_sense_configs,
    )

    print(
        f"Viewer ready with {len(scene.body_meshes)} body meshes, "
        f"{len(scene.site_meshes)} sites, "
        f"{len(scene.real_sense_cameras)} RealSense cameras. Press Ctrl+C to exit."
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        scene.close()
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal standalone viser mesh viewer for MuJoCo models.")
    parser.add_argument(
        "--xml",
        type=str,
        default=str(DEFAULT_XML),
        help=f"Path to a MuJoCo XML file. Default: {DEFAULT_XML}",
    )
    parser.add_argument("--collision", action="store_true", help="Include collision geoms.")
    parser.add_argument("--no-ground", action="store_true", help="Disable ground grid.")
    parser.add_argument("--no-sites", action="store_true", help="Disable site visualization.")
    parser.add_argument("--hide-camera-frustums", action="store_true", help="Hide camera frustums.")
    parser.add_argument("--realsense", action="store_true", help="Poll one connected RealSense camera.")
    parser.add_argument("--realsense-serial", type=str, default=None, help="Optional RealSense serial number.")
    parser.add_argument("--realsense-width", type=int, default=640, help="RealSense color/depth width.")
    parser.add_argument("--realsense-height", type=int, default=480, help="RealSense color/depth height.")
    parser.add_argument("--realsense-fps", type=int, default=30, help="RealSense stream FPS.")
    parser.add_argument("--realsense-no-depth", action="store_true", help="Disable RealSense depth streaming.")
    parser.add_argument("--summary", action="store_true", help="Only print a summary; do not launch viser.")
    args = parser.parse_args()

    mj_model = load_model(args.xml)
    body_meshes = extract_body_meshes(
        mj_model,
        include_collision=args.collision,
        skip_plane_geoms=not args.no_ground,
    )
    site_meshes = extract_site_meshes(mj_model) if not args.no_sites else []

    vertices = sum(int(len(body.mesh.vertices)) for body in body_meshes)
    faces = sum(int(len(body.mesh.faces)) for body in body_meshes)
    fixed = sum(int(body.fixed) for body in body_meshes)
    fixed_sites = sum(int(site.fixed) for site in site_meshes)
    print(
        f"bodies={len(body_meshes)} fixed={fixed} dynamic={len(body_meshes) - fixed} "
        f"sites={len(site_meshes)} fixed_sites={fixed_sites} dynamic_sites={len(site_meshes) - fixed_sites} "
        f"model_cameras={int(mj_model.ncam)} "
        f"vertices={vertices} faces={faces}"
    )

    if not args.summary:
        real_sense_configs = None
        if args.realsense:
            real_sense_configs = [
                RealSenseCameraConfig(
                    camera_name="realsense",
                    serial_number=args.realsense_serial,
                    color_width=args.realsense_width,
                    color_height=args.realsense_height,
                    depth_width=args.realsense_width,
                    depth_height=args.realsense_height,
                    fps=args.realsense_fps,
                    enable_depth=not args.realsense_no_depth,
                )
            ]
        show_body_meshes(
            mj_model,
            include_collision=args.collision,
            show_sites=not args.no_sites,
            add_ground=not args.no_ground,
            show_camera_frustums=not args.hide_camera_frustums,
            real_sense_configs=real_sense_configs,
        )


if __name__ == "__main__":
    main()
