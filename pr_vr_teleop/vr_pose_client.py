#!/usr/bin/env python3
# vr_pose_client.py - Poll FastAPI for VR poses and map into robot frame.

import time
import threading
import requests
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class VRPoseClient:
    """
    Polls a FastAPI endpoint that returns:
        { "pose": [x, y, z, qx, qy, qz, qw], "trigger": 0.0 or 1.0 }

    and exposes a smoothed pose in the robot TCP frame.

    Use:
        vr = VRPoseClient("http://IP:PORT/pose")
        ...
        pos, wxyz = vr.get_robot_frame_pose()
    """

    def __init__(
        self,
        fastapi_url: str = "http://192.168.3.53:48420/pose",
        smoothing_factor: float = 0.7,
        poll_period: float = 0.01,  # 100 Hz
    ) -> None:
        self.fastapi_url = fastapi_url
        self.smoothing_factor = smoothing_factor
        self.poll_period = poll_period

        # VR state
        self.current_vr_pose = None
        self.initial_vr_pose = None
        self.vr_data_received = False
        self.trigger = False  # trigger state from VR

        # Map VR frame -> robot TCP frame (same matrix you used)
        self.M_vr_to_tcp = np.array(
            [
                [0.0,  1.0, 0.0],   # x_robot =x_vr
                [-1.0,  0.0,  0.0],   # y_robot =  z_vr
                [0.0, 0.0,  1.0],   # z_robot = -y_vr
            ],
            dtype=float,
        )

        
        self.Rmap = Rotation.from_matrix(self.M_vr_to_tcp)

        # Smoothed state in robot frame
        self.smoothed_position = np.zeros(3, dtype=float)
        self.smoothed_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # xyzw

        # Threading
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    # Internal polling loop
    # ------------------------------------------------------------------ #
    def _poll_loop(self) -> None:
        """Background thread to poll FastAPI for VR poses."""
        while not self._stop:
            try:
                resp = requests.get(self.fastapi_url, timeout=0.5)
                if resp.status_code == 200:
                    data = resp.json()
                    pose = data.get("pose")
                    trigger = data.get("trigger")
                    gripper = data.get("gripper")

                    if pose is not None and len(pose) >= 7:
                        pos = np.array(pose[:3], dtype=float)
                        quat_xyzw = np.array(pose[3:7], dtype=float)

                        with self._lock:
                            self.current_vr_pose = {
                                "position": pos,
                                "orientation": quat_xyzw,
                            }
                            self.trigger = (float(trigger) == 1.0) if trigger is not None else False

                            self.gripper = 1.0 if gripper >= 0.5 else 0.0
                            
                            if not self.vr_data_received:
                                # Capture initial pose as reference
                                self.initial_vr_pose = {
                                    "position": pos.copy(),
                                    "orientation": quat_xyzw.copy(),
                                }
                                self.vr_data_received = True
            except Exception:
                # Silent failure; you can log if you want
                pass

            time.sleep(self.poll_period)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_robot_frame_pose(self):
        """
        Returns:
            position (np.ndarray, shape (3,)) in robot TCP frame
            orientation_wxyz (np.ndarray, shape (4,)) in robot TCP frame (w, x, y, z)

        or (None, None) if no data yet.
        """
        with self._lock:
            if not self.vr_data_received or self.current_vr_pose is None:
                return None, None

            # --- Position: VR delta -> robot frame, smoothed ---
            vr_pos_delta = (
                self.current_vr_pose["position"] - self.initial_vr_pose["position"]
            )

            # Map delta into robot TCP frame
            transformed_delta = self.M_vr_to_tcp @ vr_pos_delta

            # Exponential smoothing
            alpha = 1.0 - self.smoothing_factor
            self.smoothed_position = (
                self.smoothing_factor * self.smoothed_position
                + alpha * transformed_delta
            )

            # --- Orientation: relative VR rot -> robot frame, smoothed ---
            initial_rot_vr = Rotation.from_quat(self.initial_vr_pose["orientation"])
            current_rot_vr = Rotation.from_quat(self.current_vr_pose["orientation"])
            relative_rot_vr = current_rot_vr * initial_rot_vr.inv()

            # Map relative rotation into robot TCP frame:
            # R_robot = M * R_vr * M^T
            relative_rot_robot = self.Rmap * relative_rot_vr * self.Rmap.inv()

            # Smooth orientation with SLERP between previous smoothed and new target
            slerp_t = alpha
            current_smoothed_rot = Rotation.from_quat(self.smoothed_orientation)
            key_rots = Rotation.from_quat(
                [
                    current_smoothed_rot.as_quat(),
                    relative_rot_robot.as_quat(),
                ]
            )
            slerp = Slerp([0.0, 1.0], key_rots)
            smoothed_rot = slerp(slerp_t)
            self.smoothed_orientation = smoothed_rot.as_quat()
            self.smoothed_orientation /= np.linalg.norm(self.smoothed_orientation)

            # Convert xyzw -> wxyz for Pyroki
            x, y, z, w = self.smoothed_orientation
            quat_wxyz = np.array([w, x, y, z], dtype=float)

            return self.smoothed_position.copy(), quat_wxyz.copy()

    def get_trigger(self) -> bool:
        """Return latest trigger state (e.g. for gripper control)."""
        with self._lock:
            return bool(self.trigger)

    def stop(self):
        """Stop background polling thread."""
        self._stop = True
        self._thread.join(timeout=1.0)
