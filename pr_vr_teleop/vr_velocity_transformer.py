#!/usr/bin/env python3
"""
vr_velocity_transformer.py - Transform VR velocity through TF kinematic chain.

Transforms VR controller velocity:
    VR Frame -> base_link (static) -> right_ee_tip (dynamic TF lookup)

Author: Perceptyne
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros


class VRVelocityTransformer:
    """
    Transforms VR pose deltas into end-effector frame velocities.
    
    Pipeline:
        1. VR delta -> base_link frame (static transform)
        2. base_link -> ee_frame (dynamic TF lookup, inverted)
        3. Output: Twist [vx, vy, vz, wx, wy, wz] in ee_frame
    """

    # Static transform: VR -> base_link
    # base_link +X = VR -X, base_link +Y = VR +Z, base_link +Z = VR +Y
    M_VR_TO_BASE = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  1.0,  0.0],
    ], dtype=np.float64)

    def __init__(
        self,
        tf_buffer: tf2_ros.Buffer,
        base_frame: str = "base_link",
        ee_frame: str = "right_ee_tip",
        smoothing_factor: float = 0.7,
    ) -> None:
        """
        Args:
            tf_buffer: ROS2 TF buffer (must have listener attached externally).
            base_frame: Robot base frame name.
            ee_frame: End-effector frame name.
            smoothing_factor: Exponential smoothing [0=no smoothing, 1=infinite smoothing].
        """
        self.tf_buffer = tf_buffer
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.smoothing_factor = smoothing_factor

        # Rotation object for static VR->base transform
        self._R_vr_to_base = Rotation.from_matrix(self.M_VR_TO_BASE)

        # Smoothed state (in base_link frame, before ee transform)
        self._smoothed_pos_base = np.zeros(3, dtype=np.float64)
        self._smoothed_rot_base = Rotation.identity()
        self._prev_smoothed_pos_base = np.zeros(3, dtype=np.float64)
        self._prev_smoothed_rot_base = Rotation.identity()

        # Reference VR pose (captured on first update)
        self._initial_vr_pos: Optional[np.ndarray] = None
        self._initial_vr_rot: Optional[Rotation] = None

    def reset(self) -> None:
        """Reset initial reference and smoothing state."""
        self._initial_vr_pos = None
        self._initial_vr_rot = None
        self._smoothed_pos_base = np.zeros(3, dtype=np.float64)
        self._smoothed_rot_base = Rotation.identity()
        self._prev_smoothed_pos_base = np.zeros(3, dtype=np.float64)
        self._prev_smoothed_rot_base = Rotation.identity()

    def update(
        self,
        vr_position: np.ndarray,
        vr_orientation_xyzw: np.ndarray,
        dt: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Process VR pose and return velocity in end-effector frame.

        Args:
            vr_position: VR position [x, y, z].
            vr_orientation_xyzw: VR quaternion [x, y, z, w].
            dt: Time step (seconds).

        Returns:
            (linear_vel, angular_vel) in ee_frame, or None if TF unavailable.
        """
        vr_pos = np.asarray(vr_position, dtype=np.float64)
        vr_rot = Rotation.from_quat(vr_orientation_xyzw)

        # Capture initial reference on first call
        if self._initial_vr_pos is None:
            self._initial_vr_pos = vr_pos.copy()
            self._initial_vr_rot = vr_rot
            return np.zeros(3), np.zeros(3)

        # ---- Step 1: Compute relative delta in VR frame ----
        delta_pos_vr = vr_pos - self._initial_vr_pos
        delta_rot_vr = vr_rot * self._initial_vr_rot.inv()

        # ---- Step 2: Transform delta to base_link frame ----
        delta_pos_base = self.M_VR_TO_BASE @ delta_pos_vr
        delta_rot_base = self._R_vr_to_base * delta_rot_vr * self._R_vr_to_base.inv()

        # ---- Step 3: Exponential smoothing in base frame ----
        alpha = 1.0 - self.smoothing_factor
        self._smoothed_pos_base = (
            self.smoothing_factor * self._smoothed_pos_base + alpha * delta_pos_base
        )
        self._smoothed_rot_base = self._slerp(
            self._smoothed_rot_base, delta_rot_base, alpha
        )

        # ---- Step 4: Compute velocity (derivative of smoothed delta) ----
        vel_pos_base = (self._smoothed_pos_base - self._prev_smoothed_pos_base) / dt
        vel_rot_base = (
            self._smoothed_rot_base * self._prev_smoothed_rot_base.inv()
        ).as_rotvec() / dt

        # Update previous state
        self._prev_smoothed_pos_base = self._smoothed_pos_base.copy()
        self._prev_smoothed_rot_base = self._smoothed_rot_base

        # ---- Step 5: Lookup TF and transform to ee_frame ----
        R_base_to_ee = self._get_rotation_base_to_ee()
        if R_base_to_ee is None:
            return None  # TF not available

        # Transform velocity from base_link to ee_frame
        # v_ee = R_base_to_ee^T * v_base  (inverse rotation to re-express vector)
        R_ee_to_base = R_base_to_ee.inv()
        linear_vel_ee = R_ee_to_base.apply(vel_pos_base)
        angular_vel_ee = R_ee_to_base.apply(vel_rot_base)

        return linear_vel_ee, angular_vel_ee

    def _get_rotation_base_to_ee(self) -> Optional[Rotation]:
        """Lookup rotation from base_link to ee_frame from TF tree."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.ee_frame,  # target frame
                self.base_frame,  # source frame
                Time(),  # latest available
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            q = transform.transform.rotation
            return Rotation.from_quat([q.x, q.y, q.z, q.w])
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None

    @staticmethod
    def _slerp(r1: Rotation, r2: Rotation, t: float) -> Rotation:
        """Spherical linear interpolation between two rotations."""
        key_rots = Rotation.from_quat([r1.as_quat(), r2.as_quat()])
        slerp = Slerp([0.0, 1.0], key_rots)
        return slerp(t)


# ------------------------------------------------------------------------- #
# Backward-compatible wrapper (drop-in for VRPoseClient + VRProcessor)
# ------------------------------------------------------------------------- #


class VRTwistProvider:
    """
    Combines VR pose polling with velocity transformation.
    
    Drop-in replacement exposing same interface as old VRPoseClient,
    plus get_ee_twist() for the new transformed velocity.
    """

    def __init__(
        self,
        node: Node,
        fastapi_url: str = "http://192.168.3.53:48420/pose",
        base_frame: str = "base_link",
        ee_frame: str = "right_ee_tip",
        smoothing_factor: float = 0.7,
        poll_period: float = 0.01,
    ) -> None:
        # Import here to avoid circular dependency
        from vr_pose_client import VRPoseClient

        self._node = node
        self._poll_period = poll_period

        # TF2 setup
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        # VR client (existing implementation)
        self._vr_client = VRPoseClient(
            fastapi_url=fastapi_url,
            smoothing_factor=smoothing_factor,
            poll_period=poll_period,
        )

        # Velocity transformer
        self._transformer = VRVelocityTransformer(
            tf_buffer=self._tf_buffer,
            base_frame=base_frame,
            ee_frame=ee_frame,
            smoothing_factor=smoothing_factor,
        )

        # State for velocity computation
        self._last_update_time: Optional[float] = None

    def get_ee_twist(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get velocity twist in end-effector frame.

        Returns:
            (linear_vel [3], angular_vel [3]) or None if not ready.
        """
        # Get raw VR pose (already smoothed by VRPoseClient)
        pos, quat_wxyz = self._vr_client.get_robot_frame_pose()
        if pos is None:
            return None

        # Compute dt
        now = self._node.get_clock().now().nanoseconds / 1e9
        if self._last_update_time is None:
            self._last_update_time = now
            return np.zeros(3), np.zeros(3)

        dt = now - self._last_update_time
        self._last_update_time = now

        if dt <= 0:
            dt = self._poll_period  # Fallback

        # Convert wxyz -> xyzw for scipy
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

        return self._transformer.update(pos, quat_xyzw, dt)

    # ---- Backward compatibility: expose VRPoseClient methods ----

    def get_robot_frame_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Legacy interface: returns (position, quat_wxyz) in base frame."""
        return self._vr_client.get_robot_frame_pose()

    def get_trigger(self) -> bool:
        """Return trigger state from VR controller."""
        return self._vr_client.get_trigger()

    def stop(self) -> None:
        """Stop VR polling thread."""
        self._vr_client.stop()

    def reset(self) -> None:
        """Reset transformer state (e.g., after re-homing)."""
        self._transformer.reset()
        self._last_update_time = None
