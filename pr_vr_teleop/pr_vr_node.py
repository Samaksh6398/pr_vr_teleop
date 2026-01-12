import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from vr_pose_client import VRPoseClient  # assuming this gives you VR pose in robot frame
import time
from rclpy.action import ActionClient
try:
    from pr_msgs.action import MoveGripper
except Exception:
    # pr_msgs may not be available in this environment (linting/static analysis).
    # Defer runtime errors and handle missing action gracefully.
    MoveGripper = None


class VRProcessor:
    def __init__(self, params):
        self.params = params
        self.filtered_vr_position = np.zeros(3)
        self.filtered_vr_orientation = Rotation.identity()
        self.prev_filtered_vr_position = np.zeros(3)
        self.prev_filtered_vr_orientation = Rotation.identity()

    def update(self, cmd):
        vr_pos = np.array([cmd[0], cmd[1], cmd[2]], dtype=float)
        vr_quat = Rotation.from_quat([cmd[3], cmd[4], cmd[5], cmd[6]], scalar_first=False)

        alpha = 1.0 - self.params.vr_smoothing
        self.filtered_vr_position = (
            self.params.vr_smoothing * self.filtered_vr_position + alpha * vr_pos
        )

        key_times = [0, 1]
        key_rots = Rotation.from_quat([
            self.filtered_vr_orientation.as_quat(),
            vr_quat.as_quat(),
        ])
        slerp = Slerp(key_times, key_rots)
        self.filtered_vr_orientation = slerp(alpha)

        net_vr_pos_delta = self.filtered_vr_position - self.prev_filtered_vr_position
        net_vr_quat_delta = self.filtered_vr_orientation * self.prev_filtered_vr_orientation.inv()

        dt = self.params.dt
        linear_vel = net_vr_pos_delta / dt
        angular_vel = net_vr_quat_delta.as_rotvec() / dt

        self.prev_filtered_vr_position = self.filtered_vr_position.copy()
        self.prev_filtered_vr_orientation = self.filtered_vr_orientation

        return linear_vel, angular_vel


class VRTwistPublisher(Node):
    def __init__(self):
        super().__init__("vr_twist_publisher")

        self.publisher_ = self.create_publisher(TwistStamped, "/right/command_ee_vel", 10)
        timer_period = 0.025  # 100 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # VR processing config
        self.params = lambda: None
        self.params.vr_smoothing = 0.7
        self.params.dt = timer_period
        self.vr_processor = VRProcessor(self.params)

        self.vr_client = VRPoseClient(
            fastapi_url="http://192.168.3.53:48420/pose",
            smoothing_factor=0.7,
        )

        self.home_pose = [-90.0, 90.0, 0.0, -110.0, 0, -90.0, 0.0]

        # Gripper action client (only if action type is importable)
        if MoveGripper is not None:
            self._gripper_action_client = ActionClient(self, MoveGripper, '/right/move_gripper')
        else:
            self.get_logger().warn('pr_msgs.action.MoveGripper not importable; gripper actions disabled')
            self._gripper_action_client = None

        # Keep track of last trigger to detect edges and avoid spamming goals
        self._last_trigger = False

    def timer_callback(self):
        vr_position, vr_quat_wxyz = self.vr_client.get_robot_frame_pose()
        if vr_position is None:
            return

        # Handle trigger -> gripper action (edge-detected)
        try:
            trigger_state = self.vr_client.get_trigger()
        except Exception:
            trigger_state = False

        if trigger_state and not self._last_trigger:
            # Trigger pressed -> open gripper
            self.get_logger().info('Trigger ON: sending gripper OPEN goal')
            self.send_gripper_goal(position_opening=0.04)
        elif (not trigger_state) and self._last_trigger:
            # Trigger released -> close gripper
            self.get_logger().info('Trigger OFF: sending gripper CLOSE goal')
            self.send_gripper_goal(position_opening=0.0)

        self._last_trigger = bool(trigger_state)

        cmd = [
            vr_position[0],
            vr_position[1],
            vr_position[2],
            vr_quat_wxyz[1],  # x
            vr_quat_wxyz[2],  # y
            vr_quat_wxyz[3],  # z
            vr_quat_wxyz[0],  # w
        ]

        linear_vel, angular_vel = self.vr_processor.update(cmd)

        msg = TwistStamped()
        msg.header.frame_id = "right_ee_tip"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = linear_vel
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = angular_vel
    
        self.publisher_.publish(msg)

    def publish_zero_twist(self):
        """Publish a zero TwistStamped (best-effort) to stop controllers on shutdown."""
        try:
            zero_msg = TwistStamped()
            zero_msg.header.frame_id = "right_ee_tip"
            zero_msg.header.stamp = self.get_clock().now().to_msg()
            zero_msg.twist.linear.x = 0.0
            zero_msg.twist.linear.y = 0.0
            zero_msg.twist.linear.z = 0.0
            zero_msg.twist.angular.x = 0.0
            zero_msg.twist.angular.y = 0.0
            zero_msg.twist.angular.z = 0.0
            self.publisher_.publish(zero_msg)
        except Exception:
            # Best-effort; ignore errors during shutdown
            pass

    # ---------------- Gripper action helpers ----------------
    def send_gripper_goal(self, position_opening: float, velocity: float = 0.0, torque: float = 150.0):
        """Send a non-blocking goal to the gripper action server.

        This is best-effort: if the action server isn't available the goal is skipped.
        """
        # Ensure action client exists
        if self._gripper_action_client is None or MoveGripper is None:
            self.get_logger().warn('Gripper action client not available; skipping goal')
            return

        # Ensure action server available (short wait)
        try:
            available = self._gripper_action_client.wait_for_server(timeout_sec=1.0)
        except Exception:
            available = False

        if not available:
            self.get_logger().warn('Gripper action server not available; skipping goal')
            return

        goal_msg = MoveGripper.Goal()
        # Fields according to action: velocity, torque, position_opening
        goal_msg.velocity = float(velocity)
        goal_msg.torque = float(torque)
        goal_msg.position_opening = float(position_opening)

        send_goal_future = self._gripper_action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'Exception while sending gripper goal: {e}')
            return

        if not goal_handle.accepted:
            self.get_logger().info('Gripper goal rejected')
            return

        self.get_logger().info('Gripper goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        try:
            result = future.result().result
            status = future.result().status
        except Exception as e:
            self.get_logger().error(f'Exception getting gripper result: {e}')
            return

        self.get_logger().info(f'Gripper action finished with status={status}, result={result}')

   
def main(args=None):
    rclpy.init(args=args)
    node = VRTwistPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_zero_twist()
        # Give middleware a short time to deliver the message
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.05)
    finally:
        node.vr_client.stop()

    node.vr_client.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
