"""
cannon_node.py
the brain - reads /joint_state and semantic command topics, decides what
joint target to send, publishes /joint_command (radians) to joint_state_node.
no xarm here; only joint_state_node owns the xarm connection.
"""

import rclpy
import math
import numpy as np
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Vector3
from ourbeloved.wx250s_kinematics import fk, ik


CONTROL_RATE    = 20.0   # Hz
GOAL_TOLERANCE  = 0.1    # degrees - assignment requires <0.1 deg
GOAL_HOLD_SECS  = 2.0    # seconds within tolerance to count as 'achieved'
CART_DEADZONE_MM = 0.5   # ignore tiny cart deltas


class CannonNode(Node):
    def __init__(self):
        super().__init__("cannon_node")

        # internal state - all joints kept in degrees to match fk/ik units
        self.mode = "joint"
        self.curr_joints = [0.0] * 6
        self.joint_goals = None
        self._goal_hold_time = 0.0
        self.cmd_joint_vel = [0.0] * 6      # rad/s
        self.cmd_cart_vel = (0.0, 0.0, 0.0)  # m/s

        # subs
        self.create_subscription(JointState, "joint_state",
                                 self.joint_state_callback, 10)
        self.create_subscription(Float64MultiArray, "/joint_goals",
                                 self.goal_callback, 10)
        self.create_subscription(String, "/mode", self.mode_callback, 10)
        self.create_subscription(Float64MultiArray, "/cmd_joint_vel",
                                 self.joint_vel_callback, 10)
        self.create_subscription(Vector3, "/cmd_cart_vel",
                                 self.cart_vel_callback, 10)

        # pub
        self.cmd_pub = self.create_publisher(Float64MultiArray,
                                             "/joint_command", 10)

        self.timer = self.create_timer(1.0 / CONTROL_RATE, self.control_loop)
        self.get_logger().info("cannon_node ready (mode: joint)")

    # callbacks ------------------------------------------------------------
    def joint_state_callback(self, msg):
        # topic is radians; store in degrees
        self.curr_joints = [math.degrees(a) for a in msg.position]

    def goal_callback(self, msg):
        # /joint_goals is radians; store in degrees
        if len(msg.data) == 6:
            self.joint_goals = [math.degrees(r) for r in msg.data]
            self._goal_hold_time = 0.0
            self.get_logger().info(f"received goal (rad): {list(msg.data)}")
        else:
            self.get_logger().warn(
                f"/joint_goals expected 6 vals, got {len(msg.data)}")

    def mode_callback(self, msg):
        if msg.data in ("joint", "cartesian") and msg.data != self.mode:
            self.mode = msg.data
            self.get_logger().info(f"mode -> {self.mode}")

    def joint_vel_callback(self, msg):
        if len(msg.data) == 6:
            self.cmd_joint_vel = list(msg.data)

    def cart_vel_callback(self, msg):
        self.cmd_cart_vel = (msg.x, msg.y, msg.z)

    # helpers --------------------------------------------------------------
    def publish_joint_command(self, joints_deg):
        # topic convention is radians
        out = Float64MultiArray()
        out.data = [math.radians(v) for v in joints_deg]
        self.cmd_pub.publish(out)

    # control loop ---------------------------------------------------------
    def control_loop(self):
        # precise goal always wins over manual input
        if self.joint_goals is not None:
            self.do_joint_goal()
            return

        if self.mode == "joint":
            self.do_manual_joint()
        else:
            self.do_manual_cartesian()

    def do_joint_goal(self):
        goals = self.joint_goals
        self.publish_joint_command(goals)

        errors = np.abs(np.array(self.curr_joints) - np.array(goals))
        dt = 1.0 / CONTROL_RATE
        if np.all(errors < GOAL_TOLERANCE):
            self._goal_hold_time += dt
            if self._goal_hold_time >= GOAL_HOLD_SECS:
                self.get_logger().info("goal reached and held - clearing")
                self.joint_goals = None
                self._goal_hold_time = 0.0
        else:
            self._goal_hold_time = 0.0

    def do_manual_joint(self):
        dt = 1.0 / CONTROL_RATE
        # cmd is rad/s, internal joints are degrees
        deltas_deg = [math.degrees(v * dt) for v in self.cmd_joint_vel]
        if all(abs(d) < 1e-4 for d in deltas_deg):
            return
        target = [c + d for c, d in zip(self.curr_joints, deltas_deg)]
        self.publish_joint_command(target)

    def do_manual_cartesian(self):
        dt = 1.0 / CONTROL_RATE
        vx, vy, vz = self.cmd_cart_vel
        # m/s -> mm per tick (fk/ik units)
        dx = vx * dt * 1000.0
        dy = vy * dt * 1000.0
        dz = vz * dt * 1000.0
        if abs(dx) < CART_DEADZONE_MM and abs(dy) < CART_DEADZONE_MM and abs(dz) < CART_DEADZONE_MM:
            return

        current_htm, _ = fk(self.curr_joints)
        goal_htm = current_htm.copy()
        goal_htm[0, 3] += dx
        goal_htm[1, 3] += dy
        goal_htm[2, 3] += dz

        new_joints = ik(self.curr_joints, goal_htm)
        if new_joints is None:
            self.get_logger().warn("IK didnt converge",
                                   throttle_duration_sec=1.0)
            return

        self.publish_joint_command(list(new_joints))


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = CannonNode()
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
