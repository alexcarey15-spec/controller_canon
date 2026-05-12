import math
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray
from xarmclient import XArm


class JointStateNode(Node):
    # owns the xarm connection so other nodes dont need their own
    # publishes /joint_state (rad), executes /joint_command (rad) and /home_trigger
    def __init__(self):
        super().__init__("joint_state_node")
        self.xarm = XArm()

        # publish at 5Hz
        self.publisher = self.create_publisher(JointState, "joint_state", 10)
        self.timer = self.create_timer(0.2, self.timer_callback)
        self.angles = [0.0] * 6

        # commands from the brain
        self.cmd_sub = self.create_subscription(
            Float64MultiArray, "/joint_command", self.cmd_callback, 10)
        self.home_sub = self.create_subscription(
            Empty, "/home_trigger", self.home_callback, 10)

    def timer_callback(self):
        msg = JointState()
        # xarm.get_joints returns degrees; topic is radians
        self.angles = [math.radians(v) for v in self.xarm.get_joints()]
        msg.name = ['J1_Waist', 'J2_Shoulder', 'J3_Elbow',
                    'J4_ForearmRoll', 'J5_WristAngle', 'J6_WristRotate']
        msg.position = self.angles
        self.publisher.publish(msg)

    def cmd_callback(self, msg):
        # /joint_command is radians, xarm wants degrees
        if len(msg.data) != 6:
            self.get_logger().warn(
                f"/joint_command expected 6 vals, got {len(msg.data)}")
            return
        joints_deg = [math.degrees(v) for v in msg.data]
        self.xarm.set_joints(joints_deg, "high_acc", [30]*6)

    def home_callback(self, msg):
        self.get_logger().info("homing")
        self.xarm.home()


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = JointStateNode()
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
