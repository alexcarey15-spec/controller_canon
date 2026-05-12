import rclpy
import math
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from ourbeloved.wx250s_kinematics import fk


class PoseNode(Node):
    # subscribes /joint_state (rad), runs fk, publishes /pose (m)
    def __init__(self):
        super().__init__("pose_node")
        self.subscription = self.create_subscription(
            JointState, "joint_state", self.joint_state_callback, 10)
        self.publisher = self.create_publisher(PoseStamped, "pose", 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.curr_joints = [0.0] * 6

    def joint_state_callback(self, msg):
        # topic is radians, fk wants degrees
        self.curr_joints = [math.degrees(a) for a in msg.position]

    def timer_callback(self):
        # fk returns 3x4 htm in mm
        htm, _ = fk(self.curr_joints)

        # position: last column, mm -> m
        x = htm[0, 3] / 1000.0
        y = htm[1, 3] / 1000.0
        z = htm[2, 3] / 1000.0

        # rotation: top-left 3x3 -> quaternion
        rot_matrix = htm[:3, :3]
        quat = Rotation.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.publisher.publish(msg)


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = PoseNode()
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
