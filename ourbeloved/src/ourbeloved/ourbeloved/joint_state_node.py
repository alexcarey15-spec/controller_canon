import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
from xarmclient import XArm

class JointStateNode(Node):
    # create joint state node and its service 
    def __init__(self):
        super().__init__("joint_state_node")
        self.publisher = self.create_publisher(JointState, "joint_state", 10)
        timer_period = 0.2  # 5Hz 
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.angles = [0.0] * 6## array of the joints
        self.xarm = XArm()
        
    # runs every 0.2 seconds -> names the joints, and sets pos to list of angles, uses publisher to publish message 
    def timer_callback(self):
        msg = JointState()
        self.angles = [math.radians(v) for v in self.xarm.get_joints()]
        msg.name = ['J1_Waist', 'J2_Shoulder', 'J3_Elbow', 'J4_ForearmRoll', 'J5_WristAngle', 'J6_WristRotate']
        msg.position = self.angles 
        self.publisher.publish(msg)
        

def main(args=None):
    try:
        with rclpy.init(args=args):
            node = JointStateNode()
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

if __name__ == '__main__':
    main()