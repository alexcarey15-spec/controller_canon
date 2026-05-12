import threading
import time
import math
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ourbeloved_interface.action import CartesianPTP
from ourbeloved.wx250s_kinematics import fk,ik
#from xarmclient import XArm



class CartesianPTPNode(Node):
    def __init__(self):
        super().__init__("cartesian_ptp_node")
        self.goal_handle = None
        self.goal_lock = threading.Lock()
        self.subscription = self.create_subscription(JointState, "joint_state", self.joint_state_callback, 10)
        self.curr_joints = [0.0] * 6
        #self.xarm = XArm()

        self.action_server = ActionServer(
            self,
            CartesianPTP,
            'set_cartesian_ptp',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

    # grabs joint angles from joint_state topic
    def joint_state_callback(self, msg):
        self.curr_joints = [math.degrees(a) for a in msg.position]

    def goal_callback(self, request):
        # check if ik can actually reach the goal position before accepting
        # get current htm to keep the same orientation
        htm, _ = fk(self.curr_joints)
        goal_htm = htm.copy()
        goal_htm[0, 3] = request.x
        goal_htm[1, 3] = request.y
        goal_htm[2, 3] = request.z

        result = ik(self.curr_joints, goal_htm)

        if not self.xarm.is_goal_valid(result): # 0 if valid
            return GoalResponse.ACCEPT
        else:
            self.get_logger().info("goal position unreachable")
            return GoalResponse.REJECT

    def handle_accepted_callback(self, goal_handle):
        # only one action at a time
        with self.goal_lock:
            if self.goal_handle is not None and self.goal_handle.is_active:
                self.goal_handle.abort()
            self.goal_handle = goal_handle
        goal_handle.execute()

    def cancel_callback(self, cancel_request):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info("moving to cartesian goal")
        feedback_msg = CartesianPTP.Feedback()
        result_msg = CartesianPTP.Result()

        # goal position in mm (fk/ik work in mm)
        goal_x = goal_handle.request.x
        goal_y = goal_handle.request.y
        goal_z = goal_handle.request.z

        # get current ee pose and keep the orientation
        current_htm, _ = fk(self.curr_joints)
        current_pos = current_htm[:3, 3].copy()
        orientation = current_htm[:3, :3].copy()

        goal_pos = np.array([goal_x, goal_y, goal_z])

        # break path into small segments so ik doesnt diverge
        total_dist = np.linalg.norm(goal_pos - current_pos)
        segment_length = 10.0  # 10mm per segment
        num_segments = max(int(total_dist / segment_length), 1)

        joints = list(self.curr_joints)

        for i in range(1, num_segments + 1):
            # check if cancelled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result_msg.success = False
                return result_msg

            # interpolate position linearly
            t = i / num_segments
            intermediate_pos = current_pos + t * (goal_pos - current_pos)

            # build intermediate htm keeping same orientation
            intermediate_htm = np.zeros((3, 4))
            intermediate_htm[:3, :3] = orientation
            intermediate_htm[:3, 3] = intermediate_pos
            
            """ THIS HAS BEEN CHANGED FROM PREVIOUS MILESTONE SUBMISSION
                    Changed to return the correct amount of values"""
            # solve ik from previous joints
            new_joints = ik(joints, intermediate_htm)
            if not self.xarm.is_goal_valid(new_joints):   # consistent with goal_callback
                self.get_logger().info("ik failed during trajectory")
                goal_handle.abort()
                result_msg.success = False
                return result_msg

            joints = list(new_joints)

            # send joints to robot
            self.xarm.set_joints(joints)

            # wait a bit for robot to move
            time.sleep(0.3)

            # publish feedback - euclidean distance to goal
            ee_htm, _ = fk(self.curr_joints)
            ee_pos = ee_htm[:3, 3]
            dist = np.linalg.norm(goal_pos - ee_pos)
            feedback_msg.distance = float(dist)
            goal_handle.publish_feedback(feedback_msg)

        # done
        self.get_logger().info("reached cartesian goal")
        goal_handle.succeed()
        result_msg.success = True
        return result_msg


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = CartesianPTPNode()
            executor = MultiThreadedExecutor()
            rclpy.spin(node, executor=executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

if __name__ == '__main__':
    main()