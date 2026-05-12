#!/usr/bin/env python3
"""
cannon_node.py
==============
Runs on the CANNON computer.

Subscriptions
-------------
/joy                      sensor_msgs/Joy
    Raw PlayStation 4 controller state published by joy_node on the
    controller computer.

/joint_goals              std_msgs/Float64MultiArray
    Precise joint position goals (radians) sent externally, e.g. from a
    tower-defence game-logic node.  Array layout:
        [joint1, joint2, joint3, joint4, joint5, joint6]  (wx250s has 6 DOF)

joint_state               sensor_msgs/JointState
    Published by xarmserver_sim; used to read current joint positions for
    tolerance checking and cartesian control seeding.

Publications
------------
/fire                     std_msgs/Bool
    True  while R2 is held down.
    False when R2 is released.

Action Clients
--------------
set_cartesian_ptp         ourbeloved/action/CartesianPTP
    Delegates cartesian mode moves to cartesian_ptp_node.

Robot control
-------------
Two control modes are toggled via controller buttons (see BUTTON MAP):
    • Joint mode      sticks + d-pad control joints 1-6
    • Cartesian mode  sticks controls end-effector x/y/z

NOTE: All button/axis indices below are for PS4 DualShock
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float64MultiArray
import threading
import time
import math
import numpy as np
from rclpy.action import ActionClient,ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from sensor_msgs.msg import JointState
from ourbeloved_interface.action import CartesianPTP
from ourbeloved.wx250s_kinematics import fk,ik
from xarmclient import XArm



# --─ PS4 button / axis indices ----------------
# Axes
AXIS_LEFT_X   = 0   # left stick horizontal  (joint mode:  joint 2 / cart: –)
AXIS_LEFT_Y   = 1   # left stick vertical    (joint mode:  joint 1 / cart: x)
AXIS_RIGHT_X  = 3   # right stick horizontal (cart: y)
AXIS_RIGHT_Y  = 4   # right stick vertical   (cart: z)
AXIS_L2       = 2   # L2 analogue            (joint mode: joint 5)
AXIS_R2       = 5   # R2 analogue            (joint mode: joint 6 / fire)
AXIS_DPAD_X   = 6   # D-pad horizontal       (joint mode: joint 3)
AXIS_DPAD_Y   = 7   # D-pad vertical         (joint mode: joint 4)

# Buttons
BTN_L1        = 4   
BTN_R1        = 5  
BTN_SHARE     = 8   # Change to Cartesian mode
BTN_OPTIONS   = 9   # Change to joint mode
BTN_HOME      = 10  # Homing



# ======= Tuning parameters =======
JOINT_SPEED   = 20   # rad/s scaling for manual joint motion
CART_STEP_MM   = 1  # mm per full stick deflection tick in cartesian mode
GOAL_TOLERANCE = 0.1  # in degrees NOT radians
#GOAL_TOLERANCE = np.deg2rad(0.1)   # 0.1 degree tolerance for precise goals
GOAL_HOLD_SECS = 2.0               # seconds to hold within tolerance
SET_JOINTS_IN_RADIANS = False

CONTROL_RATE  = 20.0  # Hz – how often the control loop runs


class CannonNode(Node):
    """Drives the wx250s for the ourbeloved tower-defence package."""

    def __init__(self):
        super().__init__('cannon_node')

        # xarmclient connection
        self.xarm = XArm()

        #  State 
        self.mode            = 'joint'   # 'joint' | 'cartesian'
        self.latest_joy      = Joy()
        self.joint_goals     = None      # Float64MultiArray data or None
        self.fire_active     = False
        self._goal_hold_time = 0.0       # seconds spent within tolerance
        self.curr_joints       = [0.0] * 6  # degrees, updated from joint_state
        self._cart_goal_active = False       # True while CartesianPTP action is in flight

        #  ROS Subscribers 
        self.joy_sub = self.create_subscription(
            Joy, '/joy', self._joy_cb, 10)

        self.joint_goal_sub = self.create_subscription(
            Float64MultiArray, '/joint_goals', self._goal_cb, 10)

        # Read joint_state so we can do FK for cartesian seeding and
        # tolerance checking for precise goals
        self.js_sub = self.create_subscription(
            JointState, 'joint_state', self._js_cb, 10) #CHANGE IN LAB
            #JointState, '/vm_sim/joint_state', self._js_cb, 10)
        
        #  Publishers
        self.fire_pub = self.create_publisher(Bool, '/fire', 10)

        '''# -- Action client for cartesian moves ----------------------------─
        self._cart_client = ActionClient(
            self,
            CartesianPTP,
            'set_cartesian_ptp',
            callback_group=ReentrantCallbackGroup(),
        )'''

        # -- Control loop --------------------------------------------------─
        self.timer = self.create_timer(
            1.0 / CONTROL_RATE,
            self._control_loop,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(
            'cannon_node ready  |  mode: JOINT  |  '
            'Share->Cartesian  Opitions->Joint  PS->Home'
        )

    def _set_joints(self, joints_deg: list):
        """Send joint angles to xarm, converting if set_joints expects radians."""
        if SET_JOINTS_IN_RADIANS:
            self.xarm.set_joints([math.radians(v) for v in joints_deg])
        else:
            self.xarm.set_joints(joints_deg, "high_acc", [30]*6) # tuple velocities for each joint

    
    # --------------------------------------------------------------
    # Subscriber Callbacks
    # --------------------------------------------------------------
    def _js_cb(self, msg: JointState):
        """Keep curr_joints up to date (degrees) from xarmserver_sim."""
        self.curr_joints = [math.degrees(a) for a in msg.position]

    def _goal_cb(self, msg: Float64MultiArray):
        """Accept a precise joint goal from an external node (degrees).
        
        /joint_goals publishes in radians (standard ROS convention).
        Converted to degrees here for internal use, since fk()/ik() and
        xarm.set_joints() operate in degrees (matching cartesian_ptp_node).
        """
        if len(msg.data) == 6:
            self.joint_goals = [math.degrees(r) for r in msg.data]
            self._goal_hold_time = 0.0
            self.get_logger().info(f'Received joint goal: {list(msg.data)}')
        else:
            self.get_logger().warn(
                f'/joint_goals: expected 6 values, got {len(msg.data)} - ignoring')

    def _joy_cb(self, msg: Joy):
        prev = self.latest_joy

        # Just prints to inform working
        if not prev.axes:  # first message ever received
            self.get_logger().info(
                f'First /joy message received - '
                f'{len(msg.axes)} axes, {len(msg.buttons)} buttons')
            # 
            # CHECK IF NEEDED:
            #
            self.get_logger().info(
                f'curr_joints at first joy: {[f"{v:.4f}" for v in self.curr_joints]} '
                f'(should be ~0.0 at home regardless of unit)')
            
        self.latest_joy = msg

        # Rising-edge detection: button was 0 last tick, is 1 now
        def rising(idx):
            if idx >= len(msg.buttons):
                return False
            prev_val = prev.buttons[idx] if idx < len(prev.buttons) else 0
            return msg.buttons[idx] == 1 and prev_val == 0

        # Mode switching
        if rising(BTN_SHARE):
            self.mode = 'cartesian'
            self._cart_goal_active = False
            self.get_logger().info('Mode → CARTESIAN')

        if rising(BTN_OPTIONS):
            self.mode = 'joint'
            self.get_logger().info('Mode → JOINT')

        if rising(BTN_HOME):
            self.get_logger().info('Homing…')
            self.xarm.home()

        # -- Fire (R2 analogue: resting value +1.0, fully pressed –1.0) --
        r2_axis = msg.axes[AXIS_R2] if len(msg.axes) > AXIS_R2 else 1.0
        fired = r2_axis < 0.0   # pressed past the midpoint
        if fired != self.fire_active:
            self.fire_active = fired
            out = Bool()
            out.data = fired
            self.fire_pub.publish(out)
            self.get_logger().info(f'fire → {fired}')

    # --------------------------------------------------------------
    # Control loop
    # --------------------------------------------------------------

    def _control_loop(self):
        """Called at CONTROL_RATE Hz to move the arm."""

        # -- Precise goal takes priority over manual input ----------
        if self.joint_goals is not None:
            self._execute_joint_goal()
            return

        # -- Manual control ----------------------------------------─
        joy = self.latest_joy
        if not joy.axes:
            return  # no message received yet

        dt = 1.0 / CONTROL_RATE

        if self.mode == 'joint':
            self._manual_joint_control(joy, dt)
        else:
            self._manual_cartesian_control(joy)
            #self._manual_joint_control(joy, dt)

    # --------------------------------------------------------------
    # Precise joint goal execution
    # --------------------------------------------------------------

    def _execute_joint_goal(self):
        """
        Command the arm toward self.joint_goals (degrees).
        Clears once every joint has been within GOAL_TOLERANCE for GOAL_HOLD_SECS.
        Satisfies the assessed criterion: < 0.1 degree absolute error held >= 2 s.
        """
        goals = self.joint_goals
        self.xarm.set_joints(goals)

        errors = np.abs(np.array(self.curr_joints) - np.array(goals))
        dt = 1.0 / CONTROL_RATE

        if np.all(errors < GOAL_TOLERANCE):
            self._goal_hold_time += dt
            if self._goal_hold_time >= GOAL_HOLD_SECS:
                self.get_logger().info('Joint goal reached and held - clearing.')
                self.joint_goals     = None
                self._goal_hold_time = 0.0
        else:
            self._goal_hold_time = 0.0

    # --------------------------------------------------------------
    # Manual joint control (joint mode)
    # --------------------------------------------------------------

    def _manual_joint_control(self, joy: Joy, dt: float):
        """
        Axis → joint mapping (tune indices / signs to your preference):
            Left stick Y  → joint 1  (waist)
            Left stick X  → joint 2  (shoulder)
            D-pad X       → joint 3  (elbow)
            D-pad Y       → joint 4  (forearm roll)
            Right stick Y → joint 5  (wrist angle)
            Right stick X → joint 6  (wrist rotate)
        """
        def ax(i):
            return joy.axes[i] if len(joy.axes) > i else 0.0

        deltas = [
            ax(AXIS_LEFT_Y)  * JOINT_SPEED * dt,   # joint 1
            ax(AXIS_LEFT_X)  * JOINT_SPEED * dt,   # joint 2
            ax(AXIS_DPAD_X)  * JOINT_SPEED * dt,   # joint 3
            ax(AXIS_DPAD_Y)  * JOINT_SPEED * dt,   # joint 4
            ax(AXIS_RIGHT_Y)  * JOINT_SPEED * dt,  # joint 5
            ax(AXIS_RIGHT_X)  * JOINT_SPEED * dt,  # joint 6
        ]

        if all(abs(d) < 1e-4 for d in deltas):
            return  # no input (sticks centred) – don't spam commands

        target = [c + d for c, d in zip(self.curr_joints, deltas)]
        self.get_logger().info(
            f'[joint] target: {[f"{v:.2f}" for v in target]}',
            throttle_duration_sec=0.5)
        self._set_joints(target) # Custom made set joints that checks in fucntion for degrees or rads
        #self.xarm.set_joints(target) # Traditional set joints

    # --------------------------------------------------------------
    # Manual cartesian control (via CartesianPTP action server)
    # --------------------------------------------------------------

    def _manual_cartesian_control(self, joy: Joy):
        def ax(i):
            return joy.axes[i] if len(joy.axes) > i else 0.0

        dx =  ax(AXIS_RIGHT_Y) * CART_STEP_MM
        dy = -ax(AXIS_RIGHT_X) * CART_STEP_MM
        dz =  ax(AXIS_LEFT_Y)  * CART_STEP_MM

        self.get_logger().info

        # Deadzone: sticks near centre - mark action as available again
        if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(dz) < 0.5:
            # Commented out
            #self._cart_goal_active = False 
            return

        # Get current end effector pose from FK
        current_htm, _ = fk(self.curr_joints)

        # Build goal HTM, same orientation
        goal_htm = current_htm.copy()
        goal_htm[0, 3] += dx
        goal_htm[1, 3] += dy
        goal_htm[2, 3] += dz

        # Solve IK from current joints as seed
        # ik() returns a tuple - pass whole to is_goal_valid, extract index 0 for joints
        
        new_joints = ik(self.curr_joints, goal_htm)

        if new_joints is None:
            self.get_logger().warn('IK did not converge')
            return

        if self.xarm.is_goal_valid(new_joints) is None:
            self.get_logger().warn('Goal out of range')
            return

        self.get_logger().info(
            f'[cart] ee delta: dx={dx:.1f} dy={dy:.1f} dz={dz:.1f} mm',
            throttle_duration_sec=0.5)
        self._set_joints(list(new_joints))

        



def main(args=None):
    rclpy.init(args=args)
    node = CannonNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
