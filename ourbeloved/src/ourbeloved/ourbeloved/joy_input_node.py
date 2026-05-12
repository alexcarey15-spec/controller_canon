import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, String, Empty, Float64MultiArray
from geometry_msgs.msg import Vector3

# PS4 axes
AXIS_LEFT_X  = 0
AXIS_LEFT_Y  = 1
AXIS_RIGHT_X = 3
AXIS_RIGHT_Y = 4
AXIS_L2      = 2
AXIS_R2      = 5
AXIS_DPAD_X  = 6
AXIS_DPAD_Y  = 7

# PS4 buttons
BTN_SHARE   = 8    # -> cartesian mode
BTN_OPTIONS = 9    # -> joint mode
BTN_HOME    = 10   # PS button -> homing

# tuning - full stick deflection equals these magnitudes
JOINT_SPEED = 20.0   # rad/s (caps in joint_state_node, so this just needs to be >= cap)
CART_SPEED  = 0.250  # m/s (12.5 mm per 20Hz tick at full stick)


class JoyInputNode(Node):
    # turns raw /joy into the semantic topics cannon_node uses
    def __init__(self):
        super().__init__("joy_input_node")
        self.prev_buttons = []
        self.fire_active = False
        self.mode = "joint"

        self.sub = self.create_subscription(Joy, "/joy", self.joy_callback, 10)
        self.fire_pub = self.create_publisher(Bool, "/fire", 10)
        self.mode_pub = self.create_publisher(String, "/mode", 10)
        self.home_pub = self.create_publisher(Empty, "/home_trigger", 10)
        self.joint_vel_pub = self.create_publisher(Float64MultiArray, "/cmd_joint_vel", 10)
        self.cart_vel_pub = self.create_publisher(Vector3, "/cmd_cart_vel", 10)

    # rising-edge detection on a button index
    def rising(self, idx, buttons):
        if idx >= len(buttons):
            return False
        prev = self.prev_buttons[idx] if idx < len(self.prev_buttons) else 0
        return buttons[idx] == 1 and prev == 0

    def joy_callback(self, msg):
        # mode + home buttons (rising edge only)
        if self.rising(BTN_SHARE, msg.buttons):
            self.mode = "cartesian"
            self.get_logger().info("mode -> cartesian")
        if self.rising(BTN_OPTIONS, msg.buttons):
            self.mode = "joint"
            self.get_logger().info("mode -> joint")
        if self.rising(BTN_HOME, msg.buttons):
            self.get_logger().info("home pressed")
            self.home_pub.publish(Empty())

        self.prev_buttons = list(msg.buttons)

        # publish current mode every tick so late subscribers catch up
        mode_msg = String()
        mode_msg.data = self.mode
        self.mode_pub.publish(mode_msg)

        # R2 trigger: resting +1.0, fully pressed -1.0
        r2 = msg.axes[AXIS_R2] if len(msg.axes) > AXIS_R2 else 1.0
        fired = r2 < 0.0
        if fired != self.fire_active:
            self.fire_active = fired
            b = Bool()
            b.data = fired
            self.fire_pub.publish(b)

        # helper for safe axis read
        def ax(i):
            return msg.axes[i] if len(msg.axes) > i else 0.0

        # joint velocities in rad/s (same axis-to-joint map as before)
        vels = Float64MultiArray()
        vels.data = [
            ax(AXIS_LEFT_Y)  * JOINT_SPEED,   # joint 1
            ax(AXIS_LEFT_X)  * JOINT_SPEED,   # joint 2
            ax(AXIS_DPAD_X)  * JOINT_SPEED,   # joint 3
            ax(AXIS_DPAD_Y)  * JOINT_SPEED,   # joint 4
            ax(AXIS_RIGHT_Y) * JOINT_SPEED,   # joint 5
            ax(AXIS_RIGHT_X) * JOINT_SPEED,   # joint 6
        ]
        self.joint_vel_pub.publish(vels)

        # cartesian velocity in m/s
        v = Vector3()
        v.x =  ax(AXIS_RIGHT_Y) * CART_SPEED
        v.y = -ax(AXIS_RIGHT_X) * CART_SPEED
        v.z =  ax(AXIS_LEFT_Y)  * CART_SPEED
        self.cart_vel_pub.publish(v)


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = JoyInputNode()
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
