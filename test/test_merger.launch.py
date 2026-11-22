"""Launch test: two synthetic LaserScans -> one merged scan in base_link.

Uses the parameters of the CMEXAIII robot (cmeresearch_bringup
cmexaiii_hardware.launch.py), with the shadow and average filters enabled.
"""

import math
import threading
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

RANGE_MAX = 12.0
INF_EPSILON = 1.0

# Laser mounting in base_link: front-left, and rear-right rotated by 180 deg.
FRONT = (0.2, 0.15, 0.0)
REAR = (-0.2, -0.15, math.pi)


@pytest.mark.launch_test
def generate_test_description():
    merger = launch_ros.actions.Node(
        package='dual_laser_merger',
        executable='dual_laser_merger_node',
        name='dual_laser_merger',
        parameters=[{
            'laser_1_topic': '/test/scan_front_left',
            'laser_2_topic': '/test/scan_rear_right',
            'merged_scan_topic': '/test/scan_combined',
            'target_frame': 'base_link',
            'tolerance': 0.01,
            'queue_size': 5,
            'angle_increment': 0.008,
            'scan_time': 0.067,
            'range_min': 0.01,
            'range_max': RANGE_MAX,
            'min_height': -1.0,
            'max_height': 1.0,
            'angle_min': -3.141592654,
            'angle_max': 3.141592654,
            'inf_epsilon': INF_EPSILON,
            'use_inf': True,
            'allowed_radius': 0.45,
            'enable_shadow_filter': True,
            'enable_average_filter': True,
        }],
    )
    return (launch.LaunchDescription([merger, launch_testing.actions.ReadyToTest()]),
            {'merger': merger})


def make_scan(frame, stamp, obstacles):
    """Full-circle scan; obstacles = [(center_angle, half_width, range)]."""
    scan = LaserScan()
    scan.header.frame_id = frame
    scan.header.stamp = stamp
    scan.angle_min = -math.pi
    scan.angle_increment = 0.01
    n = int(round(2 * math.pi / scan.angle_increment))
    scan.angle_max = scan.angle_min + (n - 1) * scan.angle_increment
    scan.range_min = 0.05
    scan.range_max = RANGE_MAX
    scan.scan_time = 0.1
    ranges = [float('inf')] * n
    for center, half, rng in obstacles:
        for i in range(n):
            a = scan.angle_min + i * scan.angle_increment
            if abs(math.remainder(a - center, 2 * math.pi)) <= half:
                ranges[i] = rng
    scan.ranges = ranges
    return scan


def to_base(mount, laser_angle, rng):
    x0, y0, yaw = mount
    x = x0 + rng * math.cos(yaw + laser_angle)
    y = y0 + rng * math.sin(yaw + laser_angle)
    return math.atan2(y, x), math.hypot(x, y)


def beam(scan, angle):
    i = int(round((angle - scan.angle_min) / scan.angle_increment))
    return scan.ranges[max(0, min(i, len(scan.ranges) - 1))]


class TestMerger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('merger_test')
        cls.lock = threading.Lock()
        cls.merged = []
        cls.node.create_subscription(
            LaserScan, '/test/scan_combined', cls._on_merged, rclpy.qos.qos_profile_sensor_data)
        cls.pub_1 = cls.node.create_publisher(LaserScan, '/test/scan_front_left', 10)
        cls.pub_2 = cls.node.create_publisher(LaserScan, '/test/scan_rear_right', 10)
        cls.tf = StaticTransformBroadcaster(cls.node)
        cls.tf.sendTransform([cls._mount('laser_front', FRONT), cls._mount('laser_rear', REAR)])
        cls.executor = rclpy.executors.SingleThreadedExecutor()
        cls.executor.add_node(cls.node)
        cls.thread = threading.Thread(target=cls.executor.spin, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown()
        cls.node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def _on_merged(cls, msg):
        with cls.lock:
            cls.merged.append(msg)

    @classmethod
    def _mount(cls, frame, mount):
        x, y, yaw = mount
        t = TransformStamped()
        t.header.stamp = cls.node.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = math.sin(yaw / 2)
        t.transform.rotation.w = math.cos(yaw / 2)
        return t

    def merge(self, front_obstacles, rear_obstacles, timeout=15.0):
        """Publish synchronized scans until a merged scan arrives; return it."""
        with self.lock:
            self.merged.clear()
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            stamp = self.node.get_clock().now().to_msg()
            self.pub_1.publish(make_scan('laser_front', stamp, front_obstacles))
            self.pub_2.publish(make_scan('laser_rear', stamp, rear_obstacles))
            time.sleep(0.1)
            with self.lock:
                if len(self.merged) >= 3:  # skip anything from before this setup
                    return self.merged[-1]
        self.fail('no merged scan received')

    def test_obstacles_from_both_lasers_appear_in_base_link(self):
        scan = self.merge([(0.0, 0.1, 1.0)], [(0.0, 0.1, 1.0)])
        self.assertEqual(scan.header.frame_id, 'base_link')
        self.assertAlmostEqual(scan.range_max, RANGE_MAX)

        for mount in (FRONT, REAR):
            angle, rng = to_base(mount, 0.0, 1.0)
            got = beam(scan, angle)
            self.assertAlmostEqual(
                got, rng, delta=0.05,
                msg=f'obstacle from laser at {mount} expected at {angle:.3f} rad / {rng:.3f} m')

    def test_empty_directions_stay_free(self):
        scan = self.merge([(0.0, 0.1, 1.0)], [(0.0, 0.1, 1.0)])
        for angle in (math.pi / 2, -math.pi / 2):
            got = beam(scan, angle)
            self.assertTrue(math.isinf(got) or got > RANGE_MAX, f'{angle:.2f} rad -> {got}')

    def test_one_laser_seeing_nothing_still_publishes_the_other(self):
        """Open space behind the robot must not hide obstacles in front (regression)."""
        scan = self.merge([(0.0, 0.1, 1.0)], [])
        angle, rng = to_base(FRONT, 0.0, 1.0)
        self.assertAlmostEqual(beam(scan, angle), rng, delta=0.05)

    def test_both_lasers_seeing_nothing_publish_a_free_scan(self):
        """Costmaps need clearing scans in open space too (regression)."""
        scan = self.merge([], [])
        finite = [r for r in scan.ranges if not math.isinf(r) and r <= RANGE_MAX]
        self.assertEqual(finite, [])

    def test_isolated_single_beam_returns_are_dropped(self):
        # One-beam spike at 3 m (speckle noise) next to a real wall segment.
        scan = self.merge([(1.0, 0.004, 3.0), (-1.0, 0.1, 2.0)], [(0.0, 0.1, 1.0)])
        spike_angle, _ = to_base(FRONT, 1.0, 3.0)
        got = beam(scan, spike_angle)
        self.assertTrue(math.isinf(got) or got > RANGE_MAX, f'speckle kept: {got}')
        wall_angle, wall_rng = to_base(FRONT, -1.0, 2.0)
        self.assertAlmostEqual(beam(scan, wall_angle), wall_rng, delta=0.05)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_merger_exits_cleanly(self, proc_info, merger):
        launch_testing.asserts.assertExitCodes(proc_info, [0, -2, -15], process=merger)
