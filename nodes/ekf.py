#!/usr/bin/env python

import roshelper
import numpy as np
import math
import rospy
import tf
import time
from bebop_msgs.msg import Ardrone3PilotingStateAltitudeChanged
from sensor_msgs.msg import Range
from std_msgs.msg import Empty
from std_msgs.msg import Float64
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from foresight.msg import PoseArrayWithTimes
from foresight.msg import ForesightState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Float64MultiArray
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler

NODE_NAME = "ekf"
n = roshelper.Node(NODE_NAME, anonymous=False)

EKF_TOPIC = "ekf_pose"
EKF_COV_TOPIC = "ekf_pose_cov"
VEL_TOPIC = "bebop/odom_cov"

num_states = 6 # x, y, z, vx, vy, vz

@n.entry_point()
class EKF(object):

    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "map")

        self.tag_range_topics = rospy.get_param("~tag_range_topics")
        self.subs = list()
        self.ranges = dict()
        self.vel_data = []
        self.tag_pos = dict()
        self.tag_order = []

        self.listener = tf.TransformListener()
        self.last_time = None
        self.yaw_zero = None

        self.altitude = None

        self.F = np.eye(num_states)
        self.P = np.diag([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])
        #0.1*np.random.rand(num_states, num_states)
        #self.P = np.eye(num_states)
        self.x = np.zeros((num_states,))

        self.uwb_state = np.zeros((len(self.tag_range_topics),num_states))
        self.H = np.zeros((num_states, len(self.tag_range_topics)))
        w_x = 0.003
        w_v = 0.006
        self.Q = np.diag([w_x, w_x, w_x, w_v, w_v, w_v])
        uwb_cov = 1.0
        vel_cov = 0.15
        alt_cov = 0.1
        self.R = np.diag([uwb_cov, uwb_cov, uwb_cov, uwb_cov, uwb_cov, uwb_cov,
                             vel_cov, vel_cov, vel_cov, alt_cov])

        for topic in self.tag_range_topics:
            self.subs.append(rospy.Subscriber(topic, Range, self.range_cb))

    def range_cb(self, rng):
        self.ranges[rng.header.frame_id] = rng.range
        if len(self.tag_order) < len(self.tag_range_topics):
            self.tag_order.append(rng.header.frame_id)

            try:
                (trans, _) = self.listener.lookupTransform(
                    self.frame_id, rng.header.frame_id, rospy.Time(0))
                self.tag_pos[rng.header.frame_id] = np.array(trans[:3])
                self.uwb_state[len(self.tag_order)-1, :] = \
                    np.array([trans[0], trans[1], trans[2], 0, 0, 0])
            except tf.Exception:
                return

    @n.subscriber(VEL_TOPIC, Odometry)
    def odom_sub(self, odom):
        ori = odom.pose.pose.orientation
        ori_quat = [ori.x, ori.y, ori.z, ori.w]
        r, p, yaw = euler_from_quaternion(ori_quat)

        if self.yaw_zero is None:
            self.yaw_zero = yaw

        self.yaw = yaw - self.yaw_zero

        twist = odom.twist.twist.linear
        self.vel_data = []
        self.vel_data.append(twist.x)
        self.vel_data.append(twist.y)
        self.vel_data.append(twist.z)

    @n.subscriber("/bebop/states/ardrone3/PilotingState/AltitudeChanged",
                  Ardrone3PilotingStateAltitudeChanged)
    def altitude_sub(self, alt):
        cov = [0] * 36
        cov[17] = 0.05
        self.altitude = alt.altitude

    @n.publisher(EKF_TOPIC, Odometry)
    def ekf_pub(self, ranges, vel_data, yaw, alt):
        z = np.array([])
        new_pose = Odometry()
        ps_cov = PoseWithCovarianceStamped()
        for tag_name in self.tag_order:
            measurement = ranges[tag_name]
            z = np.append(z, measurement)

        if self.last_time is None:
            self.last_time = rospy.Time.now().to_sec()
        else:
            dt = rospy.Time.now().to_sec() - self.last_time
            self.predict(dt)
            self.update(z, vel_data, yaw, alt)
            self.last_time = rospy.Time.now().to_sec()

            new_pose.header.stamp = rospy.get_rostime()
            new_pose.header.frame_id = self.frame_id
            new_pose.pose.pose.position.x = self.x[0]
            new_pose.pose.pose.position.y = self.x[1]
            new_pose.pose.pose.position.z = self.x[2]
            cov = self.P.flatten().tolist()
            new_pose.pose.covariance = cov
            new_pose.twist.twist.linear.x = self.x[3]
            new_pose.twist.twist.linear.y = self.x[4]
            new_pose.twist.twist.linear.z = self.x[5]

        return new_pose

    # @n.publisher(EKF_COV_TOPIC, PoseWithCovarianceStamped)
    # def cov_pub(self, ps_cov):
    #     return ps_cov

    def predict(self, dt):
        self.F = self.transition_matrix(dt)
        self.x = np.dot(self.F,self.x)
        self.P = np.dot(self.F,np.dot(self.P, self.F.T)) + self.Q


    def h_uwb(self, x, uwb_z):
        uwb_z = np.power(uwb_z, 2)
        h = []

        for uwb in range(0, self.uwb_state.shape[0]):
            # only look at difference between x, y, z
            # not velocities
            diff = x[0:3] - self.uwb_state[uwb,0:3]
            h.append(np.inner(diff, diff))

        return h

    def update(self, uwb_z, vel_z, yaw, alt_z):
        x = self.x
        F = self.F
        H = self.H
        P = self.P
        R = self.R

        # I expect uwb_z to be a numpy array
        # square the elements of uwb_z to get
        # distance squared
        uwb_z = np.power(uwb_z, 2)
        z = np.append(uwb_z,vel_z)
        z = np.append(z,alt_z)

        h_uwb = self.h_uwb(x, uwb_z)
        h_vel = x[3:6]
        h_alt = x[2]

        h = np.append(h_uwb, h_vel)
        h = np.append(h, h_alt)

        y = z - h

        H = 2*(x-self.uwb_state)
        for vel_idx in range(0,len(vel_z)):
            add_vel = [0, 0, 0, 0, 0, 0]
            add_vel[vel_idx + 3] = 1
            H = np.vstack([H, add_vel])
        # this row is for altitude
        H = np.vstack([H, [0, 0, 1, 0, 0, 0]])

        S = np.dot(H,np.dot(P,H.T)) + R
        K = np.dot(P,np.dot(H.T, np.linalg.inv(S)))

        x = x + np.dot(K,y)
        P = np.dot(np.eye(num_states) - np.dot(K,H),P)
        self.x = x
        self.P = P

    def transition_matrix(self, dt):
        return np.array([[1, 0, 0, dt, 0, 0],
                            [0, 1, 0, 0, dt, 0],
                            [0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

    @n.main_loop(frequency=50)
    def run(self):
        if len(self.ranges.values()) == 6 and len(self.vel_data) == 3 and self.altitude is not None:
            self.ekf_pub(self.ranges, self.vel_data, self.yaw, self.altitude)
            self.ranges = dict()
            self.vel_data = []
            self.altitude = None


if __name__ == "__main__":
    n.start(spin=True)
