#!/usr/bin/env python

import roshelper
import numpy as np
import math
import rospy
import tf
import time
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

NODE_NAME = "ekf"
n = roshelper.Node(NODE_NAME, anonymous=False)

EKF_TOPIC = "ekf_pose"
EKF_COV_TOPIC = "ekf_pose_cov"

num_states = 6 # x, y, z, vx, vy, vz

@n.entry_point()
class EKF(object):

    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "map")

        self.tag_range_topics = rospy.get_param("~tag_range_topics")
        self.subs = list()
        self.ranges = dict()
        self.tag_pos = dict()
        self.tag_order = []

        self.listener = tf.TransformListener()
        self.last_time = None

        self.F = np.eye(num_states)
        self.P = np.eye(num_states)
        self.x = np.zeros((num_states,))

        self.uwb_state = np.zeros((len(self.tag_range_topics),num_states))
        self.H = np.zeros((num_states, len(self.tag_range_topics)))
        self.Q = np.diag([0.1, 0.1, 0.1, 0, 0, 0])
        self.R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

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

        if len(self.ranges.values()) == 6:
            self.ekf_pub(self.ranges)
            self.ranges = dict()


    @n.publisher(EKF_TOPIC, PoseStamped)
    def ekf_pub(self, ranges):
        z = np.array([])
        new_pose = PoseStamped()
        ps_cov = PoseWithCovarianceStamped()
        for tag_name in self.tag_order:
            measurement = ranges[tag_name]
            z = np.append(z, measurement)

        if self.last_time is None:
            self.last_time = rospy.Time.now().to_sec()
        else:
            dt = rospy.Time.now().to_sec() - self.last_time
            self.predict(dt)
            self.update(z)
            self.last_time = rospy.Time.now().to_sec()

            new_pose.header.stamp = rospy.get_rostime()
            new_pose.header.frame_id = self.frame_id
            new_pose.pose.position.x = self.x[0]
            new_pose.pose.position.y = self.x[1]
            new_pose.pose.position.z = self.x[2]

            cov = self.P.flatten().tolist()

            ps_cov.header = new_pose.header
            ps_cov.pose.pose = new_pose.pose
            self.cov_pub(ps_cov)

        return new_pose

    @n.publisher(EKF_COV_TOPIC, PoseWithCovarianceStamped)
    def cov_pub(self, ps_cov):
        return ps_cov

    def predict(self, dt):
        self.F = self.transition_matrix(dt)
        self.x = np.dot(self.F,self.x)
        self.P = np.dot(self.F,np.dot(self.P, self.F.T)) + self.Q

    def update(self, z):
        x = self.x
        F = self.F
        H = self.H
        P = self.P
        R = self.R

        # I expect z to be a numpy array
        z = np.power(z, 2)
        h = np.zeros((num_states,))

        for row in range(0, x.shape[0]):
            diff = x - self.uwb_state[row,:]
            h[row] = np.inner(diff, diff)

        y = z - h
        H = 2*(x-self.uwb_state)
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

if __name__ == "__main__":
    n.start(spin=True)
