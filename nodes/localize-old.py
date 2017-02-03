#!/usr/bin/env python

import tf
import numpy as np
import pykalman
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Range


NODE_NAME = "decawave_localization"
POSE_TOPIC = "pose"
POSE_COV_TOPIC = "pose_cov"
ERROR_PC_TOPIC = "error_cloud"


class DecaWaveLocalization:

    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "map")
        trans_mat = np.array(rospy.get_param("~transition_matrix"))
        obs_mat = np.array(rospy.get_param("~observation_matrix"))
        self.ps_pub = rospy.Publisher(POSE_TOPIC, PoseStamped, queue_size=1)
        self.ps_cov_pub = rospy.Publisher(
            POSE_COV_TOPIC, PoseWithCovarianceStamped, queue_size=1)
        self.rate = rospy.Rate(rospy.get_param("frequency", 30))
        self.ps = PoseStamped()
        self.ps.header.frame_id = self.frame_id
        self.ps_cov = PoseWithCovarianceStamped()
        self.ps_cov.header.frame_id = self.frame_id
        self.last = None
        self.kf = pykalman.KalmanFilter(
            transition_matrices=trans_mat.reshape(2, 2),
            observation_matrices=obs_mat.reshape(2, 2))
        self.fsm = np.array(rospy.get_param("~initial_state"))
        self.fsc = np.array(rospy.get_param("~initial_cov")).reshape(2, 2)
        self.cov_sensor = rospy.get_param("~cov_sensor", 0.01)
        self.listener = tf.TransformListener()
        self.tag_range_topics = rospy.get_param("~tag_range_topics")
        self.subs = list()
        self.ranges = dict()
        self.tag_pos = dict()
        for topic in self.tag_range_topics:
            self.subs.append(rospy.Subscriber(topic, Range, self.range_cb))

    def range_cb(self, rng):
        self.ranges[rng.header.frame_id] = rng.range
        print self.ranges
        try:
            (trans, _) = self.listener.lookupTransform(
                self.frame_id, rng.header.frame_id, rospy.Time(0))
            self.tag_pos[rng.header.frame_id] = trans[:2]
        except:
            return

        if len(self.tag_pos.keys()) >= 3:
            dists = self.ranges.values()
            x, y = self.trilaterate(dists)
            self.fsm, self.fsc = self.kf.filter_update(
                self.fsm, self.fsc, np.array([x, y]))
            self.ps.pose.position.x = self.fsm[0]
            self.ps.pose.position.y = self.fsm[1]
            self.ps.header.stamp = rospy.get_rostime()
            self.ps_cov.pose.covariance = self.cov_matrix(0.5,
                                                          0.5)
            self.ps_cov.header.stamp = rospy.get_rostime()
            self.ps_cov.pose.pose = self.ps.pose
            self.ps_pub.publish(self.ps)
            self.ps_cov_pub.publish(self.ps_cov)

    def cov_matrix(self, x_cov, y_cov):
        return [x_cov, 0, 0, 0, 0, 0,
                0, y_cov, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0]

    def trilaterate(self, rs):
        # xs, ys, rs = [], [], []
        xs, ys = [], []
        for key in self.tag_pos.keys():
            xs.append(self.tag_pos[key][0])
            ys.append(self.tag_pos[key][1])
            # rs.append(self.ranges[key])

        S = (pow(xs[2], 2.) - pow(xs[1], 2.) + pow(ys[2], 2.)
             - pow(ys[1], 2.) + pow(rs[1], 2.) - pow(rs[2], 2.)) / 2.0
        T = (pow(xs[0], 2.) - pow(xs[1], 2.) + pow(ys[0], 2.)
             - pow(ys[1], 2.) + pow(rs[1], 2.) - pow(rs[0], 2.)) / 2.0
        y = ((T * (xs[1] - xs[2])) - (S * (xs[1] - xs[0]))) \
            / (((ys[0] - ys[1]) * (xs[1] - xs[2]))
               - ((ys[2] - ys[1]) * (xs[1] - xs[0])))
        x = ((y * (ys[0] - ys[1])) - T) / (xs[1] - xs[0])
        return x, y


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    dd = DecaWaveLocalization()
    rospy.spin()
