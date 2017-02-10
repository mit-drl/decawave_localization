#!/usr/bin/env python

from __future__ import division
import tf
import numpy as np
import rospy
import scipy.optimize as opt
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Range

NODE_NAME = "decawave_localization"
POSE_TOPIC = "uwb_pose"
POSE_COV_TOPIC = "pose_cov"
ERROR_PC_TOPIC = "error_cloud"


class DecaWaveLocalization:

    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "map")
        cov_x = rospy.get_param("~cov_x", 0.6)
        cov_y = rospy.get_param("~cov_y", 0.6)
        self.cov = self.cov_matrix(cov_x, cov_y)
        self.ps_pub = rospy.Publisher(POSE_TOPIC, PoseStamped, queue_size=1)
        self.ps_cov_pub = rospy.Publisher(
            POSE_COV_TOPIC, PoseWithCovarianceStamped, queue_size=1)
        self.rate = rospy.Rate(rospy.get_param("frequency", 30))
        self.ps = PoseStamped()
        self.ps.header.frame_id = self.frame_id
        self.ps_cov = PoseWithCovarianceStamped()
        self.ps_cov.header.frame_id = self.frame_id
        self.ps_cov.pose.covariance = self.cov
        self.last = None
        self.listener = tf.TransformListener()
        self.tag_range_topics = rospy.get_param("~tag_range_topics")
        self.subs = list()
        self.ranges = dict()
        self.tag_pos = dict()
        for topic in self.tag_range_topics:
            self.subs.append(rospy.Subscriber(topic, Range, self.range_cb))

    def find_position(self):
        if self.last is None:
            self.last = self.tag_pos.values()[0]
        res = opt.minimize(self.error, self.last, jac=self.jac, method="SLSQP")
        self.last = res.x
        return res.x

    def error(self, x):
        err = 0.0
        for tag_id in self.ranges.keys():
            tag = self.tag_pos[tag_id]
            dist = self.ranges[tag_id]
            err += pow(pow(dist, 2) - pow(np.linalg.norm(x - tag), 2), 2)
        return err

    def jac(self, x):
        jac_x = 0.0
        jac_y = 0.0
        for tag_id in self.ranges.keys():
            tag = self.tag_pos[tag_id]
            dist = self.ranges[tag_id]
            err = pow(dist, 2) - pow(np.linalg.norm(x - tag), 2)
            jac_x += err * (tag[0] - x[0])
            jac_y += err * (tag[1] - x[1])
        return np.array([jac_x, jac_y])

    def range_cb(self, rng):
        self.ranges[rng.header.frame_id] = rng.range
        try:
            (trans, _) = self.listener.lookupTransform(
                self.frame_id, rng.header.frame_id, rospy.Time(0))
            self.tag_pos[rng.header.frame_id] = np.array(trans[:2])
        except:
            return

        if len(self.tag_pos.values()) == len(self.ranges.values()) \
                and len(self.tag_pos.keys()) >= 3:
            pos = self.find_position()
            x, y = pos[0], pos[1]
            self.ps.pose.position.x = x
            self.ps.pose.position.y = y
            self.ps.header.stamp = rospy.get_rostime()
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


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    dd = DecaWaveLocalization()
    rospy.spin()
