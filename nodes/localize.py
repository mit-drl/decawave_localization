#!/usr/bin/env python

from __future__ import division
import tf
import numpy as np
import math
import rospy
import scipy.optimize as opt
import roshelper
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Range
from std_msgs.msg import Float64MultiArray
from collections import OrderedDict


NODE_NAME = "decawave_localization"
POSE_TOPIC = "uwb_pose_2d"
POSE_COV_TOPIC = "pose_cov_2d"
POSE_TOPIC_3D = "uwb_pose_3d"
POSE_COV_TOPIC_3D = "pose_cov_3d"
ODOMETRY_TOPIC = "/odometry/filtered"
ALTITUDE_TOPIC = "/altitude"
n = roshelper.Node(NODE_NAME, anonymous=False)


@n.entry_point()
class DecaWaveLocalization(object):

    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "map")
        cov_x = rospy.get_param("~cov_x", 0.6)
        cov_y = rospy.get_param("~cov_y", 0.6)
        cov_z = rospy.get_param("~cov_z", 0.6)
        self.cov = self.cov_matrix(cov_x, cov_y, cov_z)
        self.ps_pub = rospy.Publisher(
            POSE_TOPIC, PoseStamped, queue_size=1)
        self.ps_cov_pub = rospy.Publisher(
            POSE_COV_TOPIC, PoseWithCovarianceStamped, queue_size=1)
        self.ps_pub_3d = rospy.Publisher(
            POSE_TOPIC_3D, PoseStamped, queue_size=1)
        self.ps_cov_pub_3d = rospy.Publisher(
            POSE_COV_TOPIC_3D, PoseWithCovarianceStamped, queue_size=1)
        self.last = None
        self.listener = tf.TransformListener()
        self.tag_range_topics = rospy.get_param("~tag_range_topics")
        self.subs = list()
        self.ranges = dict()
        self.tag_pos = dict()
        self.altitude = 0.0
        self.last_3d = None
        for topic in self.tag_range_topics:
            self.subs.append(rospy.Subscriber(topic, Range, self.range_cb))

    @n.subscriber(ODOMETRY_TOPIC, Odometry)
    def odom_callback(self, odom):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        z = odom.pose.pose.position.z
        self.altitude = z
        self.last = np.array([x, y])

    def transform_to_plane(self, tag_id, alt):
        h = self.tag_pos[tag_id][2]
        r = self.ranges[tag_id]
        dalt = abs(h - alt)
        if dalt < r:
            return math.sqrt(r ** 2 - dalt ** 2)
        else:
            return r

    @n.publisher(ALTITUDE_TOPIC, Range)
    def altitude_pub(self, alt):
        rng = Range()
        rng.field_of_view = math.pi * 0.1
        rng.max_range = 300
        rng.header.frame_id = "sonar_link"
        rng.header.stamp = rospy.Time.now()
        rng.range = alt
        return rng

    def find_xyz(self):
        if self.last is None:
            self.last = self.tag_pos.values()[0][:2]
        xy = self.error_altitude(self.altitude, return_args=True)
        return [xy[0], xy[1], self.altitude]

    def find_position_3d(self):
        if self.last_3d is None:
            self.last_3d = self.tag_pos.values()[0]
        tags, dists = [], []
        for tag_id in self.tag_pos.keys():
            tags.append(self.tag_pos[tag_id])
            dists.append(self.ranges[tag_id])
        res = opt.minimize(self.error_3d, self.last_3d,
                           jac=self.jac_3d, method="SLSQP",
                           args=(tags, dists))
        self.last_3d = res.x
        return res.x

    def error_3d(self, x, tags, dists):
        err = 0.0
        for tag, dist in zip(tags, dists):
            err += pow(pow(dist, 2) - pow(np.linalg.norm(x - tag), 2), 2)
        return err

    def jac_3d(self, x, tags, dists):
        jac_x = 0.0
        jac_y = 0.0
        jac_z = 0.0
        for tag, dist in zip(tags, dists):
            err = pow(dist, 2) - pow(np.linalg.norm(x - tag), 2)
            jac_x += err * (tag[0] - x[0])
            jac_y += err * (tag[1] - x[1])
            jac_z += err * (tag[2] - x[2])
        return np.array([jac_x, jac_y, jac_z])

    def error_altitude(self, alt, return_args=False):
        tags = list()
        dists = list()
        for tag_id in self.tag_pos.keys():
            tags.append(self.tag_pos[tag_id])
            dists.append(self.transform_to_plane(tag_id, alt))
        res = opt.minimize(
            self.error_xy, self.last,
            jac=self.jac, args=(tags, dists),
            method="SLSQP")
        if return_args:
            return res.x
        else:
            return res.fun + pow(res.fun - self.altitude, 2)

    def error_xy(self, x, tags, dists):
        err = 0.0
        for tag, dist in zip(tags, dists):
            err += pow(pow(dist, 2) - pow(np.linalg.norm(x - tag[:2]), 2), 2)
        return err

    def jac(self, x, tags, dists):
        jac_x = 0.0
        jac_y = 0.0
        for tag, dist in zip(tags, dists):
            err = pow(dist, 2) - pow(np.linalg.norm(x - tag[:2]), 2)
            jac_x += err * (tag[0] - x[0])
            jac_y += err * (tag[1] - x[1])
        return np.array([jac_x, jac_y])

    def range_cb(self, rng):
        self.ranges[rng.header.frame_id] = rng.range
        try:
            trans, _ = self.listener.lookupTransform(
                self.frame_id, rng.header.frame_id, rospy.Time(0))
            self.tag_pos[rng.header.frame_id] = np.array(trans[:3])
        except:
            return

        if len(self.ranges.values()) == 6 and len(self.tag_pos.values()) == 6:
            pos = self.find_xyz()
            pos_3d = self.find_position_3d()
            self.altitude_pub(pos[2])
            self.publish_position(
                pos, self.ps_pub, self.ps_cov_pub, self.cov)
            self.publish_position(
                pos_3d, self.ps_pub_3d, self.ps_cov_pub_3d, self.cov)
            self.ranges = dict()


    def publish_position(self, pos, ps_pub, ps_cov_pub, cov):
        x, y = pos[0], pos[1]
        if len(pos) > 2:
            z = pos[2]
        else:
            z = 0
        ps = PoseStamped()
        ps_cov = PoseWithCovarianceStamped()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.header.frame_id = self.frame_id
        ps.header.stamp = rospy.get_rostime()
        ps_cov.header = ps.header
        ps_cov.pose.pose = ps.pose
        ps_cov.pose.covariance = cov
        ps_pub.publish(ps)
        ps_cov_pub.publish(ps_cov)

    def cov_matrix(self, x_cov, y_cov, z_cov):
        return [x_cov, 0, 0, 0, 0, 0,
                0, y_cov, 0, 0, 0, 0,
                0, 0, z_cov, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0]


if __name__ == "__main__":
    n.start(spin=True)
