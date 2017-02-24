#!/usr/bin/env python

from __future__ import division
import tf
import numpy as np
import rospy
import scipy.optimize as opt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Range

NODE_NAME = "decawave_localization"
POSE_TOPIC_3D = "uwb_pose_3d"
POSE_TOPIC_2D = "uwb_pose_2d"
POSE_COV_TOPIC_3D = "pose_cov_3d"
POSE_COV_TOPIC_2D = "pose_cov_2d"
ERROR_PC_TOPIC = "error_cloud"
ODOMETRY_TOPIC = "/odometry/filtered"


class DecaWaveLocalization:

    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "map")
        cov_x_2d = rospy.get_param("~cov_x_2d", 0.6)
        cov_y_2d = rospy.get_param("~cov_y_2d", 0.6)
        cov_z_2d = rospy.get_param("~cov_z_2d", 0.6)
        cov_x_3d = rospy.get_param("~cov_x_3d", 0.6)
        cov_y_3d = rospy.get_param("~cov_y_3d", 0.6)
        cov_z_3d = rospy.get_param("~cov_z_3d", 0.6)
        self.two_d_tags = rospy.get_param("~two_d_tags")
        self.cov_2d = self.cov_matrix(cov_x_2d, cov_y_2d, cov_z_2d)
        self.cov_3d = self.cov_matrix(cov_x_3d, cov_y_3d, cov_z_3d)
        self.ps_pub_3d = rospy.Publisher(
            POSE_TOPIC_3D, PoseStamped, queue_size=1)
        self.ps_pub_2d = rospy.Publisher(
            POSE_TOPIC_2D, PoseStamped, queue_size=1)
        self.ps_cov_pub_3d = rospy.Publisher(
            POSE_COV_TOPIC_3D, PoseWithCovarianceStamped, queue_size=1)
        self.ps_cov_pub_2d = rospy.Publisher(
            POSE_COV_TOPIC_2D, PoseWithCovarianceStamped, queue_size=1)
        self.odom_sub = rospy.Subscriber(
            ODOMETRY_TOPIC, Odometry, self.odom_callback)
        self.rate = rospy.Rate(rospy.get_param("~frequency", 30))
        self.last_3d = None
        self.last_2d = None
        self.listener = tf.TransformListener()
        self.tag_range_topics = rospy.get_param("~tag_range_topics")
        self.subs = list()
        self.ranges = dict()
        self.tag_pos = dict()
        for topic in self.tag_range_topics:
            self.subs.append(rospy.Subscriber(topic, Range, self.range_cb))

    def odom_callback(self, odom):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        self.last_2d = np.array([x, y])

    def find_position_3d(self):
        if self.last_3d is None:
            self.last_3d = self.tag_pos.values()[0]
        res = opt.minimize(self.error, self.last_3d,
                           jac=self.jac, method="SLSQP")
        self.last_3d = res.x
        return res.x

    def find_position_2d(self):
        if self.last_2d is None:
            self.last_2d = self.tag_pos.values()[0][:2]
        res = opt.minimize(self.error_2d, self.last_2d,
                           jac=self.jac_2d, method="SLSQP")
        # self.last_2d = res.x
        return res.x

    def error(self, x):
        err = 0.0
        for tag_id in self.ranges.keys():
            tag = self.tag_pos[tag_id]
            dist = self.ranges[tag_id]
            err += pow(pow(dist, 2) - pow(np.linalg.norm(x - tag), 2), 2)
        return err

    def error_2d(self, x):
        err = 0.0
        for tag_id in self.two_d_tags:
            tag = self.tag_pos[tag_id][:2]
            dist = self.ranges[tag_id]
            err += pow(pow(dist, 2) - pow(np.linalg.norm(x - tag), 2), 2)
        return err

    def jac(self, x):
        jac_x = 0.0
        jac_y = 0.0
        jac_z = 0.0
        for tag_id in self.ranges.keys():
            tag = self.tag_pos[tag_id]
            dist = self.ranges[tag_id]
            err = pow(dist, 2) - pow(np.linalg.norm(x - tag), 2)
            jac_x += err * (tag[0] - x[0])
            jac_y += err * (tag[1] - x[1])
            jac_z += err * (tag[2] - x[2])
        return np.array([jac_x, jac_y, jac_z])

    def jac_2d(self, x):
        jac_x = 0.0
        jac_y = 0.0
        for tag_id in self.two_d_tags:
            tag = self.tag_pos[tag_id][:2]
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
            self.tag_pos[rng.header.frame_id] = np.array(trans[:3])
        except:
            return

        if len(self.ranges.values()) == 6 and len(self.tag_pos.values()) == 6:
            pos_3d = self.find_position_3d()
            pos_2d = self.find_position_2d()
            self.publish_position(
                pos_3d, self.ps_pub_3d, self.ps_cov_pub_3d, self.cov_3d)
            self.publish_position(
                pos_2d, self.ps_pub_2d, self.ps_cov_pub_2d, self.cov_2d)
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
    rospy.init_node(NODE_NAME, anonymous=False)
    dd = DecaWaveLocalization()
    rospy.spin()
