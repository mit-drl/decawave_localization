#!/usr/bin/env python

import math
import numpy as np
import pykalman
import scipy.optimize as opt
import rospy
import serial
from itertools import combinations
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32


NODE_NAME = "decawave_localization"
POSE_TOPIC = "pose"
POSE_COV_TOPIC = "pose_cov"
ERROR_PC_TOPIC = "error_cloud"


class DecaWaveLocalization:

    def __init__(self):
        port = rospy.get_param('~port', '/dev/ttyACM3')
        baud = rospy.get_param('~baud', 9600)
        frame_id = rospy.get_param("~frame_id", "map")
        trans_mat = np.array(rospy.get_param("~transition_matrix"))
        obs_mat = np.array(rospy.get_param("~observation_matrix"))
        self.anchors = map(np.array, rospy.get_param("~anchors"))
        self.pub = rospy.Publisher(POSE_TOPIC, PoseStamped, queue_size=1)
        self.error_pc_pub = rospy.Publisher(
            ERROR_PC_TOPIC, PointCloud, queue_size=1)
        self.cov_pub = rospy.Publisher(POSE_COV_TOPIC,
                                       PoseWithCovarianceStamped,
                                       queue_size=1)
        self.rate = rospy.Rate(rospy.get_param("frequency", 30))
        self.ps = PoseStamped()
        self.pwcs = PoseWithCovarianceStamped()
        self.ps.header.frame_id = frame_id
        self.pwcs.header.frame_id = frame_id
        self.ser = serial.Serial(port=port, timeout=10, baudrate=baud)
        self.last = None
        self.kf = pykalman.KalmanFilter(
            transition_matrices=trans_mat.reshape(2, 2),
            observation_matrices=obs_mat.reshape(2, 2))
        self.fsm = np.array(rospy.get_param("~initial_state"))
        self.fsc = np.array(rospy.get_param("~initial_cov")).reshape(2, 2)

    def start(self):
        self.ser.close()
        self.ser.open()
        self.run()

    def run(self):
        while not rospy.is_shutdown():
            if self.last is None:
                x0 = self.anchors[0]
            else:
                x0 = self.last
            dists = self.get_dists()
            if not dists is None:
                # pos = self.get_position(dists)
                res = opt.minimize(
                    self.error, x0, jac=self.jac, args=(dists,),
                    method="SLSQP")
                self.fsm, self.fsc = self.kf.filter_update(
                    self.fsm, self.fsc, res.x)
                self.last = res.x
                self.publish_error_pc(dists)
            else:
                self.fsm, self.fsc = self.kf.filter_update(
                    self.fsm, self.fsc)
            self.ps.pose.position.x = self.fsm[0]
            self.ps.pose.position.y = self.fsm[1]
            self.ps.pose.position.z = 0
            self.ps.header.stamp = rospy.get_rostime()
            self.pwcs.pose.pose.position.x = self.fsm[0]
            self.pwcs.pose.pose.position.y = self.fsm[1]
            self.pwcs.pose.pose.position.z = 0
            self.pwcs.pose.covariance = self.covariance_matrix(
                self.fsc[0][0], self.fsc[1][1])
            self.pwcs.header.stamp = rospy.get_rostime()
            self.pub.publish(self.ps)
            self.cov_pub.publish(self.pwcs)
            self.rate.sleep()
        self.ser.close()

    def covariance_matrix(self, x_p, y_p):
        return [x_p, 0, 0, 0, 0, 0,
                0, y_p, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0]

    def get_circle_intersections(self, r0, r1, p0, p1):
        d = np.linalg.norm(p1 - p0)
        b = (r0 ** 2 - r1 ** 2) / (2 * d)
        h = math.sqrt(abs(r0 ** 2 - b ** 2))
        dr = (p1 - p0) / d
        C = p0 + dr * b
        int0 = C + np.array([-dr[1], dr[0]]) * h
        int1 = C - np.array([-dr[1], dr[0]]) * h
        return [int0, int1]

    def get_position(self, dists):
        err = None
        ret_p = None
        inds = set(range(len(self.anchors)))
        for i, j in combinations(range(len(self.anchors)), 2):
            left = list(inds - set([i, j]))[0]
            ps = self.get_circle_intersections(
                dists[i], dists[j], self.anchors[i], self.anchors[j])
            inner_err = None
            p_star = None
            for p in ps:
                dist = np.linalg.norm(p - self.anchors[left])
                if inner_err is None or dist < inner_err:
                    inner_err = dist
                    p_star = p
            n_err = self.error(p_star, dists)
            if err is None or n_err < err:
                err = n_err
                ret_p = p_star
        return ret_p

    def get_dists(self):
        dists = np.zeros((3,))
        raw_data = self.ser.readline()
        if raw_data == serial.to_bytes([]):
            print "serial timeout"
        else:
            data = raw_data.split()

        if len(data) > 0 and data[0] == 'mc':
            mask = int(data[1], 16)
            if (mask & 0x01):
                dists[0] = int(data[2], 16) / 1000.0
            if (mask & 0x02):
                dists[1] = int(data[3], 16) / 1000.0
            if (mask & 0x04):
                dists[2] = int(data[4], 16) / 1000.0
            return dists

    def jac(self, x, dists):
        jac_x = 0.0
        jac_y = 0.0
        for anchor, dist in zip(self.anchors, dists):
            err = pow(dist, 2) - pow(np.linalg.norm(x - anchor), 2)
            jac_x += err * (anchor[0] - x[0])
            jac_y += err * (anchor[1] - x[1])
        return np.array([jac_x, jac_y])

    def normal(self, x, mean, std):
        con = 1.0 / math.sqrt(2 * math.pi * std ** 2)
        dep = math.exp(-pow(x - mean, 2) / (2 * std ** 2))
        return con * dep

    def gauss_prob(self, x, dists, sigs):
        inv_prob = 0.0
        for anchor, dist, sig in zip(self.anchors, dists, sigs):
            est_dist = np.linalg.norm(x - anchor)
            inv_prob -= math.log1p(self.normal(est_dist, dist, sig))
            # inv_prob -= self.normal(est_dist, dist, sig)
        return inv_prob

    def error(self, x, dists):
        err = 0.0
        for anchor, dist in zip(self.anchors, dists):
            err += pow(pow(dist, 2) - pow(np.linalg.norm(x - anchor), 2), 2)
            # err += abs(dist - np.linalg.norm(x - anchor))
        return err

    def publish_error_pc(self, dists):
        xmin = -5
        xmax = 5
        ymin = -5
        ymax = 5
        xres = 0.08
        yres = 0.08
        pc = PointCloud()
        ch = ChannelFloat32()
        pc.header.stamp = rospy.get_rostime()
        pc.header.frame_id = "map"
        ch.name = "error"
        ys = np.linspace(ymin, ymax, (ymax - ymin) / yres)
        xs = np.linspace(xmin, xmax, (xmax - xmin) / xres)
        for x in xs:
            for y in ys:
                p = Point32()
                p.x = x
                p.y = y
                p.z = -0.5
                pc.points.append(p)
                ch.values.append(self.error(np.array([x, y]), dists))
        pc.channels.append(ch)
        self.error_pc_pub.publish(pc)


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    dd = DecaWaveLocalization()
    dd.start()
