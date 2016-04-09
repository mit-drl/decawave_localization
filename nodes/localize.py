#!/usr/bin/env python

import numpy as np
import pykalman
import scipy.optimize as opt
import rospy
import serial
from geometry_msgs.msg import PoseStamped


NODE_NAME = "decawave_localization"
POSE_TOPIC = "pose"


class DecaWaveLocalization:

    def __init__(self):
        port = rospy.get_param('~port', '/dev/ttyACM3')
        baud = rospy.get_param('~baud', 9600)
        frame_id = rospy.get_param("~frame_id", "map")
        trans_mat = np.array(rospy.get_param("~transition_matrix"))
        obs_mat = np.array(rospy.get_param("~observation_matrix"))
        self.anchors = rospy.get_param("~anchors")
        self.pub = rospy.Publisher(POSE_TOPIC, PoseStamped, queue_size=1)
        self.rate = rospy.Rate(30)
        self.ps = PoseStamped()
        self.ps.header.frame_id = frame_id
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
                res = opt.minimize(
                    self.error, x0, jac=self.jac, args=(dists),
                    method="SLSQP")
                self.fsm, self.fsc = self.kf.filter_update(
                    self.fsm, self.fsc, res.x)
                self.last = res.x
            else:
                self.fsm, self.fsc = self.kf.filter_update(
                    self.fsm, self.fsc)
            self.ps.pose.position.x = self.fsm[0]
            self.ps.pose.position.y = self.fsm[1]
            self.ps.pose.position.z = 0
            self.ps.header.stamp = rospy.get_rostime()
            self.pub.publish(self.ps)
            self.rate.sleep()
        self.ser.close()

    def get_dists(self):
        dists = np.zeros((3, 1))
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

    def error(self, x, dists):
        err = 0.0
        for anchor, dist in zip(self.anchors, dists):
            err += pow(pow(dist, 2) - pow(np.linalg.norm(x - anchor), 2), 2)
        return err

if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    dd = DecaWaveLocalization()
    dd.start()
