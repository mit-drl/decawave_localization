#!/usr/bin/env python

import numpy as np
import scipy.optimize as opt
import rospy
import serial
from geometry_msgs.msg import PoseStamped


NODE_NAME = "decawave_localization"
POSE_TOPIC = "radio_pose"


class DecaWaveLocalization:

    def __init__(self):
        port = rospy.get_param('~port', '/dev/ttyACM3')
        baud = rospy.get_param('~baud', 9600)
        frame_id = rospy.get_param("~frame_id", "map")
        self.anchors = [np.array([0, 0]),
                        np.array([0.5, 0.4]),
                        np.array([0.34, 0])]
        self.pub = rospy.Publisher(POSE_TOPIC, PoseStamped, queue_size=1)
        self.rate = rospy.Rate(30)
        self.ps = PoseStamped()
        self.ps.header.frame_id = frame_id
        self.ser = serial.Serial(port=port, timeout=10, baudrate=baud)

    def start(self):
        self.ser.close()
        self.ser.open()
        self.run()

    def run(self):
        range0 = -1
        range1 = -1
        range2 = -1
        while not rospy.is_shutdown():
            raw_data = self.ser.readline()
            if raw_data == serial.to_bytes([]):
                print "serial timeout"
            else:
                data = raw_data.split()

            if data[0] == 'mc':
                mask = int(data[1], 16)
                if (mask & 0x01):
                    range0 = int(data[2], 16) / 1000.0
                if (mask & 0x02):
                    range1 = int(data[3], 16) / 1000.0
                if (mask & 0x04):
                    range2 = int(data[4], 16) / 1000.0

                dists = np.array([range0, range1, range2])
                res = opt.minimize(self.error, self.anchors[0],
                                   # jac=self.jac,
                                   args=(dists),
                                   method="SLSQP")
                if res.success:
                    self.ps.pose.position.x = res.x[0]
                    self.ps.pose.position.y = res.x[1]
                    self.ps.pose.position.z = 0
                    self.ps.header.stamp = rospy.get_rostime()
                    self.pub.publish(self.ps)
                    self.rate.sleep()
        self.ser.close()

    def jac(self, x, dists):
        diff_x = 0.0
        diff_y = 0.0
        for anchor, dist in zip(self.anchors, dists):
            diff_x += anchor[0] - x[0]
            diff_y += anchor[1] - x[1]
        err_x = 4 * self.error(x, dists, pw=1) * diff_x
        err_y = 4 * self.error(x, dists, pw=1) * diff_y
        return np.array([err_x, err_y])

    def error(self, x, dists, pw=2):
        err = 0.0
        for anchor, dist in zip(self.anchors, dists):
            err += pow(pow(dist, 2) - pow(np.linalg.norm(x - anchor), 2), pw)
        return err

if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    dd = DecaWaveLocalization()
    dd.start()
