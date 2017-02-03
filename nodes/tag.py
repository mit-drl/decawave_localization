#!/usr/bin/env python

import math
import rospy
import serial
from sensor_msgs.msg import Range


NODE_NAME = "decawave_tag"
DIST_TOPIC = "range"


class DecaWaveTag:

    def __init__(self):
        port = rospy.get_param('~port', '/dev/ttyACM0')
        baud = rospy.get_param('~baud', 115200)
        self.tag_names = rospy.get_param("tag_names")
        self.offsets = rospy.get_param("offsets")
        self.rate = rospy.Rate(rospy.get_param("frequency", 100))
        self.rng = Range()
        self.rng.field_of_view = math.pi * 0.1
        self.rng.min_range = 0
        self.rng.max_range = 300
        self.ser = serial.Serial(port=port, timeout=None, baudrate=baud)
        self.pub = None
        self.offset = 0.0

    def start(self):
        self.ser.close()
        self.ser.open()
        self.run()

    def run(self):
        while not rospy.is_shutdown():
            dist = self.get_dist()
            if dist is not None and self.pub is not None:
                self.rng.range = dist
                # if self.rng.header.frame_id == "tag_right_front":
                #     print dist
                self.pub.publish(self.rng)
            self.rate.sleep()
        self.ser.close()

    def get_dist(self):
        raw_data = self.ser.readline()
        data = raw_data.split()

        if self.pub is None:
            try:
                tag_id = int(data[-1].split(":")[0][-1])
                self.offset = float(self.offsets[tag_id])
                self.rng.header.frame_id = self.tag_names[tag_id]
                topic_name = "/{}/{}".format(self.tag_names[tag_id], DIST_TOPIC)
                self.pub = rospy.Publisher(topic_name, Range, queue_size=1)
            except IndexError:
                pass
        self.rng.header.stamp = rospy.Time.now()

        if len(data) > 0 and data[0] == 'mc':
            mask = int(data[1], 16)
            if (mask & 0x01):
                dist = int(data[2], 16) / 1000.0

            return dist + self.offset *(dist/8.0)


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    da = DecaWaveTag()
    da.start()