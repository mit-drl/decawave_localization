#!/usr/bin/env python

import math
import rospy
from sensor_msgs.msg import Range

NODE_NAME = "decawave_kalman"

class Kalman(object):

	def __init__(self):
		self.x = 0 # process prediction
		self.p = 0 # error prediction
		self.q = 0.0001 # process noise
		self.k = 0 # kalman gain
		self.r = rospy.get_param("sensor_noise") # sensor noise

		self.tag_range_topic = rospy.get_param("~tag_range_topic")
		self.tag_range_filtered_topic = self.tag_range_topic + "_filtered"
		self.sub = rospy.Subscriber(self.tag_range_topic, Range, self.range_cb)
		self.pub = rospy.Publisher(self.tag_range_filtered_topic, Range, queue_size=1)

	def range_cb(self, data):
		filtered_range = data

		self.p = self.p + self.q
		self.k = self.p/(self.p + self.r)
		self.x = self.x + self.k * (data.range - self.x)
		self.p = (1.0 - self.k) * self.p

		filtered_range.range = self.x
		self.pub.publish(filtered_range)

if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    kalman = Kalman()
    rospy.spin()
