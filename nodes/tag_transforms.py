#!/usr/bin/env python

import math
import rospy
import tf
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose

NODE_NAME = "tag_transforms"
LIDAR_X = 0.55
LIDAR_Y = 0.0
LIDAR_Z = 1.99
LIDAR_X_OFFSET = 0.45
LIDAR_Y_OFFSET = 0.29
LIDAR_Z_OFFSET = 0.14

class TagTransform(object):

	def __init__(self, frequency):

		self.tag_names = rospy.get_param("tag_names")
		self.frame_id = rospy.get_param("~frame_id")
		self.transforms = rospy.get_param("tags")
		self.rate = rospy.Rate(frequency)
		#self.sub = rospy.Subscriber(self.tag_position_topic, PoseArray, self.tag_position_cb)

		self.br = tf.TransformBroadcaster()

		self.make_transforms()


	def make_transforms(self):
		keys = self.transforms.keys()

		while not rospy.is_shutdown():
			x = LIDAR_X - LIDAR_X_OFFSET
			y = LIDAR_Y + LIDAR_Y_OFFSET
			for i, key in enumerate(keys):
				tag = self.transforms[key]
				x = x + tag['x']
				y = y + tag['y']
				self.br.sendTransform((x, y, LIDAR_Z - LIDAR_Z_OFFSET),
						tf.transformations.quaternion_from_euler(0, 0, 0),
						rospy.Time.now(),
						self.tag_names[i],
						self.frame_id)
			self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    tg = TagTransform(30)
    rospy.spin()