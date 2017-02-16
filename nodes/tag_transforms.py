#!/usr/bin/env python

import math
import rospy
import tf
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose

NODE_NAME = "tag_transforms"
LIDAR_X = 0.406
LIDAR_Y = 0.0
LIDAR_Z = 1.623
LIDAR_X_OFFSET = 1.05
LIDAR_Y_OFFSET = 0.49
LIDAR_Z_OFFSET = 0.056

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
		while not rospy.is_shutdown():
			x = LIDAR_X - LIDAR_X_OFFSET
			y = LIDAR_Y + LIDAR_Y_OFFSET
			z = LIDAR_Z - LIDAR_Z_OFFSET
			for i, tag in enumerate(self.transforms):
				x = x + tag['x']
				y = y + tag['y']
				z = z + tag['z']
				self.br.sendTransform((x, y, z),
						tf.transformations.quaternion_from_euler(0, 0, 0),
						rospy.Time.now(),
						self.tag_names[i],
						self.frame_id)
			self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    tg = TagTransform(30)
    rospy.spin()
