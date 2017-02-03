#!/usr/bin/env python

import math
import rospy
import tf
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose

NODE_NAME = "tag_transforms"

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
			x = 0.0
			y = 0.0
			for i, key in enumerate(keys):
				tag = self.transforms[key]
				x = x + tag['x']
				y = y + tag['y']
				self.br.sendTransform((x, y, 0),
						tf.transformations.quaternion_from_euler(0, 0, 0),
						rospy.Time.now(),
						self.tag_names[i],
						self.frame_id)
			self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    tg = TagTransform(30)
    rospy.spin()