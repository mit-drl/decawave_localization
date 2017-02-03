#!/usr/bin/env python

import math
import rospy
import numpy as np
from sensor_msgs.msg import Range

NODE_NAME = "decawave_kalman"

class Kalman(object):

	def __init__(self):
		qx = 10
		qy = 10
		qz = 10

		rx_uwb = 0.005
		ry_uwb = 0.005
		rz_uwb = 100

		rx_odom = 0.5
		ry_odom = 0.5
		rz_odom = 0.5

		rz_alt = 0.5

		self.x = np.array([[0],[0],[0],[0],[0],[0]])

		#this dt is just a dummy variable
		dt = 0.01
		self.F = np.matrix([[1,0,0,dt,0,0],
							[0,1,0,0,dt,0],
							[0,0,1,0,0,dt],
							[0,0,0,1,0,0]
							[0,0,0,0,1,0],
							[0,0,0,0,0,1]])
		self.P = 10*np.eye(6)
		self.H_uwb = np.matrix([[1,0,0,0,0,0],
								[0,1,0,0,0,0],
								[0,0,0,0,0,0]])
		self.H_odom = np.matrix([[0,0,0,1,0,0],
								 [0,0,0,0,1,0],
								 [0,0,0,0,0,1]])
		self.H_alt = np.array([0,0,1,0,0,0])
		self.Q = np.diag(np.array([0,0,0,qx,qy,qz]))
		self.R_uwb = np.diag(np.array([rx_uwb,ry_uwb,rz_uwb]))
		self.R_odom = np.diag(np.array([rx_odom,ry_odom,rz_odom]))
		self.R_alt = rz_alt

		self.tag_range_topic = rospy.get_param("~tag_range_topic")
		self.tag_range_filtered_topic = self.tag_range_topic + "_filtered"
		self.sub = rospy.Subscriber(self.tag_range_topic, Range, self.range_cb)
		self.pub = rospy.Publisher(self.tag_range_filtered_topic, Range, queue_size=1)

	def prediction(self, dt):
		self.F = np.matrix([[1,0,0,dt,0,0],
							[0,1,0,0,dt,0],
							[0,0,1,0,0,dt],
							[0,0,0,1,0,0]
							[0,0,0,0,1,0],
							[0,0,0,0,0,1]])

		self.x = self.F * self.x
		self.P = self.F * self.P * self.F.transpose() + self.Q

	def update_uwb(self,data):
		x_m = data.pose.position.x
		y_m = data.pose.position.y
		z_m = 0.0

		meas = np.matrix([[x_m],[y_m],[z_m]])

		x = self.x
		F = self.F
		H = self.H_uwb
		P = self.P
		R = self.R_uwb

		y = meas - H*x
		S = H*P*H.transpose() + R
		K = P*H.tranpose()/S
		x = x + K*y
		P = (np.eye(6) - K*H)*P
		self.x = x
		self.P = P

	def update_odom(self,data):
		vx_m = data.twist.x
		v

		x = self.x
		F = self.F
		H = self.H_odom
		P = self.P
		R = self.R_odom

		y = meas - H*x
		S = H*P*H.transpose() + R
		K = P*H.transpose()/S
		x = x + K*y
		P = (np.eye(6) - K*H)*P
		self.x = x
		self.P = P

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