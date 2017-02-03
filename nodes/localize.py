#!/usr/bin/env python

import tf
import numpy as np
import pykalman
import math
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Range
import itertools


NODE_NAME = "decawave_localization"
POSE_TOPIC = "pose"
POSE_COV_TOPIC = "pose_cov"
ERROR_PC_TOPIC = "error_cloud"



class point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class circle(object):
    def __init__(self, point, radius):
        self.center = point
        self.radius = radius

def get_two_points_distance(p1, p2):
    return math.sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2))

def get_two_circles_intersecting_points(c1, c2):
    p1 = c1.center 
    p2 = c2.center
    r1 = c1.radius
    r2 = c2.radius

    d = get_two_points_distance(p1, p2)
    # if to far away, or self contained - can't be done
    if d >= (r1 + r2) or d <= math.fabs(r1 -r2):
        return None

    a = (pow(r1, 2) - pow(r2, 2) + pow(d, 2)) / (2*d)
    h  = math.sqrt(pow(r1, 2) - pow(a, 2))
    x0 = p1.x + a*(p2.x - p1.x)/d 
    y0 = p1.y + a*(p2.y - p1.y)/d
    rx = -(p2.y - p1.y) * (h/d)
    ry = -(p2.x - p1.x) * (h / d)
    return [point(x0+rx, y0-ry), point(x0-rx, y0+ry)]

def get_all_intersecting_points(circles):
    points = []
    num = len(circles)
    for i in range(num):
        j = i + 1
        for k in range(j, num):
            res = get_two_circles_intersecting_points(circles[i], circles[k])
            if res:
                points.extend(res)
    return points

def is_contained_in_circles(point, circles):
    for i in range(len(circles)):
        if (get_two_points_distance(point, circles[i].center) > (circles[i].radius)):
            return False
    return True

def get_polygon_center(points):
    center = point(0, 0)
    num = len(points)
    for i in range(num):
        center.x += points[i].x
        center.y += points[i].y
    center.x /= num
    center.y /= num
    return center



class DecaWaveLocalization:

    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "map")
        trans_mat = np.array(rospy.get_param("~transition_matrix"))
        obs_mat = np.array(rospy.get_param("~observation_matrix"))
        self.ps_pub = rospy.Publisher(POSE_TOPIC, PoseStamped, queue_size=1)
        self.ps_cov_pub = rospy.Publisher(
            POSE_COV_TOPIC, PoseWithCovarianceStamped, queue_size=1)
        self.rate = rospy.Rate(rospy.get_param("frequency", 30))
        self.ps = PoseStamped()
        self.ps.header.frame_id = self.frame_id
        self.ps_cov = PoseWithCovarianceStamped()
        self.ps_cov.header.frame_id = self.frame_id
        self.last = None
        self.kf = pykalman.KalmanFilter(
            transition_matrices=trans_mat.reshape(2, 2),
            observation_matrices=obs_mat.reshape(2, 2))
        self.fsm = np.array(rospy.get_param("~initial_state"))
        self.fsc = np.array(rospy.get_param("~initial_cov")).reshape(2, 2)
        self.cov_sensor = rospy.get_param("~cov_sensor", 0.01)
        self.listener = tf.TransformListener()
        self.tag_range_topics = rospy.get_param("~tag_range_topics")
        self.subs = list()
        self.ranges = dict()
        self.tag_pos = dict()
        for topic in self.tag_range_topics:
            self.subs.append(rospy.Subscriber(topic, Range, self.range_cb))

    def range_cb(self, rng):
        self.ranges[rng.header.frame_id] = rng.range
        #print self.ranges
        try:
            (trans, _) = self.listener.lookupTransform(
                self.frame_id, rng.header.frame_id, rospy.Time(0))
            self.tag_pos[rng.header.frame_id] = trans[:2]
        except:
            return

        dists = self.ranges.values()
        # print self.ranges
        # print self.tag_pos
        # print dists
        # print self.tag_pos
        if len(self.tag_pos.keys()) == len(dists) and len(self.tag_pos.keys()) >= 3:

            # get combinations of 3 from the 4 distances
            combinations = itertools.combinations(range(len(dists)), 3)
            xs, ys = [], []

            for combo in combinations:
                if 2 in combo:
                    pass
                distances = [dists[i] for i in combo]
                all_keys = self.ranges.keys()
                tag_keys = [all_keys[i] for i in combo]
                x, y = self.trilaterate(distances, tag_keys)
                xs.append(x)
                ys.append(y)

            x = np.mean(xs)
            y = np.mean(ys)
            #x = xs[0]
            #y = ys[0]

            print math.sqrt(x**2 + y**2)

            self.fsm, self.fsc = self.kf.filter_update(
                self.fsm, self.fsc, np.array([x, y]))
            self.ps.pose.position.x = self.fsm[0]
            self.ps.pose.position.y = self.fsm[1]
            self.ps.header.stamp = rospy.get_rostime()
            self.ps_cov.pose.covariance = self.cov_matrix(0.5,
                                                          0.5)
            self.ps_cov.header.stamp = rospy.get_rostime()
            self.ps_cov.pose.pose = self.ps.pose
            self.ps_pub.publish(self.ps)
            self.ps_cov_pub.publish(self.ps_cov)

    def cov_matrix(self, x_cov, y_cov):
        return [x_cov, 0, 0, 0, 0, 0,
                0, y_cov, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0]

    def trilaterate(self, rs, tag_keys):
        # xs, ys, rs = [], [], []
        xs, ys, points, circles = [], [], [], []
        for i, key in enumerate(tag_keys):
            new_x = self.tag_pos[key][0]
            new_y = self.tag_pos[key][1]
            xs.append(new_x)
            ys.append(new_y)
            # rs.append(self.ranges[key])


            # new_point = point(new_x,new_y)
            # points.append(new_point)
            # new_circle = circle(new_point,rs[i])
            # circles.append(new_circle)

        # inner_points = []
        # for p in get_all_intersecting_points(circles):
        #     if is_contained_in_circles(p, circles):
        #         inner_points.append(p) 
        # if len(inner_points) > 0:   
        #     center = get_polygon_center(inner_points)
        #     return center.x, center.y
        # print xs
        # print ys
        # print rs

        S = (pow(xs[2], 2.) - pow(xs[1], 2.) + pow(ys[2], 2.)
             - pow(ys[1], 2.) + pow(rs[1], 2.) - pow(rs[2], 2.)) / 2.0
        T = (pow(xs[0], 2.) - pow(xs[1], 2.) + pow(ys[0], 2.)
             - pow(ys[1], 2.) + pow(rs[1], 2.) - pow(rs[0], 2.)) / 2.0
        y = ((T * (xs[1] - xs[2])) - (S * (xs[1] - xs[0]))) \
            / (((ys[0] - ys[1]) * (xs[1] - xs[2]))
               - ((ys[2] - ys[1]) * (xs[1] - xs[0])))
        x = ((y * (ys[0] - ys[1])) - T) / (xs[1] - xs[0])

        #print x, y

        return x, y

if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    dd = DecaWaveLocalization()
    rospy.spin()
