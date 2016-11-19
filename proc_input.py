import rospy
import time
import datetime
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import sys
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np

class proc_input:
	Queue_size = 10
	proc_rgb = False
	proc_cloud = False
	proc_odom = False

	def __init__(self,name,rgb=True,cloud=False,odom=False,depth=False):	
		rospy.init_node(name)	
		if rgb:		
			self.proc_rgb = True
			self.RGB_TOPIC = '/camera/rgb/image_raw'
			self.RGB_Queue = []
			self.bridge = CvBridge()
			self.sub_rgb = rospy.Subscriber(self.RGB_TOPIC, Image, self.callback_rgb, queue_size=1)
		if cloud: 
			self.proc_cloud = True
			self.Cloud_TOPIC = '/camera/depth/points'
			self.Cloud_Queue = []
			self.sub_cloud = rospy.Subscriber(self.Cloud_TOPIC, PointCloud2, self.callback_point, queue_size=1)
		if odom:
			self.proc_odom = True
			self.Odom_TOPIC = '/odom'
			self.Odom_Queue = []
			self.sub_odom = rospy.Subscriber(self.Odom_TOPIC, Odometry, self.callback_odom, queue_size=1)
		if depth:
			self.proc_depth = True
			self.Depth_TOPIC = '/camera/depth/image_raw'
			self.Depth_Queue = []
			self.sub_depth = rospy.Subscriber(self.Depth_TOPIC, Image, self.callback_depth, queue_size=1)


	def callback_rgb(self, msg):		
		self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
		self.RGB_Queue.append([msg.header.stamp.to_sec(),self.img])
		if len(self.RGB_Queue) > self.Queue_size : self.RGB_Queue.remove(self.RGB_Queue[0])
		#print len(self.RGB_Queue)
		
	def callback_point(self, msg):		
		self.cloud = msg
		self.Cloud_Queue.append([msg.header.stamp.to_sec(),msg])
		if len(self.Cloud_Queue) > self.Queue_size : self.Cloud_Queue.remove(self.Cloud_Queue[0])
		#print len(self.Cloud_Queue)


	def callback_odom(self, msg):		
		q = [msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w]
		roll,pitch,yaw = euler_from_quaternion(q)
		self.odom = [msg.pose.pose.position.x,msg.pose.pose.position.y,yaw]
		self.Odom_Queue.append([msg.header.stamp.to_sec(),self.odom])
		if len(self.Odom_Queue) > self.Queue_size : self.Odom_Queue.remove(self.Odom_Queue[0])
		#print len(self.Odom_Queue)
	def callback_depth(self, msg):		
		self.img = self.bridge.imgmsg_to_cv2(msg, '16UC1')
		self.Depth_Queue.append([msg.header.stamp.to_sec(),self.img])
		if len(self.Depth_Queue) > self.Queue_size : self.Depth_Queue.remove(self.Depth_Queue[0])
		#print len(self.Depth_Queue)

	def get_rgb(self,idx = -1):
		return self.RGB_Queue[idx]

	def get_depth(self,idx = -1):
		return self.Depth_Queue[idx]

	def get_cloud(self,x,y,idx = -1):		
		temp = pc2.read_points(self.Cloud_Queue[idx][1], field_names=None, skip_nans=False, uvs=[[x,y]])
		result = np.array([])		
		for item in temp:
			result= np.append(result,np.array([item[0],item[1],item[2]]),axis=0)			
		return [self.Cloud_Queue[idx][0],result]

	def get_odom(self,idx = -1):
		return self.Odom_Queue[idx]

	def get_rgb_by_time(self,timestamp):
		idx=0
		for i in range(1,len(self.RGB_Queue)):
			if abs(self.RGB_Queue[i][0] - timestamp) <= abs(self.RGB_Queue[i-1][0] - timestamp) : idx = i
		return self.RGB_Queue[idx]

	def get_depth_by_time(self,timestamp):
		idx=0
		for i in range(1,len(self.Depth_Queue)):
			if abs(self.Depth_Queue[i][0] - timestamp) <= abs(self.Depth_Queue[i-1][0] - timestamp) : idx = i
		return self.Depth_Queue[idx]

	def get_cloud_by_time(self,x,y,timestamp):
		idx=0
		for i in range(1,len(self.Cloud_Queue)):
			if abs(self.Cloud_Queue[i][0] - timestamp) <= abs(self.Cloud_Queue[i-1][0] - timestamp) : idx = i

		temp = pc2.read_points(self.Cloud_Queue[idx][1], field_names=None, skip_nans=False, uvs=[[x,y]])
		result = np.array([])		
		for item in temp:
			result= np.append(result,np.array([item[0],item[1],item[2]]),axis=0)			
		return [self.Cloud_Queue[idx][0],result]

	def get_odom_by_time(self,timestamp):
		idx=0
		for i in range(1,len(self.Odom_Queue)):
			if abs(self.Odom_Queue[i][0] - timestamp) <= abs(self.Odom_Queue[i-1][0] - timestamp) : idx = i
		return self.Odom_Queue[idx]	

	def get_synched_by_time(self,x,y,timestamp):
		minidx_cloud = 0
		minidx_odom = 0
		minidx_rgb = 0
		for i in range(1,len(self.Cloud_Queue)):
			if abs(self.RGB_Queue[i][0] - timestamp) <= abs(self.RGB_Queue[i-1][0] - timestamp) : minidx_rgb = i
			if abs(self.Cloud_Queue[i][0] - timestamp) <= abs(self.Cloud_Queue[i-1][0] - timestamp) : minidx_cloud = i
			if abs(self.Odom_Queue[i][0] - timestamp) <= abs(self.Odom_Queue[i-1][0] - timestamp) : minidx_odom = i
		#print [minidx_rgb, minidx_cloud,minidx_odom]
		#print timestamp
		#print [minidx_cloud,minidx_odom,minidx_rgb]
		return [self.get_rgb(minidx_rgb),self.get_cloud(x,y,minidx_cloud),self.get_odom(minidx_odom)]

	def get_synched_by_rgb(self,x,y,rgb_obj):
		minidx_cloud = 0
		minidx_odom = 0
		timestamp = rgb_obj[0]
		for i in range(1,len(self.Cloud_Queue)):
			if abs(self.Cloud_Queue[i][0] - timestamp) <= abs(self.Cloud_Queue[i-1][0] - timestamp) : minidx_cloud = i
			if abs(self.Odom_Queue[i][0] - timestamp) <= abs(self.Odom_Queue[i-1][0] - timestamp) : minidx_odom = i
		#print [minidx_rgb, minidx_cloud,minidx_odom]
		return [rgb_obj,self.get_cloud(x,y,minidx_cloud),self.get_odom(minidx_odom)]

	def get_synched_by_point(self,x,y,point_obj):
		minidx_rgb = 0
		timestamp = point_obj[0]
		minidx_odom = 0
		for i in range(1,len(self.Cloud_Queue)):
			if abs(self.RGB_Queue[i][0] - timestamp) <= abs(self.RGB_Queue[i-1][0] - timestamp) : minidx_rgb = i
			if abs(self.Odom_Queue[i][0] - timestamp) <= abs(self.Odom_Queue[i-1][0] - timestamp) : minidx_odom = i
		#print [minidx_rgb, minidx_cloud,minidx_odom]
		#print [minidx_rgb, minidx_odom]
		return [self.get_rgb(minidx_rgb),point_obj,self.get_odom(minidx_odom)]

	def get_synched_by_odom(self,x,y,odom_obj):
		minidx_rgb = 0
		minidx_cloud = 0
		timestamp = odom_obj[0]
		for i in range(1,len(self.Cloud_Queue)):
			if abs(self.RGB_Queue[i][0] - timestamp) <= abs(self.RGB_Queue[i-1][0] - timestamp) : minidx_rgb = i
			if abs(self.Cloud_Queue[i][0] - timestamp) <= abs(self.Cloud_Queue[i-1][0] - timestamp) : minidx_cloud = i
		#print [minidx_rgb, minidx_cloud,minidx_odom]
		return [self.get_rgb(minidx_rgb),self.get_cloud(x,y,minidx_cloud),odom_obj]

	def write_rgb(self,name):
		temp = self.get_rgb()
		cv2.imwrite(name,temp[1])
		return temp[0]
		
	def __del__(self): 
		#cv2.destroyAllWindows()
		if self.proc_rgb : self.sub_rgb.unregister()
		if self.proc_cloud : self.sub_cloud.unregister()
		if self.proc_odom : self.sub_odom.unregister()

def main():
	proc = proc_input('test',True,True,True)
	
	time.sleep(3)
	a = proc.get_synched_by_time(100,100,float(time.time()))
	print a[1]
	#rospy.spin()

if __name__ == '__main__':
    main()
