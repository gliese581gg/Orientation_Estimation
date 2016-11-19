import os.path
import re
import sys
import tarfile
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
from scipy import misc
from tensorflow.python.platform import gfile
import time
import cv2
import thread
from detecting_msg.msg import *
import rospy
import proc_input
import signal

MODE = 'training'
CKPT = 'ckpt/weights_ori_H36_6300.ckpt'

class ori_esti:
	ROS_TIMEOUT = 60
	alpha = 0.1
	gpu_fraction = 0.9
	WH = 32
	CHANNEL = 3
	LABEL_DIM = 8	
	BUFFER_SIZE = 1000
	TRAIN_SIZE = -1
	TEST_SIZE = 3000
	MODEL_DIR = 'model/'
	IMG_PATH = 'crop_resize/'
	IMG_PATH_TEST = 'crop_resize/'
	IMG_PATH_REAL = 'crop_resize/'
	LABEL_FILE = 'labels/labels_training_depth.txt'
	LABEL_FILE_TEST = 'labels/labels_training_depth.txt'
	LABEL_FILE_REAL = 'labels/labels_training_depth.txt'
	CKPT_NAME_PREFIX = 'ckpt/weights_ori_finetuned_'
	BATCH_SIZE=100
	DECAY_STEP = BATCH_SIZE * 100
	INITIAL_LR = 0.0001
	NUM_EPOCHS = 100
	EVAL_FREQUENCY = 10
	TEST_FREQUENCY = 100
	REAL_FREQUENCY = 500
	SAVE_FREQUENCY = 100
	keep_prob = tf.placeholder("float")
	x = tf.placeholder('float',shape=(None, WH,WH,CHANNEL))
	y_ = tf.placeholder('float',shape=(None, LABEL_DIM))
	keep_prob = tf.placeholder(tf.float32)
	MAX_DISTANCE = 4000
	LOG_TRAIN = 'log_train_finetuned.txt'
	LOG_TEST = 'log_test_finetuned.txt'
	LOG_CONFU = 'log_confu_finetuned.txt'


	def __init__(self,weight_file = ''):
		self.conv_1 = self.conv_layer(1,self.x,64,5,2)
		self.norm_1 = tf.nn.lrn(self.conv_1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')
		#self.pool_1 = self.pooling_layer(6,self.norm_1,3,2)
		self.conv_2 = self.conv_layer(2,self.norm_1,64,5,2)
		self.norm_2 = tf.nn.lrn(self.conv_2,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')
		self.fc_3 = self.fc_layer(3,self.norm_2,384,flat=True,linear=False)
		self.fc_4 = self.fc_layer(4,self.fc_3,192,flat=False,linear=False)
		self.fc_5 = self.fc_layer(5,self.fc_4,8,flat=False,linear=True)
		
		self.y = tf.nn.softmax(self.fc_5)
		#Loss
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.fc_5, self.y_))

		self.glob_step = tf.Variable(tf.constant(0,shape=[]),name='global_step',trainable=False)
		self.optimizer = tf.train.AdamOptimizer(self.INITIAL_LR).minimize(self.cross_entropy,global_step = self.glob_step)

		self.saver = tf.train.Saver()
		self.gpu_config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction))
		self.sess = tf.Session(config=self.gpu_config)
		self.sess.run(tf.initialize_all_variables())

		if weight_file != '' :
			print 'Loading weights from ' + weight_file
			self.saver.restore(self.sess, weight_file)


	def conv_layer(self,idx,inputs,filters,size,stride):
		channels = inputs.get_shape()[3]
		weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
		biases = tf.Variable(tf.constant(0.1, shape=[filters]))
		conv = tf.nn.conv2d(inputs, weight, strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_conv')	
		conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')	
		print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels))
		return tf.nn.relu(conv_biased,name=str(idx)+'_relu')

	def pooling_layer(self,idx,inputs,size,stride):
		print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride)
		return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

	def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
		input_shape = inputs.get_shape().as_list()		
		if flat:
			dim = input_shape[1]*input_shape[2]*input_shape[3]
			inputs_transposed = tf.transpose(inputs,(0,3,1,2))
			inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
		else:
			dim = input_shape[1]
			inputs_processed = inputs
		weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
		biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))	
		print '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))	
		if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
		ip = tf.add(tf.matmul(inputs_processed,weight),biases)
		return tf.nn.relu(ip,name=str(idx)+'_relu')

	def pre_proc(self,img):
		img_proc = img.astype('float32')
		img_proc = cv2.resize(img_proc,(self.WH,self.WH))
		#img_proc = cv2.cvtColor(img_proc,cv2.COLOR_BGR2GRAY)
		img_proc = img_proc / 255.0

		return img_proc

		

	def train(self):
		#print list(os.walk(self.IMG_PATH))


		self.labels = np.genfromtxt(self.LABEL_FILE,delimiter=',')[:,1:]
		self.labels_test = np.genfromtxt(self.LABEL_FILE,delimiter=',')[:,1:]

		self.TRAIN_SIZE = self.labels.shape[0] - self.TEST_SIZE

		self.shuffled_arange = np.arange(self.labels.shape[0])
		np.random.shuffle(self.shuffled_arange)

		print '#Calculated training set size : ' + str(self.TRAIN_SIZE)
		print '#Calculated test set size : ' + str(self.TEST_SIZE)


		start_time = time.time()

		logf_train = open(self.LOG_TRAIN,'w')
		logf_test = open(self.LOG_TEST,'w')

		print '#Training'

		img = np.zeros((self.BATCH_SIZE,self.WH,self.WH,self.CHANNEL))
		label = np.zeros((self.BATCH_SIZE,self.LABEL_DIM))
		
		for step in range(self.TRAIN_SIZE*self.NUM_EPOCHS // self.BATCH_SIZE):
			for i in range(self.BATCH_SIZE):			
				idx = np.random.randint(0,self.TRAIN_SIZE)
				img[i,:,:,:] = self.pre_proc(cv2.imread(self.IMG_PATH + str(self.shuffled_arange[idx]) +'_RGB.jpeg'))
				label[i] = self.labels[self.shuffled_arange[idx]]
			img_disp = img[0,:,:,:]*255.0
			img_disp = cv2.resize(img_disp.astype('uint8'),(244,244))
			cv2.imshow('sample',img_disp)
			cv2.waitKey(1)
			feed_dict = {self.x: img,
				self.y_: label,
				self.keep_prob:0.5
			}
			
			# Run the graph and fetch some of the nodes.
			_ = self.sess.run(self.optimizer,feed_dict=feed_dict)
			
			if step % self.EVAL_FREQUENCY == 0:
				#print batch_label[0]
				l, predictions = self.sess.run(
					[self.cross_entropy, self.y],feed_dict=feed_dict)
				elapsed_time = time.time() - start_time
				print('  -Step %d (epoch %.2f), %.1f s' %
				(step, float(step) * self.BATCH_SIZE / self.TRAIN_SIZE,
				elapsed_time))
				print('    Training Minibatch loss: %.7f' % l)
				accu = 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(label, 1)) / predictions.shape[0]
				print('    Training Minibatch accuracy: %.3f%% \n' % accu)
				logf_train.write(str(step) + ',' + str(l) + ',' + str(accu) + '\n')
				sys.stdout.flush()	
			
			if step % self.SAVE_FREQUENCY == 0:
				print('  -Save checkpoint!\n')
				self.saver.save(self.sess, self.CKPT_NAME_PREFIX + str(step) + '.ckpt')
			
			if step % self.TEST_FREQUENCY == 0:
				print('  -Evaluation on test set')
				tt = 0
				correct_test = 0
				confu_mat = np.zeros((self.LABEL_DIM,self.LABEL_DIM),dtype='int') #h : predict, v : actual
				for ii in range(self.TEST_SIZE//self.BATCH_SIZE):
					img_test = np.zeros((self.BATCH_SIZE,self.WH,self.WH,self.CHANNEL))
					label_test = np.zeros((self.BATCH_SIZE,self.LABEL_DIM))	
					for jj in range(self.BATCH_SIZE):
						idx = self.TRAIN_SIZE + ii*self.BATCH_SIZE + jj
						#print cv2.imread(self.IMG_PATH_TEST + str(idx) +'.jpeg').shape
						img_test[jj,:,:,:] = cv2.resize(cv2.imread(self.IMG_PATH_TEST + str(self.shuffled_arange[idx]) +'_RGB.jpeg'),(self.WH,self.WH))/255.0
						label_test[jj] = self.labels_test[self.shuffled_arange[idx]]
				
					feed_dict_test = {self.x: img_test,
						self.y_: label_test,
						self.keep_prob:1.0
					}

					l_test, predictions_test = self.sess.run([self.cross_entropy, self.y],feed_dict=feed_dict_test)
					label_idxs = np.argmax(label_test, 1)
					predictions_test_idxs = np.argmax(predictions_test, 1)
					for kk in range(self.BATCH_SIZE):
						confu_mat[label_idxs[kk],predictions_test_idxs[kk]] += 1
					correct_test += (np.sum(np.argmax(predictions_test, 1) == np.argmax(label_test, 1)))

				elapsed_time = time.time() - start_time
				accu_test = 100.0*float(correct_test)/float(self.TEST_SIZE)				
				print('    Test accuracy: %.3f%%' % accu_test)
				print('    Confusion mat :')
				print confu_mat
				print ''
				log_confu = open(self.LOG_CONFU,'w')
				for confu_row in range(self.LABEL_DIM):				
					confu_mat[confu_row,:].tofile(log_confu,',')
					log_confu.write('\n')
				log_confu.close()
				logf_test.write(str(step) + ',' + str(l_test) + ',' + str(accu_test) + '\n')
				
				sys.stdout.flush()
			

		self.saver.save(self.sess, self.CKPT_NAME_PREFIX + 'final.ckpt')


	def ros_run(self):
		print '#Start ROS running'
		self.proc = proc_input.proc_input('ori_esti',True,False,False,True)
		self.sub_ = rospy.Subscriber("/AUPAIR/detection_target_without_ori", detection_array, self.callback_detections)
		self.pub_ = rospy.Publisher('/AUPAIR/detection_target_with_ori',detection_array,queue_size=10)
		self.detection = []
		time.sleep(3)
		s = time.time()
		while time.time() - s < self.ROS_TIMEOUT:
			try:
				if not len(self.detection) > 0 : continue
				ss = time.time()
				#if abs(float(ss) - self.detection[6]) > 2.0 : 
				#	print 'no new message in 2 sec..'
				#	continue

				img = self.proc.get_rgb_by_time(self.detection[6])[1]
				depth = self.proc.get_depth_by_time(self.detection[6])[1]
				depth_clip = np.clip(depth,0,self.MAX_DISTANCE)
				depth_norm = (depth_clip.astype('float') * 255.0 / float(self.MAX_DISTANCE)).astype('uint8')
				if img == None or depth == None : continue
				x1 = max(0,int(self.detection[1]))
				y1 = max(0,int(self.detection[2]))
				x2 = min(640,int(self.detection[3]))
				y2 = min(480,int(self.detection[4]))
	
				w = (y2-y1)
				h = (x2-x1)

				img_resize = cv2.resize(img[y1:y1+w,x1:x1+h], (self.WH, self.WH))	
				depth_resize = cv2.resize(depth[y1:y1+w,x1:x1+h], (self.WH, self.WH))	
				network_input = np.zeros((1,self.WH,self.WH,self.CHANNEL))
				network_input[0,:,:,0] = cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY)/255.0
				#network_input[0,:,:,1] = depth_resize/255.0
				cv2.imshow('ori_proc',img_resize)
				cv2.waitKey(2)
				feed_dict = {self.x: network_input,
					self.keep_prob:1.0
				}
				ori_class = np.argmax(self.sess.run(self.y,feed_dict=feed_dict))
				ori = ori_class * 45.0
				det_arr_msg = detection_array()
				det_msg = detection('target',x1,y1,x2,y2,ori)
				det_arr_msg.detections.append(det_msg)
				timestamp_msg = rospy.Time.from_sec(self.detection[6])
				det_arr_msg.header.stamp = timestamp_msg
				print det_msg
				self.pub_.publish(det_arr_msg)
				#time.sleep(1)
			except:
				continue

	def callback_detections(self,data):
		self.detection = [data.detections[-1].classes,data.detections[-1].x1,data.detections[-1].y1,data.detections[-1].x2,data.detections[-1].y2,data.detections[-1].prob,data.header.stamp.to_sec()]		


def terminate(what,ever):
	raise KeyboardInterrupt("CTRL-C!")

def main():
	signal.signal(signal.SIGINT, terminate)
	if MODE == 'training':
		oe = ori_esti(CKPT)
		oe.train()
	elif MODE == 'test':
		oe = ori_esti(CKPT)

	elif MODE == 'run':
		oe = ori_esti(CKPT)
		oe.ros_run()

	else : 
		print 'argument error!!!'
	

	

if __name__ == '__main__':
	main()
