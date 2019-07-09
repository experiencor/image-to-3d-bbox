import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2,os
import numpy as np
import time
from random import shuffle
import sys
import argparse
width,height=224,224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
def compute_average(file):
	annotations=np.load(file)
	print(annotations.shape)
	aver1=np.zeros([1,3])
	sum1=0
	aver2=np.zeros([1,3])
	sum2=0
	aver3=np.zeros([1,3])
	sum3=0
	aver4=np.zeros([1,3])
	sum4=0
	aver5=np.zeros([1,3])
	sum5=0
	aver6=np.zeros([1,3])
	sum6=0
	for i in range(annotations.shape[0]):
		if(annotations[i,0]==0):
			aver1=aver1+annotations[i,1:4]
			sum1=sum1+1
		if(annotations[i,0]==1):
			aver2=aver2+annotations[i,1:4]
			sum2=sum2+1
		if(annotations[i,0]==2):
			aver3=aver3+annotations[i,1:4]
			sum3=sum3+1
		if(annotations[i,0]==3):
			aver4=aver4+annotations[i,1:4]
			sum4=sum4+1
		if(annotations[i,0]==4):
			aver5=aver5+annotations[i,1:4]
			sum5=sum5+1
		if(annotations[i,0]==5):
			aver6=aver6+annotations[i,1:4]
			sum6=sum6+1
	aver1=aver1/sum1
	aver2=aver2/sum2
	aver3=aver3/sum3
	aver4=aver4/sum4
	aver5=aver5/sum5
	aver6=aver6/sum6
	dims_avg = [aver1, aver2, aver3, aver4, aver5,  aver6]
	result=[]
	for i in range(annotations.shape[0]):
		if(annotations[i,0]==0):
			result.append(annotations[i,1:4]-aver1)
		if(annotations[i,0]==1):
			result.append(annotations[i,1:4]-aver2)
		if(annotations[i,0]==2):
			result.append(annotations[i,1:4]-aver3)
		if(annotations[i,0]==3):
			result.append(annotations[i,1:4]-aver4)
		if(annotations[i,0]==4):
			result.append(annotations[i,1:4]-aver5)
		if(annotations[i,0]==5):
			result.append(annotations[i,1:4]-aver6)
		shape=np.array(result).shape
	return dims_avg,np.array(result).reshape(shape[0],shape[2])
def parse_args():
	parser = argparse.ArgumentParser(description='3D bounding box')
	parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'test')
	parser.add_argument('--image',dest = 'image',help='Image path')
	parser.add_argument('--label',dest = 'label',help='Label path')
	parser.add_argument('--box2d',dest = 'box2d',help='2D detection path')
	parser.add_argument('--output',dest = 'output',help='Output path', default = './validation/result_2/')
	parser.add_argument('--model',dest = 'model')
	parser.add_argument('--gpu',dest = 'gpu',default= '0')
	args = parser.parse_args()
	return args

def LeakyReLU(x, alpha):
	return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def build_model(inputs,d_label,b):
	learning_rate = 0.0001
	with slim.arg_scope([slim.conv2d,slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.001),
						weights_regularizer=slim.l2_regularizer(0.0005)):
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		net = slim.max_pool2d(net, [2, 2], scope='pool3')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		net = slim.max_pool2d(net, [2, 2], scope='pool5')
		conv5 = tf.contrib.layers.flatten(net)
		dimension = slim.fully_connected(conv5, 512, activation_fn=None, scope='fc7_d')
		dimension = LeakyReLU(dimension, 0.1)
		dimension = slim.dropout(dimension, b, scope='dropout7_d')
		dimension = slim.fully_connected(dimension, 3, activation_fn=None, scope='fc8_d')
		loss_d = tf.losses.mean_squared_error(d_label, dimension)
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_d)
		return dimension,loss_d,train_step
def test(model, image_dir,label_dir):
	inputs=tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
	d_label = tf.placeholder(tf.float32, shape = [None, 3])
	b=tf.placeholder(tf.float32,shape=[])
	dimension,loss_d,train_step=build_model(inputs,d_label,b)
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth =True
	sess = tf.Session(config=tfconfig)
	init = tf.global_variables_initializer()
	sess.run(init)
	saver = tf.train.Saver()
	saver.restore(sess, model)
	dims_avg,result= compute_average(label_dir)
	data=np.load(image_dir)
	annotations=np.load(label_dir)
	loss=0
	for i in range(0,data.shape[0],10):
		prediction = sess.run(dimension, feed_dict={inputs: data[i:i+10].reshape([-1,224,224,3]),b:1})
		loss=loss+np.sum(np.square(prediction-result[i:i+10,:]))
		for j in range(0,10):
			print("annotations",annotations[i+j])
			prediction[j]=prediction[j]+dims_avg[int(annotations[i+j,0])]
			print("prediction",prediction[j])
	loss=loss/data.shape[0]/3
	print("testloss:"+str(loss))
def train():
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True
	sess = tf.Session(config=tfconfig)
	BATCH_SIZE = 24
	epochs = 10
	save_path = './model/'
	dims_avg,result= compute_average('annotationsfortrain.npy')
	data=np.load('datafortrain.npy')
	inputs=tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
	d_label = tf.placeholder(tf.float32, shape = [None, 3])
	b=tf.placeholder(tf.float32,shape=[])
	dimension,loss_d,train_step=build_model(inputs,d_label,b)
	#Load pretrain VGG model
	saver = tf.train.Saver(max_to_keep=100)
	if(os.path.exists('./model/checkpoint')):
		saver.restore(sess,save_path+"model.ckpt")
		print("Restored")
		#saver.restore(sess,save_path+"checkpoint")
	else:
		ckpt_list = tf.contrib.framework.list_variables('./vgg_16.ckpt')[1:-7]
		variables_to_restore = slim.get_variables()[:26]
		for name in range(1,len(ckpt_list),2):
			tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name-1][0]: variables_to_restore[name]})
			tf.contrib.framework.init_from_checkpoint('./vgg_16.ckpt', {ckpt_list[name][0]: variables_to_restore[name-1]})
		sess.run(tf.global_variables_initializer())
	try:
		for epoch in range(epochs):
			tStart_epoch = time.time()
			ran=list(range(data.shape[0]))
			shuffle(ran)
			data=data[ran]
			result=result[ran]
			epoch_loss=[]
			for i in range(0,data.shape[0],BATCH_SIZE):
				if(i+BATCH_SIZE<data.shape[0]):
					_,loss,dimen=sess.run([train_step,loss_d,dimension],feed_dict={inputs: data[i:i+BATCH_SIZE,:].reshape(BATCH_SIZE,224,224,3), d_label:result[i:i+BATCH_SIZE,:],b:0.5})
					print(str(i)+':'+str(loss))
					epoch_loss.append(loss)
			if (epoch+1) % 5 == 0:
				saver.save(sess,save_path+"model.ckpt")
			# Print some information
			print("Epoch:", epoch+1, " done. Loss:", np.mean(np.array(epoch_loss)))
			tStop_epoch = time.time()
			print ("Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s")
			sys.stdout.flush()
	except KeyboardInterrupt:
		saver.save(sess,save_path+"model.ckpt")
		print("saved")


if __name__=="__main__":
	train()
	#test('./model/model.ckpt','datafortest.npy','annotationsfortest.npy')