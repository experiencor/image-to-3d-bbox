import tensorflow as tf
import numpy as np
import vgg
import os

def LeakyReLU(x,alpha):
	return tf.nn.relu(x)-tf.nn.relu(-alpha*x)

slim = tf.contrib.slim

#LearningRate=0.0001
LearningRate=1e-4

BatchSize=100

BinCount=6
Overlap=0.1
AngleOverlap=1

Bin,sStep=np.linspace(-np.pi,np.pi,BinCount,endpoint=False,retstep=True)
Bin+=sStep/2
Width=sStep*(Overlap+1)
AngleWidth=sStep*(AngleOverlap+1)

sCheckPoint="../model/model.ckpt"
sCheckPointRestore="../model/checkpoint"

with np.load('../data/split.npz') as dData:
	aImage_train=dData['aImage_train']
	aType_train=dData['aType_train']
	aAngle_train=dData['aAngle_train']
	aImage_test=dData['aImage_test']
	aType_test=dData['aType_test']
	aAngle_test=dData['aAngle_test']

inputs=tf.placeholder(dtype=tf.dtypes.float32,shape=(None,224,224,3))
label=tf.placeholder(dtype=tf.dtypes.float32,shape=(None))
keep_prob=tf.placeholder(tf.float32)

scope='vgg_16'

with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
	end_points_collection = sc.original_name_scope + '_end_points'
	# Collect outputs for conv2d, fully_connected and max_pool2d.
	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
		outputs_collections=end_points_collection):
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

conv5=tf.contrib.layers.flatten(net)
orientation=slim.fully_connected(conv5,256,activation_fn=None,scope='fc7_o')
orientation=LeakyReLU(orientation,0.1)
orientation=slim.dropout(orientation,keep_prob,scope='dropout7_o')
orientation=slim.fully_connected(orientation,BinCount*2,activation_fn=None,scope='fc8_o')
orientation=tf.reshape(orientation,[-1,BinCount,2])
orientation=tf.nn.l2_normalize(orientation,dim=2)

confidence=slim.fully_connected(conv5,256,activation_fn=None,scope='fc7_c')
confidence=LeakyReLU(confidence,0.1)
confidence=slim.dropout(confidence,keep_prob,scope='dropout7_c')
confidence=slim.fully_connected(confidence,BinCount,activation_fn=None,scope='fc8_c')
confidence_out=tf.nn.softmax(confidence)

theta_label=tf.reshape(label,[-1,1])-np.reshape(Bin,[1,BinCount])
orientation_label=tf.concat([tf.reshape(tf.math.cos(theta_label),[-1,BinCount,1]),tf.reshape(tf.math.sin(theta_label),[-1,BinCount,1])],axis=2)
confidence_label_org=tf.to_float(orientation_label[:,:,0]>np.cos(Width))
confidence_label_lin=(Width-np.abs(np.mod(theta_label+np.pi,2*np.pi)-np.pi))*confidence_label_org
confidence_count=tf.reduce_sum(confidence_label_lin,axis=1)
confidence_label=confidence_label_lin/tf.reshape(confidence_count,[-1,1])

angle_filter_org=tf.to_float(orientation_label[:,:,0]>np.cos(AngleWidth))
angle_filter=(AngleWidth-np.abs(np.mod(theta_label+np.pi,2*np.pi)-np.pi))*angle_filter_org

Lconf=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=confidence_label,logits=confidence))
Lloc=-tf.reduce_mean(tf.reduce_sum((orientation_label[:,:,0]*orientation[:,:,0]+orientation_label[:,:,1]*orientation[:,:,1]-1)*angle_filter,axis=1))

Loss=Lconf+Lloc*8

optimizer=tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

config=tf.ConfigProto(log_device_placement=True)

session=tf.Session(config=config)
saver=tf.train.Saver()

if os.path.isfile(sCheckPointRestore):
	saver.restore(session,sCheckPoint)
	print("Session Restored")
else:
	ckpt_list=tf.contrib.framework.list_variables('../data/vgg_16.ckpt')[1:-7]
	variables_to_restore = slim.get_variables()[:26] ## vgg16-conv5
	for name in range(1,len(ckpt_list),2):
		tf.contrib.framework.init_from_checkpoint('../data/vgg_16.ckpt',{ckpt_list[name-1][0]:variables_to_restore[name]})
		tf.contrib.framework.init_from_checkpoint('../data/vgg_16.ckpt',{ckpt_list[name][0]:variables_to_restore[name-1]})
	init=tf.global_variables_initializer()
	session.run(init)
	print("Session Initialized")

EPOCH=0
iLength_train=aImage_train.shape[0]

order=np.linspace(0,iLength_train,iLength_train,dtype=int,endpoint=False)

try:
	while True:
		rLossMean=0
		rLosslocMean=0
		rLossconfMean=0
		np.random.shuffle(order)
		for Start in range(0,aAngle_train.shape[0],BatchSize):
			End=Start+BatchSize
			if End>aAngle_train.shape[0]:
				End=aAngle_train.shape[0]
			InputData=aImage_train[order[Start:End]]
			LabelData=aAngle_train[order[Start:End]]
			lconf,lloc,rLoss,_=session.run([Lconf,Lloc,Loss,optimizer],feed_dict={inputs:InputData,label:LabelData,keep_prob:0.5})
			rLossMean+=rLoss*(End-Start)
			rLosslocMean+=lloc*(End-Start)
			rLossconfMean+=lconf*(End-Start)
			print("Epoch %d Range %d %d Loss: %f %f %f"%(EPOCH,Start,End,rLoss,lconf,lloc))
		rLossMean/=aAngle_train.shape[0]
		rLosslocMean/=aAngle_train.shape[0]
		rLossconfMean/=aAngle_train.shape[0]
		print("Epoch %d Finish"%(EPOCH))
		print("MeanLoss: %f %f %f"%(rLossMean,rLossconfMean,rLosslocMean))

		if EPOCH%10==0:
			AngleLoss=0
			IndexLoss=0
			ConfLoss=0
			OS=0

			for Start in range(0,aAngle_train.shape[0],BatchSize):
				End=Start+BatchSize
				if End>aAngle_train.shape[0]:
					End=aAngle_train.shape[0]
				InputData=aImage_train[Start:End]
				LabelData=aAngle_train[Start:End]

				#confo,conflo,orient=session.run([confidence_label,confidence_label,orientation_label],feed_dict={inputs:InputData,label:LabelData,keep_prob:1})
				confo,conflo,orient=session.run([confidence_out,confidence_label_org,orientation],feed_dict={inputs:InputData,label:LabelData,keep_prob:1})

				Angleloc=np.arctan2(orient[:,:,1],orient[:,:,0])
				index=np.argmax(confo,axis=1)
				Angle=Bin[index]+Angleloc[list(range(index.shape[0])),index]
				angleloss=np.mod(Angle-LabelData+np.pi,np.pi*2)-np.pi
				AngleLoss+=np.sum(angleloss**2)
				OS+=np.sum((1+np.cos(angleloss))/2)

				Index=np.argmin(np.abs(np.mod(np.reshape(LabelData,[-1,1])-np.reshape(Bin,[1,-1])+np.pi,np.pi*2)-np.pi),axis=1)
				indexloss=np.mod(index-Index+BinCount//2,BinCount)-BinCount//2

				IndexLoss+=np.sum(np.abs(indexloss))

				confloss=np.sum((confo-conflo)**2)
				ConfLoss+=confloss

			AngleLoss=np.sqrt(AngleLoss/aAngle_train.shape[0])
			IndexLoss/=aAngle_train.shape[0]
			ConfLoss=np.sqrt(ConfLoss/aAngle_train.shape[0])
			OS/=aAngle_train.shape[0]
			print("Test on train:\n\tAngleLoss:%f %f\n\tIndexLoss:%f\n\tConfLoss:%f\n\tOS:%f"%(AngleLoss,AngleLoss*180/np.pi,IndexLoss,ConfLoss,OS))




			AngleLoss=0
			IndexLoss=0
			ConfLoss=0
			OS=0

			for Start in range(0,aAngle_test.shape[0],BatchSize):
				End=Start+BatchSize
				if End>aAngle_test.shape[0]:
					End=aAngle_test.shape[0]
				InputData=aImage_test[Start:End]
				LabelData=aAngle_test[Start:End]

				#confo,conflo,orient=session.run([confidence_label,confidence_label,orientation_label],feed_dict={inputs:InputData,label:LabelData,keep_prob:1})
				confo,conflo,orient=session.run([confidence_out,confidence_label_org,orientation],feed_dict={inputs:InputData,label:LabelData,keep_prob:1})

				Angleloc=np.arctan2(orient[:,:,1],orient[:,:,0])
				index=np.argmax(confo,axis=1)
				Angle=Bin[index]+Angleloc[list(range(index.shape[0])),index]
				angleloss=np.mod(Angle-LabelData+np.pi,np.pi*2)-np.pi
				AngleLoss+=np.sum(angleloss**2)
				OS+=np.sum((1+np.cos(angleloss))/2)

				Index=np.argmin(np.abs(np.mod(np.reshape(LabelData,[-1,1])-np.reshape(Bin,[1,-1])+np.pi,np.pi*2)-np.pi),axis=1)
				indexloss=np.mod(index-Index+BinCount//2,BinCount)-BinCount//2

				IndexLoss+=np.sum(np.abs(indexloss))

				confloss=np.sum((confo-conflo)**2)
				ConfLoss+=confloss

			AngleLoss=np.sqrt(AngleLoss/aAngle_test.shape[0])
			IndexLoss/=aAngle_test.shape[0]
			ConfLoss=np.sqrt(ConfLoss/aAngle_test.shape[0])
			OS/=aAngle_test.shape[0]
			print("Test on test:\n\tAngleLoss:%f %f\n\tIndexLoss:%f\n\tConfLoss:%f\n\tOS:%f"%(AngleLoss,AngleLoss*180/np.pi,IndexLoss,ConfLoss,OS))

		EPOCH+=1
except KeyboardInterrupt:
	saver.save(session,sCheckPoint)
	print("Session Saved")
