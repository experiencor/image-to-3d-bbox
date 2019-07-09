import numpy as np

with np.load('../data/data.npz') as dData:
	aImage=dData['aImage']
	aType=dData['aType']
	aAngle=dData['aAngle']

iLength=aImage.shape[0]

iTestCount=iLength//10

aList=np.linspace(0,iLength,iLength,dtype=int,endpoint=False)
np.random.shuffle(aList)
aImage_train=aImage[aList[iTestCount:]]
aType_train=aType[aList[iTestCount:]]
aAngle_train=aAngle[aList[iTestCount:]]

aImage_test=aImage[aList[:iTestCount]]
aType_test=aType[aList[:iTestCount]]
aAngle_test=aAngle[aList[:iTestCount]]

np.savez('../data/split.npz',
	aImage_train=aImage_train,aType_train=aType_train,aAngle_train=aAngle_train,
	aImage_test=aImage_test,aType_test=aType_test,aAngle_test=aAngle_test)
