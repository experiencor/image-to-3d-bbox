import numpy as np
import scipy.interpolate as ip
import os
from PIL import Image

sBaseDir="/data/KITTI/training/"

sCalibDir=os.path.join(sBaseDir,"calib")
sImageDir=os.path.join(sBaseDir,"image_2")
sLabelDir=os.path.join(sBaseDir,"label_2")

lInterest=['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']

sSet=set()

lImage=[]
lType=[]
lAngle=[]

for sFileName in os.listdir(sLabelDir):
	sTxtName=sFileName
	sImgName=sFileName.replace(".txt",".png")
	with Image.open(os.path.join(sImageDir,sImgName)) as iImage:
		with open(os.path.join(sLabelDir,sTxtName),'r') as fLabel:
			print(sTxtName)
			for sLabel in fLabel.readlines():
				dLabel=sLabel.split(' ')
				dLabel[1:]=[float(x) for x in dLabel[1:]]
				if dLabel[0] in lInterest and np.abs(float(dLabel[1]))<0.1 and np.abs(float(dLabel[2]))<0.1:
					lImage.append(np.array(iImage.crop((int(round(float(dLabel[4]))),int(round(float(dLabel[5]))),int(round(float(dLabel[6]))),int(round(float(dLabel[7]))))).resize((224,224))))
					lType.append(lInterest.index(dLabel[0]))
					lAngle.append(dLabel[14])

aImage=np.array(lImage)
aType=np.array(lType)
aAngle=np.array(lAngle)
print("aImage:",aImage.shape)
print("aType:",aType.shape)
print("aAngle:",aAngle.shape)
np.savez("../data/data.npz",aImage=aImage,aType=aType,aAngle=aAngle)

