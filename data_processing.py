import tensorflow as tf
import os
from PIL import Image
import numpy as np
VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
result=[]
annotations=[]
path='image_2/'
size=224,224
for root,dirs,files in os.walk(path):
    for f in files:
        name=f.replace(".png","")
        im=Image.open('image_2/'+name+'.png')
        with open('label_2/'+name+'.txt') as f:
            for line in f.readlines():
                data = line.split(' ')
                if(data[0]  in VEHICLES and float(data[1])<0.1 and float(data[2])<0.1):
                    xmin = int(round(float(data[4]))) 
                    ymin = int(round(float(data[5])))
                    xmax = int(round(float(data[6])))
                    ymax = int(round(float(data[7])))
                    
                    index=VEHICLES.index(data[0])
                    height=float(data[8])
                    width=float(data[9])
                    length=float(data[10])
                    rect=im.crop((xmin,ymin,xmax,ymax))
                    rect=rect.resize(size,Image.ANTIALIAS)
                    rect=np.array(rect).reshape([1,-1])
                    result.append(rect)
                    annotations.append([index,height,width,length])
        print(name)
shape=np.array(result).shape
np.save("data",np.array(result).reshape(shape[0],shape[2]))
np.save("annotations",np.array(annotations))