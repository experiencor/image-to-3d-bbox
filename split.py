import numpy as np
from random import shuffle
data=np.load("data.npy")
annotations=np.load("annotations.npy")
ran=list(range(data.shape[0]))
shuffle(ran)
data=data[ran]
annotations=annotations[ran]
datafortest=data[0:1000]
np.save("datafortest",datafortest)
annotationsfortest=annotations[0:1000]
np.save("annotationsfortest",annotationsfortest)
datafortrain=data[1000:data.shape[0]]
np.save("datafortrain",datafortrain)
annotationsfortrain=annotations[1000:data.shape[0]]
np.save("annotationsfortrain",annotationsfortrain)
