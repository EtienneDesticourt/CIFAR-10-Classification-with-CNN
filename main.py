import numpy as np
from FC import FullyConnectedLayer as fc
from CNN import ConvolutionalLayer as cnn
from Network import Network
import random
IMAGE_SHAPE  = (32,32,3)
NUM_KERNELS  = 12
SIZE_KERNELS = 3
NUM_LABELS   = 10


def relu(array):
    return np.maximum(array, 0)

def sigmoid(array):
    return 1.0 / (np.exp(-array) + 1)

def run():
    #SIMPLE TEST
    NUM_EPOCH = 100
    inputArray = []
    labels = []
    for i in range(10):
        a = random.randrange(0,2)
        b = random.randrange(0,2)
        c = random.randrange(0,2)
        inputArray.append([a,b])
        labels.append(a+2*b)
    
    labels = np.array(labels)
    inputArray = np.array(inputArray)



    
    N = Network(sigmoid, 0.0001)
    N.addLayer(2)
    N.addLayer(1)
    N.fit(inputArray, labels, NUM_EPOCH)
    return N
    
N = run()    
N.predict([1,2])
