import numpy as np
from FC import FullyConnectedLayer as fc
from CNN import ConvolutionalLayer as cnn
from Network import Network
import random
import matplotlib.pyplot as plt
import matplotlib
IMAGE_SHAPE  = (32,32,3)
NUM_KERNELS  = 12
SIZE_KERNELS = 3
NUM_LABELS   = 10
NUM_EPOCH = 1000
ALPHA = 0.5
def relu(array):
    return np.maximum(array, 0)

def sigmoid(array):
    return 1.0 / (np.exp(-array) + 1)

def run():
    #SIMPLE TEST
    
    #Gen training data
    inputArray = []
    labels = []
    
    for i in range(8):
        for j in range(8):
            t = i**2+j**2
##            print(i,j,t)
            if  (i-2.5)**2 + (j-2.5)**2 < 3: y = 1
##            if j >= i and j < i + 3: y = 1
            else: y = 0
            inputArray.append([i,j])
            labels.append(y)

    labels = np.array(labels)
    inputArray = np.array(inputArray)

    #Plot training data
    fig = plt.figure()
    plt.scatter(inputArray[:,0], inputArray[:,1], c=labels, s=500)
    plt.show()

    

    #Create network
    N = Network(sigmoid, ALPHA)
    N.addLayer(2, biased=True)
    N.addLayer(7, biased=True)
    N.addLayer(1)

    #Train network    
    N.fit(inputArray, labels, NUM_EPOCH, verbose=True)


    #Plot predictions of training set
    newLabels = []
    for inputs in inputArray:
        label = N.predict(inputs)[0]        
        newLabels.append(label)        
    fig = plt.figure()
    plt.scatter(inputArray[:,0], inputArray[:,1], c=newLabels, s =500)
    plt.show()
    
    return N

    
N = run()    
