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


def relu(array):
    return np.maximum(array, 0)

def sigmoid(array):
    return 1.0 / (np.exp(-array) + 1)

def run():
    #SIMPLE TEST
    NUM_EPOCH = 50
    inputArray = []
    labels = []
    for x in range(50):
        for y in range(50):
            if x**2+y**2 > 1000 and x**2+y**2<2000: labels.append(1)
            else: labels.append(0)
                
            inputArray.append([x,y])


    labels = np.array(labels)
    inputArray = np.array(inputArray)
    print(inputArray[0,:].shape)
    colors = ['red', 'blue']
    fig = plt.figure()
    plt.scatter(inputArray[:,0], inputArray[:,1], c=labels)
    plt.show()

    

    
    N = Network(sigmoid, 0.01)
    N.addLayer(2, biased=True)
    N.addLayer(3, biased=True)
    N.addLayer(1)
    
##    N.layers[1].params = np.array([[0.2,0.4],
##                                [0.4,0.5],
##                                [0.1,0.3]])
##    N.layers[2].params = np.array([[0.1, 0.2, 0.3]])
##    print(N.layers[1].params)
##    print(N.layers[2].params)
    N.fit(inputArray, labels, NUM_EPOCH, verbose=True)
    newLabels = []
    for inputs in inputArray:
        label = N.predict(inputs)[0]
        
        newLabels.append(label)
    fig = plt.figure()
    plt.scatter(inputArray[:,0], inputArray[:,1], c=newLabels)
    plt.show()
    return inputArray, N

    
N = run()    
