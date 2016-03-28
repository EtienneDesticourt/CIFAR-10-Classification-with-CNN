import numpy as np
from FC import FullyConnectedLayer as fc
from CNN import ConvolutionalLayer as cnn

IMAGE_SHAPE  = (32,32,3)
NUM_KERNELS  = 12
SIZE_KERNELS = 3
NUM_LABELS   = 10


def relu(array):
    return np.maximum(array, 0)

def sigmoid(array):
    pass

def run():  
    cnnLayer = cnn(IMAGE_SHAPE, NUM_KERNELS, SIZE_KERNELS)
    fcLayer = fc(NUM_LABELS)

    #Load image
    inputLayer = None
    #Predict with network
    cnnOutput = cnnLayer.forwardProp(inputLayer)
    fcInput = relu(sigmoid(cnnOutput))
    fcOutput = fcLayer.forwardProp(fcInput)
    prediction = sigmoid(fcOutput)
    #Update weights
    
    
    
