import numpy as np
from FC import FullyConnectedLayer as fc
from CNN import ConvolutionalLayer as cnn
from Network import Network
import random
import matplotlib.pyplot as plt
import matplotlib
import pickle


def relu(array):
    return np.maximum(array, 0)

def sigmoid(array):
    return 1.0 / (np.exp(-array) + 1)

def runTest1():
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
    #N.addLayer(7, biased=True)
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

def runTest2():
    inputArray = np.array([[[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
                            [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
                            [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
                            [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
                            [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
                            [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
                            [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
                            [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
                            [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]]])
    labels = [1]
    #print(inputArray.shape)

    #Conv layer config
    numKernels = 4
    kernelSize = 3

    #Create network
    N = Network(sigmoid, ALPHA)
    N.addLayer(inputArray[0].shape)
    N.addConvLayer(inputArray[0].shape, numKernels, kernelSize, flatten=True) #We flatten it if it's gonna be followed by a FC layer
    N.addLayer(10, biased=True)
    N.addLayer(1)
##    return inputArray
##    return N
    #Train network    
    N.fit(inputArray, labels, NUM_EPOCH, verbose=True)

    return N

def unpickle(file):
    fo = open(file, 'rb')
    dic = pickle.load(fo, encoding='latin1')
    fo.close()
    return dic

def shapeAsImage(inputs):    
    inputs = np.dstack((inputs[:,:1024], inputs[:,1024:2048], inputs[:,2048:]))
    return np.reshape(inputs, (10000, 32, 32, 3))

def oneHotEncode(labels):
    labels = np.array(labels)
    maxLabel = max(labels)
    oneHotEncoded = np.zeros((labels.size, maxLabel+1))
    oneHotEncoded[np.arange(labels.size), labels] = 1
    return oneHotEncoded


PATH = r"E:\Users\Etienne2\Downloads\cifar-10-python\cifar-10-batches-py\data_batch_1"
IMAGE_SHAPE  = (32,32,3)
NUM_KERNELS  = 12
SIZE_KERNELS = 6
NUM_LABELS   = 10

NUM_EPOCH = 100
ALPHA = 0.5

def run():
    #Unpickle data and put it in usable shape
    data = unpickle(PATH)
    inputs = shapeAsImage(data['data'])
    labels = oneHotEncode(data['labels'])
    print(labels[0])
    #Check what it looks like
    plt.imshow(inputs[0], interpolation='nearest')
    plt.show()

    #Create classification network
    N = Network(sigmoid, ALPHA)
    N.addLayer(IMAGE_SHAPE)
    N.addConvLayer(IMAGE_SHAPE, NUM_KERNELS, SIZE_KERNELS, flatten=True)
    N.addLayer(NUM_LABELS)


    import cProfile

    fit = N.fit

    #Train network
    command = 'N.fit(inputs[:10], labels[:10], NUM_EPOCH, verbose=True)'
    cProfile.runctx(command, globals(), locals(), filename=None)
    #cProfile.run('fit(inputs[:10], labels[:10], NUM_EPOCH, verbose=True)')

    return N
    



    

N = run()
