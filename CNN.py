import numpy as np
import functools, operator
from Layer import Layer
from scipy.signal import convolve2d

class ConvolutionalLayer(Layer):
    def __init__(self, inputShape, numKernels, kernelSize, flatten=False):        
        self.numKernels = numKernels
        self.kernelSize = kernelSize
        self.biased     = False
        #Initialize weights
        x, y, depth     = inputShape
        self.inputDepth = depth
        self.params     = np.random.uniform(-1.0, 1.0, size=(kernelSize, kernelSize, depth, numKernels))
        #Calculate output shape
        self.outputShape = ( x - kernelSize + 1,
                             y - kernelSize + 1,
                             numKernels)
        if flatten: self.shape = (functools.reduce(operator.mul, self.outputShape, 1),)
        else:       self.shape = self.outputShape
        self.flatten = flatten
        
    def calcOutput(self, inputs):
        output = np.zeros(self.outputShape)
        for d in range(self.inputDepth):
            for k in range(self.numKernels):
                kernel = self.params[:,:,d,k]
                r = convolve2d(inputs[:,:,d], kernel, 'valid')
                output[:,:,k] += r
        if self.flatten:
            return np.ravel(output)
        return output
    
    def updateWeights(self, outputs, deltas, alpha):
        if self.flatten:
            deltas = np.reshape(deltas, self.outputShape) #unflatten the output array
            
        convoluted = np.zeros(self.params.shape)        
        for k in range(self.numKernels):
            for d in range(self.inputDepth):
                convoluted[:,:,d,k] += convolve2d(outputs[:,:,d],deltas[:,:,k], 'valid')
                
        self.params -= alpha * convoluted
