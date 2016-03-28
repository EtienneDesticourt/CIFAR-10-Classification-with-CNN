import numpy as np
from scipy.signal import convolve2d

class ConvolutionalLayer(object):
    def __init__(self, inputShape, numKernels, kernelSize, learningRate=0.1):        
        self.numKernels = numKernels
        self.kernelSize = kernelSize
        self.learningRate = learningRate
        #Initialize weights
        x, y, depth = inputShape
        self.inputDepth = depth
        self.params     = np.random.rand(kernelSize, kernelSize, depth, numKernels)
        #Calculate output shape
        self.outputShape = ( x - kernelSize + 1,
                             y - kernelSize + 1,
                             numKernels)
        
    def forwardProp(self, image):
        output = np.zeros(self.outputShape)
        for d in range(self.inputDepth):
            for k in range(self.numKernels):
                kernel = self.params[:,:,d,k]
                r = convolve2d(image, kernel, 'valid')
                output[:,:,k] += r
        return output
    def backProp(self):
        pass
