import numpy as np

class Layer(object):
    def __init__(self, shape, paramShape, biased=False):
        self.shape = shape
        self.biased = biased
        self.params =  np.random.uniform(-1.0,1.0,size=paramShape)
    def calcOutput(self, inputs):
        return self.params.dot(inputs)
    def calcDeltaAsLast(self, target, actual):
        return (target - actual) * actual * (1 - actual)
    def calcDelta(self, output, previousDelta, previousParams):
##        print(output.shape, previousDelta.shape, self.params.shape)
        return (previousDelta * previousParams).sum() * output * (1 - output)
