import numpy as np

class FullyConnectedLayer(object):
    def __init__(self, neurons):
        self.params = np.random.rand(neurons)
    def forwardProp(self, array):
        return array.dot(self.params)
    def backProp(self):
        pass
        
