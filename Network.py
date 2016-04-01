from Layer import Layer
import numpy as np


class Network(object):
    def __init__(self, sigmoidFunc, learningRate = 0.01):
        #Input layer isn't represented by a Layer Object
        #Even though it's added as if it were
        self.inputSize = 0 
        self.layers = []
        self.sigmoid = sigmoidFunc
        self.alpha = learningRate
        
    def addLayer(self, numNeurons):
        if not len(self.layers):
            NewLayer = Layer(numNeurons, 0)            
        else:
            paramsShape = (numNeurons,self.layers[-1].shape)
            NewLayer = Layer(numNeurons, paramsShape)
        self.layers.append(NewLayer)            
        
    def addConvLayer(self):
        pass

    def predict(self, inputs):
        activatedOutput = inputs        
        for layer in self.layers[1:]:
            output = layer.calcOutput(activatedOutput)
            activatedOutput = self.sigmoid(output)            
        return output

    def fit(self, inputArray, labelsArray, numEpochs):
        for e in range(numEpochs):
            for inputs, labels in zip(inputArray, labelsArray):
                outputs = self.forwardPropagation(inputs)
                deltas  = self.backwardPropagation(labels, outputs)
                self.updateWeights(outputs, deltas)
                print(self.predict(inputs))
            
    def forwardPropagation(self, inputs):
        outputs = []
        outputs.append(inputs)
        activatedOutput = inputs
        
        for layer in self.layers[1:]:
            output = layer.calcOutput(activatedOutput)
            activatedOutput = self.sigmoid(output)
            outputs.append(activatedOutput)
            
        return outputs
    
    def backwardPropagation(self, labels, outputs):
        lastDelta = self.layers[-1].calcDeltaAsLast(labels, outputs[-1])
        deltas = [lastDelta]
        for i in range(len(self.layers[1:-1])):
            delta = self.layers[-2-i].calcDelta(outputs[-2-i], deltas[-1])
            deltas.append(delta)

        deltas.reverse()
        return deltas

    def updateWeights(self, outputs, deltas):
        for i in range(len(deltas)):
            layer = self.layers[i+1]
            layer.params += self.alpha * np.outer(deltas[i], outputs[i]) 
            
            

    
            
        
