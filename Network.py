from Layer import Layer
import numpy as np
import matplotlib.pyplot as plt


class Network(object):
    def __init__(self, sigmoidFunc, learningRate = 0.01):
        self.layers = []
        self.sigmoid = sigmoidFunc
        self.alpha = learningRate
        
    def addLayer(self, numNeurons, biased=False):
        if biased:
            numNeurons += 1
        if not len(self.layers):
            NewLayer = Layer(numNeurons, 0, biased)     
        else:
            paramsShape = (numNeurons,self.layers[-1].shape)
            NewLayer = Layer(numNeurons, paramsShape, biased)
            
        self.layers.append(NewLayer)            
        
    def addConvLayer(self):
        pass

    def predict(self, inputs):
        return self.forwardPropagation(inputs)[-1]

    def fit(self, inputArray, labelsArray, numEpochs, verbose=False):
        errors = []
        for e in range(numEpochs):
            error, num = 0, 0
            for inputs, labels in zip(inputArray, labelsArray):
                outputs = self.forwardPropagation(inputs)
                deltas  = self.backwardPropagation(outputs, labels)
                self.updateWeights(outputs, deltas)
                
                error += (0.5 * (labels - outputs[-1])**2).sum()
                num += 1
                
            errors.append(error/ num)
            if verbose:
                if e % (numEpochs/10) == 0: print("Epoch:", e)
                
        if verbose:
            fig = plt.figure()
            x = np.linspace(0, numEpochs, numEpochs)
            plt.plot(x, errors, 'r')
            plt.show()
            
    def forwardPropagation(self, inputs):
        inputs = np.hstack((inputs, [1])) #add bias
        outputs = []
        outputs.append(inputs)
        activatedOutput = inputs
        
        for layer in self.layers[1:]:
            output = layer.calcOutput(activatedOutput)
            activatedOutput = self.sigmoid(output)
            if layer.biased:
                activatedOutput[-1] = 1 #reset bias neuron
            outputs.append(activatedOutput)

        return outputs
    
    def backwardPropagation(self, outputs, labels):
        lastDelta = self.layers[-1].calcDeltaAsLast(labels, outputs[-1])
        deltas = [lastDelta]
        for i in range(len(self.layers[1:-1])):
            delta = self.layers[-2-i].calcDelta(outputs[-2-i], deltas[-1], self.layers[-1-i].params)
            deltas.append(delta)

        deltas.reverse()
        return deltas

    def updateWeights(self, outputs, deltas):
        for i in range(len(deltas)):
            layer = self.layers[i+1]
            layer.params -= self.alpha * np.outer(deltas[i], outputs[i])
            
            

    
            
        
