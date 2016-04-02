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
        return self.forwardPropagation(inputs)[-1][-1]

    def fit(self, inputArray, labelsArray, numEpochs, verbose=False):
        errors = []
        for e in range(numEpochs):
            error = 0
            num = 0
            for inputs, labels in zip(inputArray, labelsArray):
                outputs, activatedOutputs = self.forwardPropagation(inputs)
                deltas  = self.backwardPropagation(labels, activatedOutputs)
                self.updateWeights(activatedOutputs, deltas)
                #print(self.predict(inputs))
                error += (0.5 * (labels - activatedOutputs[-1])**2).sum()
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
        inputs = np.hstack((inputs, [1]))
        outputs = []
        activatedOutputs = []
        outputs.append(inputs)
        activatedOutputs.append(inputs)
        activatedOutput = inputs
        
        for layer in self.layers[1:]:
##            print("input:",activatedOutput)
            output = layer.calcOutput(activatedOutput)
##            print("output:", output)
##            print("---")
            if layer.biased:
                output[-1] = 1 #reset bias neuron
            activatedOutput = self.sigmoid(output)
            outputs.append(output)
            activatedOutputs.append(activatedOutput)

##        print(outputs)
        return outputs, activatedOutputs
    
    def backwardPropagation(self, labels, outputs):
        lastDelta = self.layers[-1].calcDeltaAsLast(labels, outputs[-1])
        deltas = [lastDelta]
        for i in range(len(self.layers[1:-1])):
            delta = self.layers[-2-i].calcDelta(outputs[-2-i], deltas[-1], self.layers[-1-i].params)
            deltas.append(delta)

        deltas.reverse()
##        print(deltas)
        return deltas

    def updateWeights(self, outputs, deltas):
        for i in range(len(deltas)):
            layer = self.layers[i+1]
            
            layer.params += self.alpha * np.outer(deltas[i], outputs[i])
##            print(layer.params)
            
            

    
            
        
