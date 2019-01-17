import numpy as np
import random

class Adaline(object):

    def __init__(self, inputsNumber):
        # + 1 for bias
        self.weights = np.asarray([random.random() for i in range(0, inputsNumber + 1)])

    def getOutput(self, inputs):
        inputs = self.addBias(inputs)
        return np.array([np.dot(inputs, self.weights.T)])

    def updateWeights(self, inputs, outputs, expectedOutputs, learningRate):
        inputs = self.addBias(inputs)
        for idx, weight in enumerate(self.weights):
            self.weights[idx] += learningRate * (expectedOutputs - outputs) * inputs[idx]

    def addBias(self, inputs):
        return np.asarray(list(inputs) + [1])




