import numpy as np
import Adaline
import matplotlib.pyplot as plt


def readData(filename, delimiter, inputsNumber):
    data = np.genfromtxt(filename, delimiter=delimiter)
    normalizedData = (data - data.min()) / (data.max() - data.min())
    return [(np.asarray(item[:inputsNumber]), np.asarray(item[inputsNumber:])) for item in normalizedData]


iterations = 100000
inputsNumber = 1
learningRate = 0.001
data = readData('data.txt', ',', inputsNumber)
adaline = Adaline.Adaline(inputsNumber)
counter = 0
for i in range(0, iterations):
    x, y = data[counter][0], data[counter][1]
    output = adaline.getOutput(x)
    adaline.updateWeights(x, output, y, learningRate)
    counter += 1
    if counter >= len(data):
        counter = 0

x = np.arange(0.0, 1.0, 0.02)
plt.plot(x, [adaline.getOutput(np.array([value])) for value in x], 'r')
plt.plot([item[0] for item in data], [item[1] for item in data], 'b.')
plt.show()