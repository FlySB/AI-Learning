from  BP import NeuralNetwork
import numpy as np

nn = NeuralNetwork([6, 10, 1], 'logistic')
trainX = [[1, 1, 1, 1, 1, 1],
          [2, 1, 2, 1, 1, 1],
          [2, 1, 1, 1, 1, 1],
          [1, 2, 1, 1, 2, 2],
          [2, 2, 1, 2, 2, 2],
          [1, 3, 3, 1, 3, 2],
          [3, 2, 2, 2, 1, 1],
          [2, 2, 1, 1, 2, 2],
          [3, 1, 1, 3, 3, 1],
          [1, 1, 2, 2, 2, 1]]
X = np.array(trainX)
trainY = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y = np.array(trainY)
nn.fit(X, y)
testX = [[1, 1, 2, 1, 1, 1],
        [3, 1, 1, 1, 1, 1],
        [2, 2, 1, 1, 2, 1],
        [2, 2, 2, 2, 2, 1],
        [3, 3, 3, 3, 3, 1],
        [3, 1, 1, 3, 3, 2],
        [1, 2, 1, 2, 1, 1]]
for i in testX:
  print(i, nn.predict(i))