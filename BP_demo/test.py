from NewBP import BP
import numpy as np
np.set_printoptions(suppress=True)

nn = BP(6, 7, 1)
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
y = (np.array(trainY)).T

nn.fit(X, y, 0.2, 10000)
testX = [[1, 1, 2, 1, 1, 1],
        [3, 1, 1, 1, 1, 1],
        [2, 2, 1, 1, 2, 1],
        [2, 2, 2, 2, 2, 1],
        [3, 3, 3, 3, 3, 1],
        [3, 1, 1, 3, 3, 2],
        [1, 2, 1, 2, 1, 1]]

print(nn.predict(X))