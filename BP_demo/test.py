from NewBP import BP
import numpy as np
np.set_printoptions(suppress=True)


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
Y = (np.array(trainY)).T

testX = [[1, 1, 2, 1, 1, 1],
        [3, 1, 1, 1, 1, 1],
        [2, 2, 1, 1, 2, 1],
        [2, 2, 2, 2, 2, 1],
        [3, 3, 3, 3, 3, 1],
        [3, 1, 1, 3, 3, 2],
        [1, 2, 1, 2, 1, 1]]
testY = [0, 0, 0, 1, 1, 1, 1]
testY = np.array(testY)

E = 0;
frequency = 10

for i in range(frequency):
    nn = BP(6, 7, 1)
    nn.fit(X, Y, 0.2, 10000)
    predictY = nn.predict(np.array(testX))
    predictY = predictY.T
    print("第"+str(i+1)+"次预测结果：")
    print(predictY)
    E += np.sum((testY - predictY)*(testY - predictY))/len(testY)

print("10次随机BP后的平均方差：")
print(E/frequency)

