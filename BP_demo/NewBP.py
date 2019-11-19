import numpy as np
import random
import math

# sigmoid函数
def logistic(iX,dimension):#iX is a matrix with a dimension
    if dimension==1:
        for i in range(len(iX)):
            iX[i] = 1 / (1 + math.exp(-iX[i]))
    else:
        for i in range(len(iX)):
            iX[i] = logistic(iX[i],dimension-1)
    return iX

class BP:
    def __init__(self, InVecter, HideNode, OutVecter):
        self.InVecter = InVecter
        self.HideNode = HideNode
        self.OutVecter = OutVecter

        self.theta = [random.random() for i in range(OutVecter)]

        self.gamma = [random.random() for i in range(HideNode)]

        self.v = [[random.random() for i in range(HideNode)] for j in range(InVecter)]

        self.w = [[random.random() for i in range(OutVecter)] for j in range(HideNode)]

    def fit(self, X, Y, learning_rate, epochs):

        m, n = np.shape(X)
        for k in range(epochs):
            sunE = 0
            for i in range(m):
                alpha = np.dot(X[i],self.v)
                b = logistic(alpha - self.gamma, 1)
                beta = np.dot(b, self.w)
                predictY = logistic(beta - self.theta, 1)
                E = sum((predictY - Y[i])*(predictY - Y[i]))/2.0
                sunE += E

                g = predictY*(1 - predictY)*(Y[i] - predictY)
                e = b*(1-b)*((np.dot(self.w,g.T)).T)
                self.w += learning_rate * np.dot(b.reshape((self.HideNode,1)), g.reshape((1, self.OutVecter)))
                self.theta -= learning_rate*g
                self.v += learning_rate*np.dot(X[i].reshape((self.InVecter,1)), e.reshape((1, self.HideNode)))
                self.gamma -= learning_rate*e

    def predict(self, TestX):
        alpha = np.dot(TestX, self.v)
        b = logistic(alpha - self.gamma, 2)
        beta = np.dot(b, self.w)
        predictY = logistic(beta - self.theta, 2)
        return predictY
