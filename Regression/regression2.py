from loadDataSet import loadDataSet
from lwlrTest import lwlrTest
from numpy import *
import matplotlib.pyplot as plt
def regression2():
    xArr, yArr = loadDataSet('/Users/gong/Desktop/data.txt')
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s = 2, c = 'red')
    plt.show()