from numpy import *
from lwlr import lwlr
def lwlrTest(testArr, xArr, yArr, k = 1.0):

    m = shape(testArr)[0]
    yHat = zeros(m)

    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)


    return yHat