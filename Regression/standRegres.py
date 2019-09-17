from numpy import *
def standRegres(xArr, yArr):

    xMat = mat(xArr)
    yMat = mat(yArr).T

    xTx = xMat.T * xMat

    if linalg.det(xTx) == 0:
        print("矩阵无法求逆")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws