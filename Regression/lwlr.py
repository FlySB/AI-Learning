from numpy import *

def lwlr(testPoint, xArr, yArr, k = 1.0):

    xMat = mat(xArr)
    yMat = mat(yArr)

    m = shape(xMat)[0] # xMat矩阵的行数
    weights = mat(eye((m)))

    for j in range(m):

        diffMat = testPoint - xMat[j,:]
        weights[j, j] = exp(diffMat*diffMat.T / (-2.0 * k**2))

    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("该矩阵无法求逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws