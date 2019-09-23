from numpy import *

def lwlr(testPoint, xArr, yArr, k = 1.0):

    xMat = mat(xArr)
    yMat = mat(yArr).T

    m = shape(xMat)[0] # xMat矩阵的行数
    weights = mat(eye((m))) # 返回SIZE为m的单位矩阵

    for j in range(m):

        diffMat = testPoint - xMat[j,:]
        weights[j, j] = exp(diffMat*diffMat.T / (-2.0 * k**2))

    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("该矩阵无法求逆")
        return

    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws