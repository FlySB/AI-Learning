from loadDataSet import loadDataSet
from standRegres import standRegres
from numpy import *
import matplotlib.pyplot as plt

def regression1():
    xArr, yArr = loadDataSet("/Users/gong/Desktop/data.txt")
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres(xArr, yArr)

    fig = plt.figure()
    ax = fig.add_subplot(111)  # add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块

    ax.scatter(xMat[:, 1].flatten().tolist(), yMat.T[:, 0].flatten().tolist())  # scatter 的x是xMat中的第二列，y是yMat的第一列
    xCopy = xMat.copy()
    xCopy.sort(0)   #numpy里的sort函数：'0'按列排 '1'按行排
    print(xCopy)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
