from loadDataSet import loadDataSet
from numpy import *
x, y = loadDataSet('/Users/gong/Desktop/data.txt')
m = shape(x)[0]
print(m)